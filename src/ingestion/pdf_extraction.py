"""
PDF extraction and text/table separation.

Handles reading medical literature PDFs, extracting narrative text and tables
separately, and removing repetitive headers/footers/page numbers.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails."""
    pass


class PDFContent:
    """Represents extracted content from a single PDF page."""

    def __init__(
        self,
        page_number: int,
        narrative_text: str,
        tables: List[str],
        section_header: Optional[str] = None,
    ):
        """
        Initialize PDF page content.

        Args:
            page_number: Page number (1-indexed)
            narrative_text: Extracted narrative text (headers/footers removed)
            tables: List of Markdown-formatted tables
            section_header: Current section header/title, if available
        """
        self.page_number = page_number
        self.narrative_text = narrative_text
        self.tables = tables
        self.section_header = section_header


class PDFExtractor:
    """Extracts text and tables from PDF files."""

    # Lower than pdfplumber's default (3) so tightly spaced PDF glyphs are not merged into one word.
    TEXT_X_TOLERANCE = 1.5

    # Common repetitive patterns in medical documents
    HEADER_FOOTER_PATTERNS = [
        r"^page\s+\d+\s*$",
        r"^doi:\s*10\.\S+\s*$",
        r"^\d+\s*$",
        r"^clinical\s+trial",
        r"^supplementary\s+material",
        r"^\s*-\s*\d+\s*-\s*$",
    ]

    def __init__(self, min_text_length: int = 10):
        """
        Initialize PDF extractor.

        Args:
            min_text_length: Minimum character length for text blocks to keep
        """
        self.min_text_length = min_text_length

    def extract_pdf(self, pdf_path: Path) -> List[PDFContent]:
        """
        Extract all pages from a PDF, separating text and tables.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PDFContent objects, one per page

        Raises:
            PDFExtractionError: If file cannot be read
        """
        if not pdf_path.exists():
            raise PDFExtractionError(f"PDF file not found: {pdf_path}")

        try:
            results = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    content = self._extract_page(page, page_idx)
                    if content.narrative_text or content.tables:
                        results.append(content)

            return results
        except Exception as e:
            raise PDFExtractionError(f"Failed to extract PDF {pdf_path}: {e}")

    def _extract_page(self, page: "pdfplumber.PDF.pages", page_number: int) -> PDFContent:
        """
        Extract text and tables from a single PDF page.

        Args:
            page: PDF page object from pdfplumber
            page_number: Page number (1-indexed)

        Returns:
            PDFContent with extracted narrative and tables
        """
        # Extract tables first (to identify regions to skip)
        tables = self._extract_tables(page)
        table_regions = self._get_table_regions(page, tables)

        # Extract narrative text, avoiding table regions
        narrative = self._extract_narrative_text(page, table_regions)
        narrative = self._clean_narrative(narrative)

        # Detect section header (usually from top of page or previous headings)
        section_header = self._detect_section_header(page)

        return PDFContent(
            page_number=page_number,
            narrative_text=narrative,
            tables=tables,
            section_header=section_header,
        )

    def _extract_tables(self, page: "pdfplumber.PDF.pages") -> List[str]:
        """
        Extract all tables from a page and convert to Markdown.

        Args:
            page: PDF page object

        Returns:
            List of Markdown-formatted tables (one string per table)
        """
        markdown_tables = []

        try:
            tables = page.extract_tables()
            if not tables:
                return markdown_tables

            for table in tables:
                if not table or len(table) == 0:
                    continue

                md_table = self._table_to_markdown(table)
                if md_table:
                    markdown_tables.append(md_table)

        except Exception:
            # Some pages may not have parseable tables
            pass

        return markdown_tables

    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        """
        Convert table (list of lists) to Markdown format.

        Args:
            table: Table as list of rows, each row is list of cells

        Returns:
            Markdown-formatted table string
        """
        if not table or len(table) == 0:
            return ""

        # Ensure all rows have same number of columns
        num_cols = max(len(row) for row in table) if table else 0
        if num_cols == 0:
            return ""

        # Normalize rows to have same column count
        normalized = []
        for row in table:
            normalized_row = list(row) + [""] * (num_cols - len(row))
            normalized.append(normalized_row[:num_cols])

        # Create header separator
        separator = "|" + "|".join(["---"] * num_cols) + "|"

        # Build Markdown
        lines = []
        for idx, row in enumerate(normalized):
            # Escape pipe characters in cells
            escaped_row = [str(cell).replace("|", "\\|") for cell in row]
            md_row = "|" + "|".join(escaped_row) + "|"
            lines.append(md_row)

            # Insert separator after first row (header)
            if idx == 0:
                lines.append(separator)

        return "\n".join(lines)

    @staticmethod
    def _get_table_regions(page: "pdfplumber.PDF.pages", tables: List[str]) -> List[Tuple]:
        """
        Get bounding box regions of all tables on page.

        Used to avoid extracting table content as narrative text.

        Args:
            page: PDF page object
            tables: List of extracted tables

        Returns:
            List of (x0, top, x1, bottom) tuples for table regions
        """
        try:
            return [tuple(t.bbox) for t in page.find_tables()]
        except Exception:
            return []

    def _extract_narrative_text(
        self, page: "pdfplumber.PDF.pages", table_regions: List[Tuple]
    ) -> str:
        """
        Extract narrative text from page, avoiding table regions.

        Args:
            page: PDF page object
            table_regions: List of table bounding boxes to exclude

        Returns:
            Cleaned narrative text
        """
        try:
            if table_regions:
                def outside_tables(obj) -> bool:
                    cx = (obj["x0"] + obj["x1"]) / 2
                    cy = (obj["top"] + obj["bottom"]) / 2
                    return not any(
                        x0 <= cx <= x1 and top <= cy <= bottom
                        for x0, top, x1, bottom in table_regions
                    )

                page = page.filter(outside_tables)

            text = page.extract_text(x_tolerance=self.TEXT_X_TOLERANCE)

            if not text:
                return ""

            return text

        except Exception:
            return ""

    def _clean_narrative(self, text: str) -> str:
        """
        Remove headers, footers, page numbers, and other repetitive content.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Split into lines for processing
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                cleaned_lines.append("")
                continue

            # Skip if matches header/footer patterns (case-insensitive)
            if self._is_header_or_footer(stripped):
                continue

            # Skip very short lines that are likely page numbers or artifacts
            if len(stripped) < self.min_text_length and re.match(r"^[\d\s-]+$", stripped):
                continue

            cleaned_lines.append(line)

        # Remove excessive empty lines (keep max 2 consecutive)
        result_lines = []
        empty_count = 0
        for line in cleaned_lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        text = "\n".join(result_lines).strip()

        # Remove common footer/header phrases
        text = re.sub(r"(?i)^supplementary\s+materials?\s*\n", "", text)
        text = re.sub(r"(?i)\n(supplementary\s+materials?|appendix)\s*$", "", text)

        return text

    def _is_header_or_footer(self, line: str) -> bool:
        """
        Check if a line is a header, footer, or page number.

        Args:
            line: Text line to check

        Returns:
            True if line is header/footer
        """
        line_lower = line.lower()

        for pattern in self.HEADER_FOOTER_PATTERNS:
            if re.search(pattern, line_lower):
                return True

        return False

    @staticmethod
    def _detect_section_header(page: "pdfplumber.PDF.pages") -> Optional[str]:
        """
        Attempt to detect the section header/title on this page.

        Args:
            page: PDF page object

        Returns:
            Section header string, or None
        """
        try:
            text = page.extract_text(x_tolerance=PDFExtractor.TEXT_X_TOLERANCE)
            if not text:
                return None

            # Get first non-empty line as potential header
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                # Return first line if it looks like a header (short, mostly uppercase)
                first = lines[0]
                if len(first) < 100 and not re.match(r"^\d+\.\d+", first):
                    return first

        except Exception:
            pass

        return None
