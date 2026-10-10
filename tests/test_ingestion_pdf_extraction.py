"""
Tests for PDF extraction module.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

from src.ingestion.pdf_extraction import PDFExtractor, PDFExtractionError, PDFContent


@pytest.fixture
def sample_pdf() -> Generator[Path, None, None]:
    """Create a simple test PDF with text and tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "test.pdf"

        # Create a simple PDF using reportlab
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            c = canvas.Canvas(str(pdf_path), pagesize=letter)

            # Add some text
            c.drawString(100, 750, "Clinical Trial Report")
            c.drawString(100, 720, "This is a sample medical document with narrative text.")
            c.drawString(100, 690, "It contains important clinical information.")

            # Add a simple table representation
            c.drawString(100, 650, "Results Table:")
            c.drawString(100, 630, "Patient ID | Age | Treatment")
            c.drawString(100, 610, "001 | 45 | Drug A")
            c.drawString(100, 590, "002 | 52 | Drug B")

            # Add footer
            c.drawString(100, 50, "Page 1")

            c.save()
        except ImportError:
            # If reportlab not available, skip this fixture
            pytest.skip("reportlab not available for PDF creation")

        yield pdf_path


@pytest.fixture
def multipage_pdf() -> Generator[Path, None, None]:
    """Create a multi-page test PDF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "multipage.pdf"

        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            c = canvas.Canvas(str(pdf_path), pagesize=letter)

            # Page 1
            c.drawString(100, 750, "Section 1: Introduction")
            c.drawString(100, 720, "This is the introduction section.")
            c.showPage()

            # Page 2
            c.drawString(100, 750, "Section 2: Methods")
            c.drawString(100, 720, "This describes the methodology.")
            c.drawString(100, 700, "Page 2")
            c.showPage()

            # Page 3
            c.drawString(100, 750, "Section 3: Results")
            c.drawString(100, 720, "Results are shown here.")
            c.showPage()

            c.save()
        except ImportError:
            pytest.skip("reportlab not available")

        yield pdf_path


class TestPDFExtractor:
    """Test PDF extraction functionality."""

    def test_extractor_initialization(self) -> None:
        """Should initialize successfully."""
        extractor = PDFExtractor()
        assert extractor is not None
        assert extractor.min_text_length > 0

    def test_extract_nonexistent_pdf(self) -> None:
        """Should raise error for nonexistent PDF."""
        extractor = PDFExtractor()
        with pytest.raises(PDFExtractionError, match="PDF file not found"):
            extractor.extract_pdf(Path("/nonexistent/file.pdf"))

    def test_extract_pdf_returns_list(self, sample_pdf: Path) -> None:
        """Should return list of PDFContent objects."""
        extractor = PDFExtractor()
        results = extractor.extract_pdf(sample_pdf)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, PDFContent) for r in results)

    def test_pdf_content_structure(self, sample_pdf: Path) -> None:
        """PDFContent should have required attributes."""
        extractor = PDFExtractor()
        results = extractor.extract_pdf(sample_pdf)

        if results:
            content = results[0]
            assert hasattr(content, "page_number")
            assert hasattr(content, "narrative_text")
            assert hasattr(content, "tables")
            assert hasattr(content, "section_header")
            assert isinstance(content.page_number, int)
            assert isinstance(content.narrative_text, str)
            assert isinstance(content.tables, list)

    def test_multipage_pdf(self, multipage_pdf: Path) -> None:
        """Should extract content from multiple pages."""
        extractor = PDFExtractor()
        results = extractor.extract_pdf(multipage_pdf)

        # Should have at least 3 pages
        assert len(results) >= 3

    def test_clean_narrative_removes_headers(self) -> None:
        """Should remove page numbers and footer patterns."""
        extractor = PDFExtractor()

        text_with_footer = "Some content\nPage 1\nMore content"
        cleaned = extractor._clean_narrative(text_with_footer)

        assert "Page 1" not in cleaned or cleaned.count("Page") == 0
        assert "Some content" in cleaned

    def test_table_to_markdown_conversion(self) -> None:
        """Should convert table to Markdown format."""
        extractor = PDFExtractor()

        table = [
            ["Header 1", "Header 2"],
            ["Cell 1", "Cell 2"],
            ["Cell 3", "Cell 4"],
        ]

        markdown = extractor._table_to_markdown(table)

        assert "Header 1" in markdown
        assert "Cell 1" in markdown
        assert "|" in markdown
        assert "---" in markdown

    def test_table_to_markdown_empty(self) -> None:
        """Should handle empty tables."""
        extractor = PDFExtractor()

        markdown = extractor._table_to_markdown([])
        assert markdown == ""

        markdown = extractor._table_to_markdown([[]])
        assert markdown == ""

    def test_is_header_or_footer(self) -> None:
        """Should identify common headers and footers."""
        extractor = PDFExtractor()

        assert extractor._is_header_or_footer("page 1")
        assert extractor._is_header_or_footer("Page 5")
        assert extractor._is_header_or_footer("DOI: 10.1234/example")

        assert not extractor._is_header_or_footer("This is normal text")

    def test_extract_with_minimal_text_length(self) -> None:
        """Should respect min_text_length parameter."""
        extractor = PDFExtractor(min_text_length=50)
        assert extractor.min_text_length == 50

    def test_pdf_content_with_section_header(self, sample_pdf: Path) -> None:
        """Should detect section headers from PDF content."""
        extractor = PDFExtractor()
        results = extractor.extract_pdf(sample_pdf)

        if results:
            # Section header should be detected or set to default
            content = results[0]
            assert content.section_header is not None


class TestTextToleranceSetting:
    """Word-merge fix: pdfplumber must be called with the lower x_tolerance."""

    def test_narrative_extraction_uses_x_tolerance(self) -> None:
        from unittest.mock import MagicMock

        page = MagicMock()
        page.extract_text.return_value = "andacceleratingsystematicreview"
        extractor = PDFExtractor()

        text = extractor._extract_narrative_text(page, [])

        page.extract_text.assert_called_once_with(x_tolerance=PDFExtractor.TEXT_X_TOLERANCE)
        assert text == "andacceleratingsystematicreview"

    def test_section_header_detection_uses_x_tolerance(self) -> None:
        from unittest.mock import MagicMock

        page = MagicMock()
        page.extract_text.return_value = "Introduction\nBody text here."

        header = PDFExtractor._detect_section_header(page)

        page.extract_text.assert_called_once_with(x_tolerance=PDFExtractor.TEXT_X_TOLERANCE)
        assert header == "Introduction"

    def test_tolerance_is_below_pdfplumber_default(self) -> None:
        assert PDFExtractor.TEXT_X_TOLERANCE < 3
