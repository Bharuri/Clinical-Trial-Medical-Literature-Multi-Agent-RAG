"""
Text chunking with configurable size, overlap, and boundary preservation.

Splits narrative text into chunks while respecting sentence boundaries,
table boundaries, and configured overlap parameters.
"""

import re
from typing import List, Optional

import nltk

# Try to load tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class ChunkingError(Exception):
    """Raised when chunking fails."""
    pass


class Chunk:
    """Represents a single content chunk."""

    def __init__(
        self,
        content: str,
        chunk_type: str,
        page_number: int,
        source_file: str,
        section_header: Optional[str] = None,
    ):
        """
        Initialize a chunk.

        Args:
            content: Chunk text content
            chunk_type: 'text' or 'table'
            page_number: Source page number
            source_file: Source PDF filename
            section_header: Section header/title
        """
        self.content = content
        self.chunk_type = chunk_type
        self.page_number = page_number
        self.source_file = source_file
        self.section_header = section_header or "Untitled"

    def __repr__(self) -> str:
        """String representation."""
        content_preview = self.content[:50].replace("\n", " ")
        return (
            f"Chunk(type={self.chunk_type}, page={self.page_number}, "
            f"content='{content_preview}...', header='{self.section_header}')"
        )


class TextChunker:
    """Chunks narrative text with overlap while preserving sentence boundaries."""

    def __init__(
        self,
        chunk_size_tokens: int = 750,
        chunk_overlap_tokens: int = 150,
        language: str = "english",
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size_tokens: Target chunk size in tokens (approximate)
            chunk_overlap_tokens: Overlap between chunks in tokens
            language: Language for tokenization (for NLTK)

        Raises:
            ChunkingError: If parameters are invalid
        """
        if chunk_size_tokens <= 0:
            raise ChunkingError(f"chunk_size_tokens must be positive, got {chunk_size_tokens}")
        if chunk_overlap_tokens < 0:
            raise ChunkingError(f"chunk_overlap_tokens must be non-negative, got {chunk_overlap_tokens}")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ChunkingError(
                f"chunk_overlap_tokens ({chunk_overlap_tokens}) must be less than "
                f"chunk_size_tokens ({chunk_size_tokens})"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.language = language

        # Empirical ratio: ~4 characters per token (varies by text)
        self.avg_chars_per_token = 4

    def chunk_narrative_text(
        self,
        text: str,
        source_file: str,
        page_number: int,
        section_header: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Split narrative text into chunks with overlap.

        Args:
            text: Narrative text to chunk
            source_file: Source PDF filename
            page_number: Source page number
            section_header: Section header/title

        Returns:
            List of Chunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        if not text or not text.strip():
            return []

        try:
            chunks = []
            sentences = self._split_into_sentences(text)

            if not sentences:
                return []

            # Group sentences into chunks
            current_chunk_sentences: List[str] = []
            current_chunk_tokens = 0

            for sentence in sentences:
                sentence_tokens = self._estimate_tokens(sentence)

                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_chunk_tokens + sentence_tokens > self.chunk_size_tokens and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append(
                        Chunk(
                            content=chunk_text.strip(),
                            chunk_type="text",
                            page_number=page_number,
                            source_file=source_file,
                            section_header=section_header,
                        )
                    )

                    # Keep some sentences from the current chunk for overlap
                    overlap_buffer = self._get_overlap_buffer(current_chunk_sentences)
                    current_chunk_sentences = overlap_buffer
                    current_chunk_tokens = sum(self._estimate_tokens(s) for s in overlap_buffer)

                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens

            # Add final chunk
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    Chunk(
                        content=chunk_text.strip(),
                        chunk_type="text",
                        page_number=page_number,
                        source_file=source_file,
                        section_header=section_header,
                    )
                )

            return chunks

        except Exception as e:
            raise ChunkingError(f"Failed to chunk text: {e}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        try:
            # Use NLTK's punkt tokenizer
            sentences = nltk.sent_tokenize(text, language=self.language)
            # Clean up sentences
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback: simple split on periods/exclamation/question marks
            # Handle common abbreviations
            text = re.sub(r"(\w)\.(\w)", r"\1.<SEP>\2", text)  # Keep abbreviations
            sentences = re.split(r"[.!?]+", text)
            text = text.replace("<SEP>", ".")  # Restore abbreviations
            return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses empirical ratio: ~4 characters per token.
        Can be refined with actual tokenizer later.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        # Simple heuristic: split by whitespace
        words = text.split()
        # Rough estimate: average word is ~5 chars + 1 space
        # So roughly 6 chars per word/token
        return max(1, len(words))

    def _get_overlap_buffer(self, sentences: List[str]) -> List[str]:
        """
        Get sentences to keep for overlap with next chunk.

        Args:
            sentences: All sentences in current chunk

        Returns:
            Sentences to keep for overlap (usually last few)
        """
        overlap_tokens = 0
        overlap_sentences = []

        # Go backwards through sentences until we have enough overlap
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.chunk_overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens

        return overlap_sentences

    def create_table_chunk(
        self,
        table_markdown: str,
        source_file: str,
        page_number: int,
        section_header: Optional[str] = None,
    ) -> Chunk:
        """
        Create a chunk for a complete table (tables are never split).

        Args:
            table_markdown: Table in Markdown format
            source_file: Source PDF filename
            page_number: Source page number
            section_header: Section header/title

        Returns:
            Chunk object for the table
        """
        return Chunk(
            content=table_markdown,
            chunk_type="table",
            page_number=page_number,
            source_file=source_file,
            section_header=section_header,
        )
