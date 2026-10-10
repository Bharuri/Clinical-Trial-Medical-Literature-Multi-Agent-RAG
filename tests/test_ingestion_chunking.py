"""
Tests for text chunking module.
"""

import pytest

from src.ingestion.chunking import TextChunker, Chunk, ChunkingError


class TestChunk:
    """Test Chunk data structure."""

    def test_chunk_initialization(self) -> None:
        """Should initialize with required attributes."""
        chunk = Chunk(
            content="Test content",
            chunk_type="text",
            page_number=1,
            source_file="test.pdf",
            section_header="Introduction",
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_type == "text"
        assert chunk.page_number == 1
        assert chunk.source_file == "test.pdf"
        assert chunk.section_header == "Introduction"

    def test_chunk_default_header(self) -> None:
        """Should use default header if not provided."""
        chunk = Chunk(
            content="Test",
            chunk_type="table",
            page_number=1,
            source_file="test.pdf",
        )

        assert chunk.section_header == "Untitled"

    def test_chunk_repr(self) -> None:
        """Should have readable string representation."""
        chunk = Chunk(
            content="This is a longer test content that should be truncated",
            chunk_type="text",
            page_number=1,
            source_file="test.pdf",
        )

        repr_str = repr(chunk)
        assert "Chunk" in repr_str
        assert "text" in repr_str
        assert "page=1" in repr_str


class TestTextChunker:
    """Test text chunking functionality."""

    def test_chunker_initialization(self) -> None:
        """Should initialize with valid parameters."""
        chunker = TextChunker(chunk_size_tokens=750, chunk_overlap_tokens=150)
        assert chunker.chunk_size_tokens == 750
        assert chunker.chunk_overlap_tokens == 150

    def test_chunker_invalid_chunk_size(self) -> None:
        """Should reject invalid chunk size."""
        with pytest.raises(ChunkingError, match="chunk_size_tokens must be positive"):
            TextChunker(chunk_size_tokens=-1)

        with pytest.raises(ChunkingError, match="chunk_size_tokens must be positive"):
            TextChunker(chunk_size_tokens=0)

    def test_chunker_invalid_overlap(self) -> None:
        """Should reject invalid overlap."""
        with pytest.raises(ChunkingError, match="chunk_overlap_tokens must be non-negative"):
            TextChunker(chunk_overlap_tokens=-1)

    def test_chunker_overlap_greater_than_chunk_size(self) -> None:
        """Should reject overlap >= chunk size."""
        with pytest.raises(ChunkingError, match="must be less than"):
            TextChunker(chunk_size_tokens=100, chunk_overlap_tokens=100)

        with pytest.raises(ChunkingError, match="must be less than"):
            TextChunker(chunk_size_tokens=100, chunk_overlap_tokens=150)

    def test_chunk_empty_text(self) -> None:
        """Should return empty list for empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_narrative_text(
            text="",
            source_file="test.pdf",
            page_number=1,
        )

        assert chunks == []

    def test_chunk_single_sentence(self) -> None:
        """Should create single chunk for short text."""
        chunker = TextChunker()
        text = "This is a short sentence."

        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="test.pdf",
            page_number=1,
            section_header="Introduction",
        )

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_type == "text"
        assert chunks[0].source_file == "test.pdf"
        assert chunks[0].page_number == 1

    def test_chunk_multiple_sentences_with_overlap(self) -> None:
        """Should create multiple chunks with overlap for long text."""
        chunker = TextChunker(chunk_size_tokens=50, chunk_overlap_tokens=10)

        # Create a text long enough to require chunking
        sentences = [f"This is sentence number {i}." for i in range(1, 30)]
        text = " ".join(sentences)

        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="test.pdf",
            page_number=1,
        )

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should be a Chunk object
        assert all(isinstance(c, Chunk) for c in chunks)

        # Chunks should have metadata
        for chunk in chunks:
            assert chunk.source_file == "test.pdf"
            assert chunk.page_number == 1
            assert chunk.chunk_type == "text"

    def test_chunk_preserves_metadata(self) -> None:
        """Should preserve source metadata in chunks."""
        chunker = TextChunker()
        text = "Sample text for chunking with metadata preservation."

        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="important.pdf",
            page_number=42,
            section_header="Methods",
        )

        for chunk in chunks:
            assert chunk.source_file == "important.pdf"
            assert chunk.page_number == 42
            assert chunk.section_header == "Methods"

    def test_estimate_tokens(self) -> None:
        """Should estimate token count from text."""
        chunker = TextChunker()

        # Simple word count approximation
        one_token = chunker._estimate_tokens("word")
        assert one_token >= 1

        five_tokens = chunker._estimate_tokens("one two three four five")
        assert five_tokens == 5

    def test_split_into_sentences(self) -> None:
        """Should split text into sentences."""
        chunker = TextChunker()

        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._split_into_sentences(text)

        assert len(sentences) >= 3
        assert "First sentence" in sentences[0]

    def test_get_overlap_buffer(self) -> None:
        """Should create overlap buffer from sentence list."""
        chunker = TextChunker(chunk_size_tokens=50, chunk_overlap_tokens=20)

        sentences = ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four."]
        buffer = chunker._get_overlap_buffer(sentences)

        # Buffer should be non-empty
        assert len(buffer) > 0
        assert len(buffer) <= len(sentences)

    def test_create_table_chunk(self) -> None:
        """Should create a table chunk without splitting."""
        chunker = TextChunker()

        table_md = "| Header 1 | Header 2 |\n|---|---|\n| Cell 1 | Cell 2 |"
        chunk = chunker.create_table_chunk(
            table_markdown=table_md,
            source_file="test.pdf",
            page_number=5,
            section_header="Results",
        )

        assert chunk.content == table_md
        assert chunk.chunk_type == "table"
        assert chunk.page_number == 5
        assert chunk.source_file == "test.pdf"
        assert chunk.section_header == "Results"

    def test_chunk_whitespace_handling(self) -> None:
        """Should handle excess whitespace properly."""
        chunker = TextChunker()

        text = "  First sentence.   \n\n   Second sentence.   \n   Third sentence.  "
        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="test.pdf",
            page_number=1,
        )

        # Should create chunks without excess whitespace
        for chunk in chunks:
            assert chunk.content == chunk.content.strip()
            assert "\n\n\n" not in chunk.content  # No triple newlines

    def test_chunk_type_always_set(self) -> None:
        """Chunks should always have correct chunk_type."""
        chunker = TextChunker()

        text = "This is test content for chunk type validation."
        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="test.pdf",
            page_number=1,
        )

        assert all(chunk.chunk_type == "text" for chunk in chunks)

        table_chunk = chunker.create_table_chunk(
            table_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            source_file="test.pdf",
            page_number=1,
        )
        assert table_chunk.chunk_type == "table"

    def test_large_text_chunking(self) -> None:
        """Should handle large text input."""
        chunker = TextChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)

        # Create a very long text
        sentences = [f"This is sentence number {i} with some content." for i in range(1, 100)]
        text = " ".join(sentences)

        chunks = chunker.chunk_narrative_text(
            text=text,
            source_file="large.pdf",
            page_number=1,
        )

        # 99 sentences of ~8 words each with a 100-token chunk size and 20-token overlap yields 10 chunks
        assert len(chunks) >= 10

        # All content should be represented (with possible overlap)
        combined = " ".join(c.content for c in chunks)
        assert len(combined) >= len(text)  # May be longer due to overlap
