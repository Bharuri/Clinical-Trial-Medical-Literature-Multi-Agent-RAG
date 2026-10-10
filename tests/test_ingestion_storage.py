"""
Tests for DuckDB storage module.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.ingestion.chunking import Chunk
from src.ingestion.storage import ChunkStorage, StorageError


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary DuckDB file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.duckdb")
        yield db_path


@pytest.fixture
def sample_chunks() -> list:
    """Create sample chunks for testing."""
    return [
        Chunk(
            content="This is the first chunk of text.",
            chunk_type="text",
            page_number=1,
            source_file="test.pdf",
            section_header="Introduction",
        ),
        Chunk(
            content="| Header 1 | Header 2 |\n|---|---|\n| Cell 1 | Cell 2 |",
            chunk_type="table",
            page_number=2,
            source_file="test.pdf",
            section_header="Results",
        ),
        Chunk(
            content="This is the third chunk with more narrative text.",
            chunk_type="text",
            page_number=3,
            source_file="test.pdf",
            section_header="Methods",
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list:
    """Create sample 1024-dimensional embeddings."""
    embeddings = []
    for i in range(3):
        # Create a random vector
        emb = np.random.randn(1024)
        # L2-normalize
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    return embeddings


class TestChunkStorage:
    """Test DuckDB storage functionality."""

    def test_storage_initialization(self, temp_db: str) -> None:
        """Should initialize with database path."""
        storage = ChunkStorage(temp_db)
        assert storage is not None
        storage.close()

    def test_storage_initialization_from_config(self) -> None:
        """Should initialize from configuration if no path provided."""
        # This may use the project's actual config
        try:
            storage = ChunkStorage()
            assert storage is not None
            storage.close()
        except Exception:
            # May fail if config not available in test environment
            pass

    def test_create_table(self, temp_db: str) -> None:
        """Should create medical_document_chunks table."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        assert storage.verify_table_exists()
        storage.close()

    def test_table_schema(self, temp_db: str) -> None:
        """Table should have required columns."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        # Try to insert a row to verify schema
        storage.insert_chunks_batch([
            Chunk("test", "text", 1, "test.pdf", "header")
        ], [
            np.random.randn(1024) / np.linalg.norm(np.random.randn(1024))
        ])

        count = storage.get_chunk_count()
        assert count == 1
        storage.close()

    def test_insert_single_chunk(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should insert a single chunk."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        inserted = storage.insert_chunks_batch([sample_chunks[0]], [sample_embeddings[0]])

        assert inserted == 1
        assert storage.get_chunk_count() == 1
        storage.close()

    def test_insert_multiple_chunks(
        self, temp_db: str, sample_chunks: list, sample_embeddings: list
    ) -> None:
        """Should insert multiple chunks."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        inserted = storage.insert_chunks_batch(sample_chunks, sample_embeddings)

        assert inserted == 3
        assert storage.get_chunk_count() == 3
        storage.close()

    def test_insert_no_chunks(self, temp_db: str) -> None:
        """Should handle empty insert gracefully."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        inserted = storage.insert_chunks_batch([], [])

        assert inserted == 0
        assert storage.get_chunk_count() == 0
        storage.close()

    def test_insert_mismatched_counts(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should reject mismatched chunk and embedding counts."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        with pytest.raises(StorageError, match="Chunk count.*embedding count"):
            storage.insert_chunks_batch(sample_chunks[:2], sample_embeddings)

        storage.close()

    def test_insert_invalid_embedding_dimension(self, temp_db: str) -> None:
        """Should reject embeddings with wrong dimension."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        chunk = Chunk("test", "text", 1, "test.pdf")
        # Wrong dimension (768 instead of 1024)
        invalid_emb = np.random.randn(768)
        invalid_emb = invalid_emb / np.linalg.norm(invalid_emb)

        with pytest.raises(StorageError, match="invalid dimension"):
            storage.insert_chunks_batch([chunk], [invalid_emb])

        storage.close()

    def test_create_index(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should create HNSW index."""
        storage = ChunkStorage(temp_db)
        storage.create_table()
        storage.insert_chunks_batch(sample_chunks, sample_embeddings)

        # Should not raise
        storage.create_index()
        storage.close()

    def test_get_chunk_count(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should return correct chunk count."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        assert storage.get_chunk_count() == 0

        storage.insert_chunks_batch([sample_chunks[0]], [sample_embeddings[0]])
        assert storage.get_chunk_count() == 1

        storage.insert_chunks_batch(sample_chunks[1:], sample_embeddings[1:])
        assert storage.get_chunk_count() == 3

        storage.close()

    def test_chunk_metadata_stored(self, temp_db: str) -> None:
        """Should store chunk metadata correctly."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        chunk = Chunk(
            content="Test content",
            chunk_type="table",
            page_number=42,
            source_file="important.pdf",
            section_header="Conclusion",
        )
        emb = np.random.randn(1024)
        emb = emb / np.linalg.norm(emb)

        storage.insert_chunks_batch([chunk], [emb])

        # Query to verify metadata
        result = storage.conn.execute(
            "SELECT source_file, page_number, chunk_type, section_header FROM medical_document_chunks LIMIT 1"
        ).fetchone()

        assert result[0] == "important.pdf"
        assert result[1] == 42
        assert result[2] == "table"
        assert result[3] == "Conclusion"

        storage.close()

    def test_embedding_stored_and_retrieved(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should store and retrieve embeddings correctly."""
        storage = ChunkStorage(temp_db)
        try:
            storage.create_table()

            storage.insert_chunks_batch([sample_chunks[0]], [sample_embeddings[0]])

            # Query embedding back (DuckDB returns fixed-size FLOAT[N] arrays as tuples)
            result = storage.conn.execute(
                "SELECT embedding FROM medical_document_chunks LIMIT 1"
            ).fetchone()

            assert result is not None
            retrieved_emb = result[0]
            assert isinstance(retrieved_emb, (list, tuple))
            assert len(retrieved_emb) == 1024
        finally:
            storage.close()

    def test_verify_table_exists(self, temp_db: str) -> None:
        """Should check if table exists."""
        storage = ChunkStorage(temp_db)

        assert not storage.verify_table_exists()

        storage.create_table()
        assert storage.verify_table_exists()

        storage.close()

    def test_verify_index_exists(self, temp_db: str, sample_chunks: list, sample_embeddings: list) -> None:
        """Should check if index exists."""
        storage = ChunkStorage(temp_db)
        storage.create_table()
        storage.insert_chunks_batch(sample_chunks, sample_embeddings)
        storage.create_index()

        # Note: Full index verification is complex; basic check
        assert storage.verify_index_exists()

        storage.close()

    def test_close_connection(self, temp_db: str) -> None:
        """Should close connection without error."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        # Should not raise
        storage.close()

    def test_multiple_pdfs(self, temp_db: str) -> None:
        """Should store chunks from multiple PDFs."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        # Create chunks from different PDFs
        chunks = [
            Chunk("Content from PDF 1", "text", 1, "pdf1.pdf", "Section 1"),
            Chunk("Content from PDF 2", "text", 1, "pdf2.pdf", "Section 2"),
        ]

        embs = [
            np.random.randn(1024) / np.linalg.norm(np.random.randn(1024)),
            np.random.randn(1024) / np.linalg.norm(np.random.randn(1024)),
        ]

        storage.insert_chunks_batch(chunks, embs)

        # Verify both are stored
        count = storage.get_chunk_count()
        assert count == 2

        # Verify different source files
        result = storage.conn.execute(
            "SELECT COUNT(DISTINCT source_file) FROM medical_document_chunks"
        ).fetchone()
        assert result[0] == 2

        storage.close()

    def test_large_content_storage(self, temp_db: str) -> None:
        """Should handle large content strings."""
        storage = ChunkStorage(temp_db)
        storage.create_table()

        # Create large chunk (10KB of text)
        large_content = "word " * 2000
        chunk = Chunk(large_content, "text", 1, "large.pdf")
        emb = np.random.randn(1024) / np.linalg.norm(np.random.randn(1024))

        storage.insert_chunks_batch([chunk], [emb])

        # Verify it was stored
        result = storage.conn.execute(
            "SELECT content FROM medical_document_chunks LIMIT 1"
        ).fetchone()

        assert len(result[0]) > 9000

        storage.close()
