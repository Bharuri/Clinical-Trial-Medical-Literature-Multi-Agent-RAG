"""
Tests for embedding generation module.
"""

import gc

import numpy as np
import pytest
import torch

from src.ingestion.embeddings import EmbeddingGenerator, EmbeddingError


@pytest.fixture(autouse=True)
def _release_model_memory():
    """Free each test's model so repeated GPU loads do not exhaust VRAM."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""

    def test_generator_initialization(self) -> None:
        """Should initialize successfully with default model."""
        # This may download the model if not cached
        gen = EmbeddingGenerator()
        assert gen is not None
        assert gen.expected_dimensions == 1024

    def test_generator_invalid_dimensions(self) -> None:
        """Should reject non-1024 dimensions."""
        with pytest.raises(EmbeddingError, match="Expected dimensions must be 1024"):
            EmbeddingGenerator(expected_dimensions=768)

    def test_generator_model_info(self) -> None:
        """Should return model information."""
        gen = EmbeddingGenerator()
        info = gen.get_model_info()

        assert "model_name" in info
        assert "expected_dimensions" in info
        assert "batch_size" in info
        assert info["expected_dimensions"] == 1024

    def test_encode_single_text(self) -> None:
        """Should encode a single text into 1024-dimensional vector."""
        gen = EmbeddingGenerator()
        embedding = gen.encode_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1024
        assert np.issubdtype(embedding.dtype, np.floating)

    def test_encode_multiple_texts(self) -> None:
        """Should encode multiple texts into embeddings."""
        gen = EmbeddingGenerator()
        texts = ["First text", "Second text", "Third text"]
        embeddings = gen.encode_texts(texts)

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape[0] == 1024 for e in embeddings)

    def test_encode_empty_list(self) -> None:
        """Should return empty list for empty input."""
        gen = EmbeddingGenerator()
        embeddings = gen.encode_texts([])

        assert embeddings == []

    def test_embedding_is_normalized(self) -> None:
        """Should L2-normalize embeddings."""
        gen = EmbeddingGenerator()
        embedding = gen.encode_text("Sample text for normalization test")

        # L2 norm should be approximately 1
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01

    def test_validate_embedding_success(self) -> None:
        """Should validate correct embeddings."""
        gen = EmbeddingGenerator()
        embedding = gen.encode_text("Test validation")

        # Should not raise
        assert gen.validate_embedding(embedding)

    def test_validate_embedding_wrong_type(self) -> None:
        """Should reject non-numpy arrays."""
        gen = EmbeddingGenerator()

        with pytest.raises(EmbeddingError, match="must be numpy array"):
            gen.validate_embedding([1.0] * 1024)

    def test_validate_embedding_wrong_dimension(self) -> None:
        """Should reject embeddings with wrong dimension."""
        gen = EmbeddingGenerator()

        wrong_dim = np.random.randn(768)
        with pytest.raises(EmbeddingError, match="dimension"):
            gen.validate_embedding(wrong_dim)

    def test_validate_embedding_not_normalized(self) -> None:
        """Should reject embeddings that aren't L2-normalized."""
        gen = EmbeddingGenerator()

        # Create an unnormalized embedding
        unnormalized = np.random.randn(1024)
        with pytest.raises(EmbeddingError, match="not properly L2-normalized"):
            gen.validate_embedding(unnormalized)

    def test_encode_texts_consistency(self) -> None:
        """Should produce consistent embeddings for same text."""
        gen = EmbeddingGenerator()
        text = "This is a consistent text"

        emb1 = gen.encode_text(text)
        emb2 = gen.encode_text(text)

        # Should be nearly identical (may have minor floating point differences)
        assert np.allclose(emb1, emb2, atol=1e-6)

    def test_encode_different_texts_different_embeddings(self) -> None:
        """Should produce different embeddings for different texts."""
        gen = EmbeddingGenerator()

        emb1 = gen.encode_text("This is first text")
        emb2 = gen.encode_text("This is completely different text")

        # Embeddings should be different
        diff = np.linalg.norm(emb1 - emb2)
        assert diff > 0.1  # Significant difference

    def test_encode_texts_batch_processing(self) -> None:
        """Should handle batch processing correctly."""
        gen = EmbeddingGenerator(batch_size=2)

        # Create texts that span multiple batches
        texts = [f"Text number {i}" for i in range(10)]
        embeddings = gen.encode_texts(texts)

        assert len(embeddings) == 10
        assert all(e.shape[0] == 1024 for e in embeddings)

    def test_embedding_properties(self) -> None:
        """Generated embeddings should have expected properties."""
        gen = EmbeddingGenerator()
        texts = ["Sample text one", "Sample text two", "Sample text three"]
        embeddings = gen.encode_texts(texts)

        for emb in embeddings:
            # Should be 1D array
            assert emb.ndim == 1
            # Should have 1024 dimensions
            assert emb.shape[0] == 1024
            # Should be normalized (norm ~1)
            norm = np.linalg.norm(emb)
            assert 0.99 <= norm <= 1.01
            # Should be float type
            assert np.issubdtype(emb.dtype, np.floating)

    def test_encode_long_text(self) -> None:
        """Should handle long text input."""
        gen = EmbeddingGenerator()

        # Create a long text
        long_text = " ".join(["word"] * 1000)
        embedding = gen.encode_text(long_text)

        assert embedding.shape[0] == 1024
        assert np.linalg.norm(embedding) <= 1.01

    def test_encode_special_characters(self) -> None:
        """Should handle text with special characters."""
        gen = EmbeddingGenerator()

        texts = [
            "Text with numbers: 123.456",
            "Text with symbols: @#$%&*()",
            "Text with unicode: 你好世界",
        ]

        embeddings = gen.encode_texts(texts)
        assert len(embeddings) == 3
        assert all(e.shape[0] == 1024 for e in embeddings)

    def test_single_vs_batch_consistency(self) -> None:
        """Encoding single text vs in batch should give same result."""
        gen = EmbeddingGenerator()

        text = "Test consistency between single and batch encoding"

        # Encode as single
        single = gen.encode_text(text)

        # Encode as batch of one
        batch = gen.encode_texts([text])

        # Should be nearly identical
        assert np.allclose(single, batch[0], atol=1e-6)
