"""
Embedding generation and normalization using Sentence Transformers.

Generates 1,024-dimensional embeddings and L2-normalizes them for vector search.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingGenerator:
    """Generates and normalizes embeddings using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        expected_dimensions: int = 1024,
        batch_size: int = 32,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model identifier
            expected_dimensions: Expected embedding dimension (must be 1024)
            batch_size: Batch size for embedding generation

        Raises:
            EmbeddingError: If model cannot be loaded or dimensions don't match
        """
        if expected_dimensions != 1024:
            raise EmbeddingError(f"Expected dimensions must be 1024, got {expected_dimensions}")

        self.model_name = model_name
        self.expected_dimensions = expected_dimensions
        self.batch_size = batch_size

        try:
            self.model = SentenceTransformer(model_name)
            # Verify model output dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            actual_dim = len(test_embedding)
            if actual_dim != expected_dimensions:
                raise EmbeddingError(
                    f"Model {model_name} produces {actual_dim}-dimensional embeddings, "
                    f"but {expected_dimensions} were expected"
                )
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {model_name}: {e}")

    def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode a list of texts into embeddings and L2-normalize them.

        Args:
            texts: List of text strings to embed

        Returns:
            List of L2-normalized embedding vectors (numpy arrays)

        Raises:
            EmbeddingError: If encoding fails or dimensions are incorrect
        """
        if not texts:
            return []

        try:
            # Encode all texts at once (batch processing is automatic)
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Ensure embeddings is 2D (even for single input)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Validate dimensions
            if embeddings.shape[1] != self.expected_dimensions:
                raise EmbeddingError(
                    f"Embedding dimension {embeddings.shape[1]} does not match "
                    f"expected {self.expected_dimensions}"
                )

            # L2-normalize each embedding
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            normalized = embeddings / norms

            return [normalized[i] for i in range(normalized.shape[0])]

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to encode texts: {e}")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Text to embed

        Returns:
            L2-normalized embedding vector (numpy array of 1024 dimensions)

        Raises:
            EmbeddingError: If encoding fails
        """
        embeddings = self.encode_texts([text])
        if not embeddings:
            raise EmbeddingError("Failed to generate embedding")
        return embeddings[0]

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding has correct dimensions and is normalized.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if embedding is valid

        Raises:
            EmbeddingError: If embedding is invalid
        """
        if not isinstance(embedding, np.ndarray):
            raise EmbeddingError(f"Embedding must be numpy array, got {type(embedding)}")

        if len(embedding.shape) != 1:
            raise EmbeddingError(f"Embedding must be 1D, got shape {embedding.shape}")

        if embedding.shape[0] != self.expected_dimensions:
            raise EmbeddingError(
                f"Embedding dimension {embedding.shape[0]} does not match "
                f"expected {self.expected_dimensions}"
            )

        # Check if it's roughly L2-normalized (norm should be close to 1)
        norm = np.linalg.norm(embedding)
        if norm < 0.9 or norm > 1.1:
            raise EmbeddingError(f"Embedding is not properly L2-normalized (norm={norm:.4f})")

        return True

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "expected_dimensions": self.expected_dimensions,
            "batch_size": self.batch_size,
        }
