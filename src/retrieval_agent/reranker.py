"""
Optional cross-encoder re-ranker for retrieved chunks.

A cross-encoder reads the question and each chunk together, so it orders
candidates more accurately than embedding similarity. It is slower, so it runs
only on the candidate pool that hybrid search produces.
"""

from typing import Any, List, Optional, Protocol, Sequence

from src.retrieval_agent.retrieval import RetrievalError, RetrievedChunk


class Reranker(Protocol):
    """Anything that orders candidate chunks by relevance to a question."""

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        ...


class CrossEncoderReranker:
    """Re-ranks chunks with a sentence_transformers CrossEncoder."""

    def __init__(self, model_name: str, model: Optional[Any] = None, batch_size: int = 16):
        """
        Args:
            model_name: Hugging Face model ID, used only when model is None.
            model: Object with predict(pairs, batch_size=...). Tests inject a fake here.
            batch_size: Pairs scored per forward pass.

        Raises:
            RetrievalError: If the model cannot be loaded.
        """
        if model is None:
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(model_name)
            except Exception as e:
                raise RetrievalError(f"Failed to load re-ranker model '{model_name}': {e}") from e
        self._model = model
        self._batch_size = batch_size

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """
        Score every (question, chunk) pair and return the top_k best chunks.

        Ties keep the input order. The chunks' similarity_score is not changed.

        Raises:
            RetrievalError: If scoring fails or the scores do not match the chunks.
        """
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")
        if not chunks:
            return []

        pairs = [(question, chunk.content) for chunk in chunks]
        try:
            scores = self._model.predict(pairs, batch_size=self._batch_size)
        except Exception as e:
            raise RetrievalError(f"Re-ranking failed: {e}") from e

        scores = [float(s) for s in scores]
        if len(scores) != len(chunks):
            raise RetrievalError(
                f"Re-ranker returned {len(scores)} scores for {len(chunks)} chunks"
            )

        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        return [chunks[i] for i in order[:top_k]]
