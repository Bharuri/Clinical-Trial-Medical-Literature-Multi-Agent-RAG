"""
Tests for hybrid retrieval (BM25 keyword search fused with vector search) and the optional re-ranker.
"""

from pathlib import Path
from typing import List
from unittest.mock import patch

import duckdb
import numpy as np
import pytest

from src.retrieval_agent import hybrid
from src.retrieval_agent.hybrid import (
    BM25Index,
    ChunkRow,
    HybridRetriever,
    _get_bm25_index,
    score_fusion,
    tokenize,
)
from src.retrieval_agent.reranker import CrossEncoderReranker
from src.retrieval_agent.retrieval import (
    ROUTE_TABLE,
    Retriever,
    RetrievalError,
    RetrievedChunk,
)
from src.utils.config import Config

DIMS = 768


class FakeEncoder:
    """Returns a fixed vector for any input."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector

    def encode_text(self, text: str) -> np.ndarray:
        return self.vector


class RecordingReranker:
    """Records each rerank call and keeps the first top_k candidates."""

    def __init__(self):
        self.calls = []

    def rerank(self, question, chunks, top_k):
        self.calls.append((question, list(chunks), top_k))
        return list(chunks)[:top_k]


class FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size=16):
        self.calls.append((pairs, batch_size))
        return self.scores


def basis(index: int) -> np.ndarray:
    vec = np.zeros(DIMS, dtype=np.float32)
    vec[index] = 1.0
    return vec


def make_chunk(chunk_id: str, content: str, similarity: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_file="trial.pdf",
        page_number=1,
        chunk_type="text",
        section_header="Results",
        content=content,
        similarity_score=similarity,
    )


def build_db(path: Path, rows) -> str:
    """Create medical_document_chunks and insert rows of (id, source, page, type, header, content, vector)."""
    conn = duckdb.connect(str(path))
    conn.execute(
        f"""
        CREATE TABLE medical_document_chunks (
            id VARCHAR PRIMARY KEY,
            source_file VARCHAR NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_type VARCHAR NOT NULL,
            section_header VARCHAR NOT NULL,
            content VARCHAR NOT NULL,
            embedding FLOAT[{DIMS}] NOT NULL
        )
        """
    )
    conn.executemany(
        f"INSERT INTO medical_document_chunks VALUES (?, ?, ?, ?, ?, ?, CAST(? AS FLOAT[{DIMS}]))",
        [(rid, src, page, ctype, header, content, vec.tolist()) for rid, src, page, ctype, header, content, vec in rows],
    )
    conn.close()
    return str(path)


def make_hybrid(db_path: str, vector: np.ndarray, reranker_factory=None, top_k: int = 5) -> HybridRetriever:
    retriever = Retriever(encoder=FakeEncoder(vector), db_path=db_path, top_k=top_k)
    return HybridRetriever(retriever, reranker_factory=reranker_factory)


@pytest.fixture(autouse=True)
def reset_state():
    Config.reset()
    hybrid._cached_index = None
    yield
    Config.reset()
    hybrid._cached_index = None


@pytest.fixture
def keyword_db(tmp_path) -> str:
    """One chunk holds the rare term but is far from the query vector; five generic chunks sit right on it."""
    rows = [("a1", "trial.pdf", 1, "text", "Results", "Pembrolizumab improved survival in the cohort", basis(5))]
    rows += [
        (f"f{i}", "trial.pdf", 2, "text", "Methods", "Patients were enrolled at sites", basis(0))
        for i in range(5)
    ]
    return build_db(tmp_path / "keyword.duckdb", rows)


@pytest.fixture
def near_keyword_db(tmp_path) -> str:
    """The rare-term chunk has cosine 0.7 to the query, below five exact vector matches."""
    near = np.zeros(DIMS, dtype=np.float32)
    near[0] = 0.7
    near[5] = np.sqrt(1 - 0.7**2)
    rows = [("a1", "trial.pdf", 1, "text", "Results", "Pembrolizumab improved survival in the cohort", near)]
    rows += [
        (f"f{i}", "trial.pdf", 2, "text", "Methods", "Patients were enrolled at sites", basis(0))
        for i in range(5)
    ]
    return build_db(tmp_path / "near.duckdb", rows)


@pytest.fixture
def mixed_db(tmp_path) -> str:
    rows = [
        ("t1", "trial.pdf", 1, "text", "Results", "Pembrolizumab dosing narrative", basis(0)),
        ("tb1", "trial.pdf", 2, "table", "Table 1", "Pembrolizumab arm | 40", basis(0)),
        ("t2", "other.pdf", 3, "text", "Results", "Pembrolizumab in another trial", basis(0)),
    ]
    return build_db(tmp_path / "mixed.duckdb", rows)


class TestTokenize:
    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("What was the HR of 0.85 in the Table?") == ["hr", "0.85", "table"]

    def test_keeps_decimals_whole(self):
        assert "1.5" in tokenize("a value of 1.5 units")

    def test_empty_text_gives_no_tokens(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestBM25Index:
    def test_term_frequency_ranks_higher(self):
        index = BM25Index([tokenize("pembrolizumab survival benefit"), tokenize("enrolment sites"), tokenize("survival survival survival")])
        scores = index.scores(tokenize("survival"))
        assert set(scores) == {0, 2}
        assert scores[2] > scores[0]

    def test_rare_term_only_matches_its_document(self):
        index = BM25Index([tokenize("pembrolizumab survival"), tokenize("placebo arm")])
        assert set(index.scores(tokenize("pembrolizumab"))) == {0}

    def test_empty_query_scores_nothing(self):
        index = BM25Index([tokenize("some text")])
        assert index.scores([]) == {}


class TestScoreFusion:
    def test_weighted_sum_of_cosine_and_normalized_bm25(self):
        cosine = {"a": 0.9, "b": 0.95}
        bm25 = {"a": 4.0}
        assert score_fusion(["b", "a"], cosine, bm25) == ["a", "b"]

    def test_bm25_is_scaled_by_the_best_candidate(self):
        cosine = {"a": 0.8, "b": 0.8}
        bm25 = {"a": 4.0, "b": 2.0}
        assert score_fusion(["b", "a"], cosine, bm25) == ["a", "b"]

    def test_keyword_only_chunk_can_outrank_a_weaker_vector_hit(self):
        cosine = {"vector_top": 1.0, "keyword_only": 0.7}
        bm25 = {"keyword_only": 3.0}
        assert score_fusion(["vector_top", "keyword_only"], cosine, bm25) == ["keyword_only", "vector_top"]

    def test_orthogonal_keyword_chunk_does_not_outrank_vector_hit(self):
        cosine = {"vector_top": 1.0, "keyword_only": 0.0}
        bm25 = {"keyword_only": 3.0}
        assert score_fusion(["keyword_only", "vector_top"], cosine, bm25) == ["vector_top", "keyword_only"]

    def test_without_keyword_matches_order_is_by_cosine(self):
        cosine = {"a": 0.5, "b": 0.9, "c": 0.7}
        assert score_fusion(["a", "b", "c"], cosine, {}) == ["b", "c", "a"]

    def test_ties_keep_candidate_order(self):
        assert score_fusion(["a", "b"], {"a": 0.5, "b": 0.5}, {}) == ["a", "b"]

    def test_weights_can_be_overridden(self):
        cosine = {"a": 0.9, "b": 0.5}
        bm25 = {"b": 1.0}
        assert score_fusion(["a", "b"], cosine, bm25, semantic_weight=0.0, keyword_weight=1.0) == ["b", "a"]


class TestBM25IndexCache:
    def test_same_chunks_reuse_cached_index(self):
        rows = [ChunkRow("1", "f.pdf", 1, "text", "h", "alpha beta")]
        assert _get_bm25_index(rows) is _get_bm25_index(list(rows))

    def test_changed_content_rebuilds_index(self):
        before = [ChunkRow("1", "f.pdf", 1, "text", "h", "alpha beta")]
        after = [ChunkRow("1", "f.pdf", 1, "text", "h", "gamma delta")]
        first = _get_bm25_index(before)
        second = _get_bm25_index(after)
        assert second is not first
        assert second.scores(tokenize("gamma"))


class TestHybridSearch:
    def test_keyword_match_is_surfaced_when_candidate_pool_is_large(self, keyword_db):
        hybrid_retriever = make_hybrid(keyword_db, basis(0), reranker_factory=lambda: RecordingReranker())
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab survival", use_reranker=True)
        finally:
            hybrid_retriever.close()

        ids = {c.chunk_id for c in result.chunks}
        assert "a1" in ids

    def test_keyword_chunk_outranks_weaker_vector_hits_at_small_top_k(self, near_keyword_db):
        hybrid_retriever = make_hybrid(near_keyword_db, basis(0))
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab survival", top_k=2)
        finally:
            hybrid_retriever.close()

        assert result.chunks[0].chunk_id == "a1"
        assert len(result.chunks) == 2

    def test_orthogonal_keyword_chunk_does_not_displace_vector_hits(self, keyword_db):
        hybrid_retriever = make_hybrid(keyword_db, basis(0))
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab survival", top_k=2)
        finally:
            hybrid_retriever.close()

        assert "a1" not in {c.chunk_id for c in result.chunks}
        assert len(result.chunks) == 2

    def test_keyword_chunk_gets_cosine_similarity(self, keyword_db):
        hybrid_retriever = make_hybrid(keyword_db, basis(0), reranker_factory=lambda: RecordingReranker())
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab survival", use_reranker=True)
        finally:
            hybrid_retriever.close()

        keyword_chunk = next(c for c in result.chunks if c.chunk_id == "a1")
        assert keyword_chunk.similarity_score == pytest.approx(0.0, abs=1e-4)

    def test_route_filter_applies_to_keyword_leg(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        try:
            result = hybrid_retriever.retrieve("Show the table for pembrolizumab", top_k=5)
        finally:
            hybrid_retriever.close()

        assert result.route == ROUTE_TABLE
        assert result.chunks
        assert all(c.chunk_type == "table" for c in result.chunks)

    def test_source_filter_applies_to_keyword_leg(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab", top_k=5, source_files=["other.pdf"])
        finally:
            hybrid_retriever.close()

        assert {c.chunk_id for c in result.chunks} == {"t2"}
        assert all(c.source_file == "other.pdf" for c in result.chunks)

    def test_empty_question_raises(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        try:
            with pytest.raises(RetrievalError, match="empty"):
                hybrid_retriever.retrieve("   ")
        finally:
            hybrid_retriever.close()

    def test_empty_source_selection_raises(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        try:
            with pytest.raises(RetrievalError, match="source file"):
                hybrid_retriever.retrieve("Explain pembrolizumab", source_files=[])
        finally:
            hybrid_retriever.close()

    def test_top_k_zero_raises(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        try:
            with pytest.raises(RetrievalError, match="top_k"):
                hybrid_retriever.retrieve("Explain pembrolizumab", top_k=0)
        finally:
            hybrid_retriever.close()

    def test_close_closes_underlying_retriever(self, mixed_db):
        hybrid_retriever = make_hybrid(mixed_db, basis(0))
        with patch.object(hybrid_retriever.retriever, "close") as close:
            hybrid_retriever.close()
        close.assert_called_once()


class TestOptionalReranker:
    def test_reranker_off_returns_top_k_and_never_builds_reranker(self, keyword_db):
        def factory():
            raise AssertionError("re-ranker must not be built when disabled")

        hybrid_retriever = make_hybrid(keyword_db, basis(0), reranker_factory=factory)
        try:
            result = hybrid_retriever.retrieve("Explain pembrolizumab survival", top_k=3, use_reranker=False)
        finally:
            hybrid_retriever.close()

        assert len(result.chunks) == 3

    def test_reranker_on_scores_200_candidates_and_keeps_10(self, tmp_path):
        rows = [
            (f"c{i}", "trial.pdf", 1, "text", "Results", f"filler content number {i}", basis(0))
            for i in range(250)
        ]
        db_path = build_db(tmp_path / "large.duckdb", rows)
        reranker = RecordingReranker()
        hybrid_retriever = make_hybrid(db_path, basis(0), reranker_factory=lambda: reranker, top_k=5)
        try:
            result = hybrid_retriever.retrieve("Explain filler content", top_k=3, use_reranker=True)
        finally:
            hybrid_retriever.close()

        assert len(reranker.calls) == 1
        _, candidates, final_k = reranker.calls[0]
        assert len(candidates) == 200
        assert final_k == 10
        assert len(result.chunks) == 10

    def test_reranker_is_built_once_and_reused(self, keyword_db):
        built: List[RecordingReranker] = []

        def factory():
            reranker = RecordingReranker()
            built.append(reranker)
            return reranker

        hybrid_retriever = make_hybrid(keyword_db, basis(0), reranker_factory=factory)
        try:
            hybrid_retriever.retrieve("Explain pembrolizumab survival", use_reranker=True)
            hybrid_retriever.retrieve("Explain pembrolizumab survival", use_reranker=True)
        finally:
            hybrid_retriever.close()

        assert len(built) == 1
        assert len(built[0].calls) == 2

    def test_reranker_failure_propagates(self, keyword_db):
        class BrokenReranker:
            def rerank(self, question, chunks, top_k):
                raise RetrievalError("Re-ranking failed: boom")

        hybrid_retriever = make_hybrid(keyword_db, basis(0), reranker_factory=lambda: BrokenReranker())
        try:
            with pytest.raises(RetrievalError, match="boom"):
                hybrid_retriever.retrieve("Explain pembrolizumab survival", use_reranker=True)
        finally:
            hybrid_retriever.close()


class TestCrossEncoderReranker:
    def test_orders_by_score_and_keeps_top_k(self):
        model = FakeCrossEncoder([0.1, 0.9, 0.5])
        reranker = CrossEncoderReranker("unused", model=model)
        chunks = [make_chunk("c0", "zero"), make_chunk("c1", "one"), make_chunk("c2", "two")]

        result = reranker.rerank("question", chunks, top_k=2)

        assert [c.chunk_id for c in result] == ["c1", "c2"]

    def test_scores_question_content_pairs(self):
        model = FakeCrossEncoder([0.3])
        reranker = CrossEncoderReranker("unused", model=model, batch_size=8)

        reranker.rerank("what is it", [make_chunk("c0", "some content")], top_k=1)

        pairs, batch_size = model.calls[0]
        assert pairs == [("what is it", "some content")]
        assert batch_size == 8

    def test_similarity_score_is_not_changed(self):
        reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder([0.2, 0.8]))
        chunks = [make_chunk("c0", "a", similarity=0.11), make_chunk("c1", "b", similarity=0.22)]

        result = reranker.rerank("q", chunks, top_k=2)

        assert {c.chunk_id: c.similarity_score for c in result} == {"c0": 0.11, "c1": 0.22}

    def test_ties_keep_input_order(self):
        reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder([0.5, 0.5]))
        chunks = [make_chunk("c0", "a"), make_chunk("c1", "b")]

        assert reranker.rerank("q", chunks, top_k=1)[0].chunk_id == "c0"

    def test_empty_candidates_return_empty_without_scoring(self):
        model = FakeCrossEncoder([])
        reranker = CrossEncoderReranker("unused", model=model)

        assert reranker.rerank("q", [], top_k=10) == []
        assert model.calls == []

    def test_score_count_mismatch_raises(self):
        reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder([0.1]))
        with pytest.raises(RetrievalError, match="1 scores for 2 chunks"):
            reranker.rerank("q", [make_chunk("c0", "a"), make_chunk("c1", "b")], top_k=1)

    def test_scoring_error_is_wrapped(self):
        class ExplodingModel:
            def predict(self, pairs, batch_size=16):
                raise ValueError("bad input")

        reranker = CrossEncoderReranker("unused", model=ExplodingModel())
        with pytest.raises(RetrievalError, match="Re-ranking failed"):
            reranker.rerank("q", [make_chunk("c0", "a")], top_k=1)

    def test_top_k_below_one_raises(self):
        reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder([0.1]))
        with pytest.raises(RetrievalError, match="top_k"):
            reranker.rerank("q", [make_chunk("c0", "a")], top_k=0)

    def test_model_load_failure_is_wrapped(self):
        with patch("sentence_transformers.CrossEncoder", side_effect=OSError("offline")):
            with pytest.raises(RetrievalError, match="Failed to load re-ranker model"):
                CrossEncoderReranker("BAAI/bge-reranker-base")
