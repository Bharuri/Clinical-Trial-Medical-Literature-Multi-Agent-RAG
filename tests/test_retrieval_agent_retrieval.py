"""
Tests for the retrieval module: query routing and DuckDB vector search.
"""

import tempfile
from pathlib import Path
from typing import Generator

import duckdb
import numpy as np
import pytest

from src.retrieval_agent.retrieval import (
    ROUTE_BOTH,
    ROUTE_TABLE,
    ROUTE_TEXT,
    RetrievalError,
    Retriever,
    RetrievedChunk,
    route_query,
)
from src.utils.config import Config

DIMS = 1024


class FakeEncoder:
    """Returns a fixed vector for any input and records the questions it saw."""

    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.seen = []

    def encode_text(self, text: str) -> np.ndarray:
        self.seen.append(text)
        return self.vector


def basis(index: int) -> np.ndarray:
    vec = np.zeros(DIMS, dtype=np.float32)
    vec[index] = 1.0
    return vec


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


@pytest.fixture
def db_path() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "retrieval_test.duckdb")
        conn = duckdb.connect(path)
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
        rows = [
            ("t-near", "trial.pdf", 1, "text", "Results", "Narrative near query", basis(0)),
            ("t-mid", "trial.pdf", 2, "text", "Methods", "Narrative mid", basis(1)),
            ("tb-near", "trial.pdf", 3, "table", "Table 1", "| HR | 0.8 |", basis(0)),
            ("tb-far", "other.pdf", 4, "table", "Table 2", "| n | 40 |", basis(2)),
        ]
        for row_id, source, page, ctype, header, content, vec in rows:
            conn.execute(
                "INSERT INTO medical_document_chunks VALUES (?, ?, ?, ?, ?, ?, CAST(? AS FLOAT[1024]))",
                [row_id, source, page, ctype, header, content, vec.tolist()],
            )
        conn.close()
        yield path


class TestRouteQuery:
    def test_explicit_table_question_routes_to_table(self):
        assert route_query("What does the table show for column HR?") == ROUTE_TABLE

    def test_narrative_question_routes_to_text(self):
        assert route_query("Explain the mechanism behind the drug's effect") == ROUTE_TEXT

    def test_numeric_question_routes_to_both(self):
        assert route_query("What was the hazard ratio for overall survival?") == ROUTE_BOTH

    def test_narrative_with_numbers_routes_to_both(self):
        assert route_query("Explain the hazard ratio results") == ROUTE_BOTH

    def test_unrecognised_question_routes_to_both(self):
        assert route_query("Tell me about the trial") == ROUTE_BOTH


class TestRetriever:
    def test_returns_top_k_ordered_by_similarity(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=2)
        try:
            result = retriever.retrieve("What was the hazard ratio?")
        finally:
            retriever.close()

        assert result.match_count == 2
        assert result.chunks[0].chunk_id in {"t-near", "tb-near"}
        assert result.chunks[0].similarity_score == pytest.approx(1.0, abs=1e-4)
        scores = [c.similarity_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_text_route_excludes_table_chunks(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=5)
        try:
            result = retriever.retrieve("Explain the mechanism of action")
        finally:
            retriever.close()

        assert result.route == ROUTE_TEXT
        assert result.chunks
        assert all(c.chunk_type == "text" for c in result.chunks)

    def test_table_route_returns_only_tables(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=5)
        try:
            result = retriever.retrieve("Show the table columns for the baseline rows")
        finally:
            retriever.close()

        assert result.route == ROUTE_TABLE
        assert {c.chunk_id for c in result.chunks} == {"tb-near", "tb-far"}

    def test_chunk_metadata_is_returned(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(1)), db_path=db_path, top_k=1)
        try:
            result = retriever.retrieve("Explain the methods")
        finally:
            retriever.close()

        chunk = result.chunks[0]
        assert chunk.chunk_id == "t-mid"
        assert chunk.source_file == "trial.pdf"
        assert chunk.page_number == 2
        assert chunk.chunk_type == "text"
        assert chunk.section_header == "Methods"
        assert chunk.content == "Narrative mid"

    def test_query_dimensions_reported(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=1)
        try:
            result = retriever.retrieve("Explain the results")
        finally:
            retriever.close()

        assert result.query_dimensions == DIMS

    def test_wrong_query_dimension_raises(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(np.ones(8, dtype=np.float32)), db_path=db_path, top_k=1)
        try:
            with pytest.raises(RetrievalError, match="dimensions"):
                retriever.retrieve("Explain the results")
        finally:
            retriever.close()

    def test_empty_question_raises(self, db_path):
        encoder = FakeEncoder(basis(0))
        retriever = Retriever(encoder=encoder, db_path=db_path, top_k=1)
        try:
            with pytest.raises(RetrievalError, match="empty"):
                retriever.retrieve("   ")
        finally:
            retriever.close()
        assert encoder.seen == []

    def test_top_k_zero_rejected(self, db_path):
        with pytest.raises(RetrievalError, match="top_k"):
            Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=0)

    def test_per_call_top_k_overrides_default(self, db_path):
        retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=db_path, top_k=5)
        try:
            result = retriever.retrieve("Explain the results", top_k=1)
        finally:
            retriever.close()

        assert result.match_count == 1

    def test_missing_table_raises_with_ingestion_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_db = str(Path(tmpdir) / "empty.duckdb")
            retriever = Retriever(encoder=FakeEncoder(basis(0)), db_path=empty_db, top_k=1)
            try:
                with pytest.raises(RetrievalError, match="ingestion"):
                    retriever.retrieve("Explain the results")
            finally:
                retriever.close()


class TestRetrievedChunkPreview:
    def test_short_content_unchanged(self):
        chunk = RetrievedChunk("id", "f.pdf", 1, "text", "H", "Short text", 0.9)
        assert chunk.preview() == "Short text"

    def test_long_content_truncated(self):
        chunk = RetrievedChunk("id", "f.pdf", 1, "text", "H", "word " * 100, 0.9)
        preview = chunk.preview(max_chars=50)
        assert len(preview) == 50
        assert preview.endswith("...")
