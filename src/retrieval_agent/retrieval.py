"""
Query routing and DuckDB vector retrieval over medical_document_chunks.

Queries are embedded with the same EmbeddingGenerator used during ingestion,
so query and chunk vectors come from the same model.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np

from src.ingestion.embeddings import EmbeddingGenerator
from src.utils.config import Config
from src.utils.connections import DuckDBConnection

TABLE_NAME = "medical_document_chunks"

ROUTE_TEXT = "text"
ROUTE_TABLE = "table"
ROUTE_BOTH = "both"

_ROUTE_CHUNK_TYPES = {
    ROUTE_TEXT: ("text",),
    ROUTE_TABLE: ("table",),
    ROUTE_BOTH: None,
}

_EXPLICIT_TABLE_TERMS = re.compile(r"\b(tables?|columns?|rows?)\b", re.IGNORECASE)
_NUMERIC_TERMS = re.compile(
    r"\b(rates?|hazard ratios?|odds ratios?|p-?values?|confidence intervals?|ci|"
    r"percent|percentages?|means?|medians?|numbers?|how many|counts?|doses?|"
    r"enrolled|enrollment|sample size|versus|vs|compare|comparison|outcomes?|results?)\b"
    r"|%",
    re.IGNORECASE,
)
_NARRATIVE_TERMS = re.compile(
    r"\b(why|how|explain|describe|mechanism|background|discuss|summari[sz]e|summary|"
    r"overview|rationale|methodology|limitations?|context|introduction)\b",
    re.IGNORECASE,
)


def route_query(question: str) -> str:
    """
    Decide which chunk types a question needs.

    Returns:
        "table" for questions that only ask about tables,
        "text" for narrative-only questions,
        "both" otherwise. Numerical questions get "both" so the
        narrative context around the numbers is also retrieved.
    """
    text = question or ""
    needs_numbers = bool(_NUMERIC_TERMS.search(text))
    needs_narrative = bool(_NARRATIVE_TERMS.search(text))
    explicitly_tabular = bool(_EXPLICIT_TABLE_TERMS.search(text))

    if explicitly_tabular and not needs_narrative:
        return ROUTE_TABLE
    if needs_narrative and not needs_numbers:
        return ROUTE_TEXT
    return ROUTE_BOTH


class RetrievalError(Exception):
    """Raised when a query cannot be embedded or searched."""
    pass


class QueryEncoder(Protocol):
    """Anything that embeds a text string into a vector."""

    def encode_text(self, text: str) -> np.ndarray:
        ...


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_file: str
    page_number: int
    chunk_type: str
    section_header: str
    content: str
    similarity_score: float

    def preview(self, max_chars: int = 200) -> str:
        flat = " ".join(self.content.split())
        if len(flat) <= max_chars:
            return flat
        return flat[: max_chars - 3] + "..."


@dataclass
class RetrievalResult:
    question: str
    route: str
    query_dimensions: int
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def match_count(self) -> int:
        return len(self.chunks)


class Retriever:
    """Embeds questions and returns the top-k most similar stored chunks."""

    def __init__(
        self,
        encoder: Optional[QueryEncoder] = None,
        db_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            encoder: Object with encode_text(). Defaults to the local
                Sentence Transformers model from config.yaml.
            db_path: DuckDB file. Defaults to the path in config.yaml.
            top_k: Default number of chunks to return. Defaults to config.yaml.

        Raises:
            RetrievalError: If top_k is invalid or the database cannot be opened.
        """
        config = Config.get_config()
        self.dimensions = config.get_embedding_dimensions()
        self.top_k = top_k if top_k is not None else config.get_top_k()
        if self.top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {self.top_k}")

        if encoder is None:
            encoder = EmbeddingGenerator(
                model_name=config.get_embedding_model(),
                expected_dimensions=self.dimensions,
            )
        self.encoder = encoder

        try:
            self.conn = DuckDBConnection.get_connection(db_path)
        except Exception as e:
            raise RetrievalError(f"Failed to connect to DuckDB: {e}") from e

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_files: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        """
        Route the question, embed it, and return the top-k matching chunks.

        Args:
            source_files: Only search chunks from these source files. None searches all files.

        Raises:
            RetrievalError: If the question is empty, source_files is an empty list,
                the embedding has the wrong dimension, or the search fails.
        """
        if not question or not question.strip():
            raise RetrievalError("Question must not be empty")

        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {k}")

        if source_files is not None and len(source_files) == 0:
            raise RetrievalError("At least one source file must be selected")

        route = route_query(question)
        query_vector = self._embed(question)
        chunks = self._search(query_vector, route, k, source_files)
        return RetrievalResult(
            question=question,
            route=route,
            query_dimensions=int(query_vector.shape[0]),
            chunks=chunks,
        )

    def _embed(self, question: str) -> np.ndarray:
        try:
            vector = np.asarray(self.encoder.encode_text(question), dtype=np.float32).reshape(-1)
        except Exception as e:
            raise RetrievalError(f"Failed to embed question: {e}") from e

        if vector.shape[0] != self.dimensions:
            raise RetrievalError(
                f"Query embedding has {vector.shape[0]} dimensions, expected {self.dimensions}"
            )
        return vector

    def _search(
        self,
        query_vector: np.ndarray,
        route: str,
        k: int,
        source_files: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        conditions: List[str] = []
        params: list = [query_vector.tolist()]

        chunk_types = _ROUTE_CHUNK_TYPES[route]
        if chunk_types:
            conditions.append(f"chunk_type IN ({', '.join('?' for _ in chunk_types)})")
            params.extend(chunk_types)
        if source_files is not None:
            conditions.append(f"source_file IN ({', '.join('?' for _ in source_files)})")
            params.extend(source_files)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(k)

        sql = f"""
        SELECT id, source_file, page_number, chunk_type, section_header, content,
               array_cosine_distance(embedding, CAST(? AS FLOAT[{self.dimensions}])) AS distance
        FROM {TABLE_NAME}
        {where_clause}
        ORDER BY distance ASC
        LIMIT ?
        """

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except Exception as e:
            raise RetrievalError(
                f"Vector search failed (is the {TABLE_NAME} table populated? run ingestion first): {e}"
            ) from e

        return [
            RetrievedChunk(
                chunk_id=row[0],
                source_file=row[1],
                page_number=int(row[2]),
                chunk_type=row[3],
                section_header=row[4] or "",
                content=row[5],
                similarity_score=float(1.0 - row[6]),
            )
            for row in rows
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
