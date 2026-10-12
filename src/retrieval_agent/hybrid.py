"""
Hybrid retrieval: BM25 keyword search fused with the existing vector search.

Retriever (retrieval.py) stays the semantic leg and is not changed. This module
adds a BM25 keyword score over the same chunk table and combines the two with a
weighted score fusion (cosine 0.75, normalized BM25 0.25). An optional
cross-encoder re-ranks the fused candidates.
"""

import hashlib
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.retrieval_agent.reranker import CrossEncoderReranker, Reranker
from src.retrieval_agent.retrieval import (
    ROUTE_BOTH,
    ROUTE_TABLE,
    ROUTE_TEXT,
    TABLE_NAME,
    RetrievalError,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    route_query,
)
from src.utils.config import Config

SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25
BM25_K1 = 1.5
BM25_B = 0.75

_ROUTE_CHUNK_TYPES = {
    ROUTE_TEXT: frozenset({"text"}),
    ROUTE_TABLE: frozenset({"table"}),
    ROUTE_BOTH: None,
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
        "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "than",
        "that", "the", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "why", "with",
    }
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, keeping decimals like 0.85 whole, minus stopwords."""
    return [t for t in _TOKEN_PATTERN.findall((text or "").lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    source_file: str
    page_number: int
    chunk_type: str
    section_header: str
    content: str


class BM25Index:
    """Okapi BM25 over pre-tokenized documents. Document positions match the input order."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = BM25_K1, b: float = BM25_B):
        self._k1 = k1
        self._b = b
        self._doc_count = len(documents)
        self._doc_lengths = [len(doc) for doc in documents]
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / self._doc_count if self._doc_count else 1.0
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for position, doc in enumerate(documents):
            for term, frequency in Counter(doc).items():
                self._postings[term].append((position, frequency))

    def scores(self, query_terms: Sequence[str]) -> Dict[int, float]:
        """Return a BM25 score for every document that contains at least one query term."""
        scores: Dict[int, float] = defaultdict(float)
        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))
            for position, frequency in postings:
                length_norm = 1.0 - self._b + self._b * self._doc_lengths[position] / self._avg_doc_length
                scores[position] += idf * frequency * (self._k1 + 1.0) / (frequency + self._k1 * length_norm)
        return dict(scores)


_index_lock = threading.Lock()
_cached_index: Optional[Tuple[str, BM25Index]] = None


def _fingerprint(rows: Sequence[ChunkRow]) -> str:
    """Hash of every chunk id and its content. A change to either rebuilds the index."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(row.content.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _get_bm25_index(rows: Sequence[ChunkRow]) -> BM25Index:
    """Return the BM25 index for these rows, reusing the cached one when the chunks are unchanged."""
    global _cached_index
    digest = _fingerprint(rows)
    with _index_lock:
        if _cached_index is not None and _cached_index[0] == digest:
            return _cached_index[1]

    index = BM25Index([tokenize(row.content) for row in rows])
    with _index_lock:
        _cached_index = (digest, index)
    return index


def fetch_chunk_rows(conn) -> List[ChunkRow]:
    """Load every chunk, ordered by id so positions are stable across calls."""
    sql = (
        f"SELECT id, source_file, page_number, chunk_type, section_header, content "
        f"FROM {TABLE_NAME} ORDER BY id"
    )
    try:
        rows = conn.execute(sql).fetchall()
    except Exception as e:
        raise RetrievalError(f"Keyword search failed to read {TABLE_NAME}: {e}") from e

    return [
        ChunkRow(
            chunk_id=row[0],
            source_file=row[1],
            page_number=int(row[2]),
            chunk_type=row[3],
            section_header=row[4] or "",
            content=row[5] or "",
        )
        for row in rows
    ]


def score_fusion(
    candidate_ids: Sequence[str],
    cosine_by_id: Dict[str, float],
    bm25_by_id: Dict[str, float],
    semantic_weight: float = SEMANTIC_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT,
) -> List[str]:
    """
    Rank candidates by semantic_weight * cosine + keyword_weight * normalized BM25.

    Cosine similarity is already on a fixed scale and is used as is. BM25 is divided
    by the best BM25 score, so it falls in [0, 1]. Candidates with no BM25 match score 0
    on that part. Ties keep the candidate order.
    """
    max_bm25 = max(bm25_by_id.values(), default=0.0)

    def fused(chunk_id: str) -> float:
        keyword = bm25_by_id.get(chunk_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0
        return semantic_weight * cosine_by_id.get(chunk_id, 0.0) + keyword_weight * keyword

    scores = {chunk_id: fused(chunk_id) for chunk_id in candidate_ids}
    return sorted(candidate_ids, key=lambda chunk_id: -scores[chunk_id])


def _default_reranker() -> Reranker:
    return CrossEncoderReranker(Config.get_config().get_reranker_model())


class HybridRetriever:
    """Combines vector search and BM25 keyword search, with an optional re-ranker."""

    def __init__(
        self,
        retriever: Retriever,
        reranker_factory: Optional[Callable[[], Reranker]] = None,
    ):
        """
        Args:
            retriever: Vector retriever. Its connection and encoder are reused.
            reranker_factory: Returns the re-ranker, called once on first use.
                Defaults to the cross-encoder named in config.yaml.
        """
        self.retriever = retriever
        self._reranker_factory = reranker_factory
        self._reranker: Optional[Reranker] = None

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_files: Optional[Sequence[str]] = None,
        use_reranker: bool = False,
    ) -> RetrievalResult:
        """
        Return the best chunks for a question.

        Without the re-ranker, the result is the top_k of the fused ranking.
        With it, the fused top candidate_pool (200) is re-ranked and the best
        reranker top_k (10) are returned. top_k is ignored in that case.

        Args:
            source_files: Only search chunks from these source files. None searches all files.
            use_reranker: Re-rank the fused candidates with the cross-encoder.

        Raises:
            RetrievalError: If the question is empty, source_files is an empty list,
                k is invalid, or any search or re-ranking step fails.
        """
        if not question or not question.strip():
            raise RetrievalError("Question must not be empty")
        if source_files is not None and len(source_files) == 0:
            raise RetrievalError("At least one source file must be selected")

        if use_reranker:
            config = Config.get_config()
            pool = config.get_reranker_candidate_pool()
            final_k = config.get_reranker_top_k()
        else:
            final_k = self.retriever.top_k if top_k is None else top_k
            pool = final_k
        if final_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {final_k}")

        route = route_query(question)
        semantic = self.retriever.retrieve(question, top_k=pool, source_files=source_files)
        semantic_by_id = {chunk.chunk_id: chunk for chunk in semantic.chunks}

        keyword_ranking = self._keyword_ranking(question, route, source_files)
        bm25_by_id = {row.chunk_id: score for row, score in keyword_ranking}
        keyword_by_id = {row.chunk_id: row for row, _ in keyword_ranking[:pool]}

        candidate_ids = list(semantic_by_id) + [cid for cid in keyword_by_id if cid not in semantic_by_id]
        cosine = {chunk_id: chunk.similarity_score for chunk_id, chunk in semantic_by_id.items()}
        missing = [chunk_id for chunk_id in keyword_by_id if chunk_id not in semantic_by_id]
        if missing:
            cosine.update(self._cosine_scores(question, missing))

        fused_ids = score_fusion(candidate_ids, cosine, bm25_by_id)[:pool]

        candidates: List[RetrievedChunk] = []
        for chunk_id in fused_ids:
            if chunk_id in semantic_by_id:
                candidates.append(semantic_by_id[chunk_id])
                continue
            row = keyword_by_id[chunk_id]
            candidates.append(
                RetrievedChunk(
                    chunk_id=row.chunk_id,
                    source_file=row.source_file,
                    page_number=row.page_number,
                    chunk_type=row.chunk_type,
                    section_header=row.section_header,
                    content=row.content,
                    similarity_score=cosine.get(chunk_id, 0.0),
                )
            )

        if use_reranker:
            chunks = self._get_reranker().rerank(question, candidates, final_k)
        else:
            chunks = candidates[:final_k]

        return RetrievalResult(
            question=question,
            route=route,
            query_dimensions=semantic.query_dimensions,
            chunks=chunks,
        )

    def _keyword_ranking(
        self,
        question: str,
        route: str,
        source_files: Optional[Sequence[str]],
    ) -> List[Tuple[ChunkRow, float]]:
        """Every eligible chunk with a BM25 score, best first."""
        rows = fetch_chunk_rows(self.retriever.conn)
        if not rows:
            return []

        index = _get_bm25_index(rows)
        allowed_types = _ROUTE_CHUNK_TYPES[route]
        allowed_sources = set(source_files) if source_files is not None else None
        scores = index.scores(tokenize(question))

        eligible = [
            position
            for position in scores
            if (allowed_types is None or rows[position].chunk_type in allowed_types)
            and (allowed_sources is None or rows[position].source_file in allowed_sources)
        ]
        eligible.sort(key=lambda position: (-scores[position], position))
        return [(rows[position], scores[position]) for position in eligible]

    def _cosine_scores(self, question: str, chunk_ids: Sequence[str]) -> Dict[str, float]:
        """Cosine similarity of the question to chunks that keyword search found but vector search did not."""
        query_vector = self._query_vector(question)
        dimensions = self.retriever.dimensions
        placeholders = ", ".join("?" for _ in chunk_ids)
        sql = f"""
        SELECT id, 1.0 - array_cosine_distance(embedding, CAST(? AS FLOAT[{dimensions}])) AS similarity
        FROM {TABLE_NAME}
        WHERE id IN ({placeholders})
        """
        try:
            rows = self.retriever.conn.execute(sql, [query_vector.tolist(), *chunk_ids]).fetchall()
        except Exception as e:
            raise RetrievalError(f"Similarity lookup failed: {e}") from e
        return {row[0]: float(row[1]) for row in rows}

    def _query_vector(self, question: str) -> np.ndarray:
        try:
            vector = np.asarray(self.retriever.encoder.encode_text(question), dtype=np.float32).reshape(-1)
        except Exception as e:
            raise RetrievalError(f"Failed to embed question: {e}") from e
        if vector.shape[0] != self.retriever.dimensions:
            raise RetrievalError(
                f"Query embedding has {vector.shape[0]} dimensions, expected {self.retriever.dimensions}"
            )
        return vector

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            factory = self._reranker_factory or _default_reranker
            self._reranker = factory()
        return self._reranker

    def close(self) -> None:
        self.retriever.close()
