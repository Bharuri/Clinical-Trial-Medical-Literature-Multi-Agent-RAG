"""
Check a new question against the accepted answers before the LLM pipeline runs.

An exact match (ignoring case, extra spaces, and trailing punctuation) returns
the saved answer. A question whose embedding has cosine similarity of at least
SIMILARITY_THRESHOLD to a saved question is a candidate. The UI asks the user
to confirm it before the saved answer is shown.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

SIMILARITY_THRESHOLD = 0.8
EXACT = "exact"
SIMILAR = "similar"


@dataclass(frozen=True)
class CachedAnswer:
    kind: str
    id: int
    question: str
    answer: str
    created_at: object
    score: float


def normalize_question(text: str) -> str:
    return " ".join(text.split()).lower().rstrip("?.! ")


def _to_match(row: Dict, kind: str, score: float) -> CachedAnswer:
    return CachedAnswer(
        kind=kind,
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        created_at=row["created_at"],
        score=score,
    )


def find_cached_answer(
    question: str,
    saved_rows: Sequence[Dict],
    encode_texts: Callable[[List[str]], List[np.ndarray]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[CachedAnswer]:
    """
    Find a saved answer for the question, or None if the LLM pipeline should run.

    Args:
        question: The new question.
        saved_rows: Accepted answers, each with id, question, answer, and created_at.
        encode_texts: Returns one L2-normalized embedding per text, in order.
            The embedding model is only called when no exact match exists.
        threshold: Minimum cosine similarity for a similar match.

    Returns:
        A CachedAnswer with kind EXACT (score 1.0) or SIMILAR, or None.
    """
    if not saved_rows or not question.strip():
        return None

    target = normalize_question(question)
    for row in saved_rows:
        if normalize_question(row["question"]) == target:
            return _to_match(row, EXACT, 1.0)

    vectors = encode_texts([question] + [row["question"] for row in saved_rows])
    query, saved_vectors = vectors[0], vectors[1:]
    scores = [float(np.dot(query, vector)) for vector in saved_vectors]
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return None
    return _to_match(saved_rows[best], SIMILAR, scores[best])
