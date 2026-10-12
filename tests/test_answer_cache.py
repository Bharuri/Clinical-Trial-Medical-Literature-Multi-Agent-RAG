from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval_agent.answer_cache import (
    EXACT,
    SIMILAR,
    SIMILARITY_THRESHOLD,
    find_cached_answer,
    normalize_question,
)

SAVED = [
    {
        "id": 1,
        "question": "What is the hazard ratio?",
        "answer": "The hazard ratio was 0.75 [trial.pdf, p.2].",
        "created_at": datetime(2024, 5, 1, 10, 30),
    },
    {
        "id": 2,
        "question": "What are the common adverse events?",
        "answer": "Nausea and headache [safety.pdf, p.4].",
        "created_at": datetime(2024, 5, 2, 9, 0),
    },
]


def encoder_for(vectors: dict):
    """Fake encode_texts that returns a fixed vector per text and records its calls."""
    return MagicMock(side_effect=lambda texts: [vectors[t] for t in texts])


def vector_at_similarity(score: float) -> np.ndarray:
    """A unit vector whose dot product with [1, 0] is exactly `score`."""
    return np.array([score, float(np.sqrt(1 - score * score))])


def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_question("  What   IS the hazard ratio?? ") == "what is the hazard ratio"


def test_exact_match_ignores_case_and_punctuation_and_skips_the_encoder():
    encode = MagicMock()

    match = find_cached_answer("what is the HAZARD ratio", SAVED, encode)

    assert match is not None
    assert match.kind == EXACT
    assert match.id == 1
    assert match.answer == SAVED[0]["answer"]
    assert match.score == 1.0
    encode.assert_not_called()


def test_similar_match_at_threshold_is_returned():
    encode = encoder_for(
        {
            "What is the hazard ratio in the study?": np.array([1.0, 0.0]),
            "What is the hazard ratio?": vector_at_similarity(SIMILARITY_THRESHOLD),
            "What are the common adverse events?": np.array([0.0, 1.0]),
        }
    )

    match = find_cached_answer("What is the hazard ratio in the study?", SAVED, encode)

    assert match is not None
    assert match.kind == SIMILAR
    assert match.id == 1
    assert match.score == pytest.approx(SIMILARITY_THRESHOLD)


def test_similarity_below_threshold_returns_none():
    encode = encoder_for(
        {
            "Which dose was used?": np.array([1.0, 0.0]),
            "What is the hazard ratio?": vector_at_similarity(0.79),
            "What are the common adverse events?": np.array([0.0, 1.0]),
        }
    )

    assert find_cached_answer("Which dose was used?", SAVED, encode) is None


def test_best_similar_question_is_chosen_when_several_pass_the_threshold():
    encode = encoder_for(
        {
            "Describe the side effects": np.array([1.0, 0.0]),
            "What is the hazard ratio?": vector_at_similarity(0.85),
            "What are the common adverse events?": vector_at_similarity(0.82),
        }
    )

    match = find_cached_answer("Describe the side effects", SAVED, encode)

    assert match is not None
    assert match.id == 1
    assert match.score == pytest.approx(0.85)


def test_empty_store_returns_none_without_encoding():
    encode = MagicMock()

    assert find_cached_answer("What is the hazard ratio?", [], encode) is None
    encode.assert_not_called()


def test_blank_question_returns_none_without_encoding():
    encode = MagicMock()

    assert find_cached_answer("   ", SAVED, encode) is None
    encode.assert_not_called()
