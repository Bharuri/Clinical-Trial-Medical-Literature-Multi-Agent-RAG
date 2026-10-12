from unittest.mock import MagicMock, patch

import numpy as np

from src.app.accepted_answers import save_accepted_answer
from src.app.service import find_accepted_answer


def fake_encoder(vectors: dict) -> MagicMock:
    encoder = MagicMock()
    encoder.encode_texts.side_effect = lambda texts: [vectors[t] for t in texts]
    return encoder


def test_no_saved_answers_returns_none_without_loading_the_model(tmp_path):
    db = str(tmp_path / "accepted.duckdb")

    with patch("src.app.service.get_query_encoder") as factory:
        assert find_accepted_answer("What is the hazard ratio?", db_path=db) is None

    factory.assert_not_called()


def test_exact_match_returns_saved_answer_without_loading_the_model(tmp_path):
    db = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("What is the hazard ratio?", "The hazard ratio was 0.75.", db_path=db)

    with patch("src.app.service.get_query_encoder") as factory:
        match = find_accepted_answer("what is the hazard ratio", db_path=db)

    assert match is not None
    assert match.kind == "exact"
    assert match.answer == "The hazard ratio was 0.75."
    factory.assert_not_called()


def test_similar_question_uses_the_query_encoder(tmp_path):
    db = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("What is the hazard ratio?", "The hazard ratio was 0.75.", db_path=db)
    encoder = fake_encoder(
        {
            "Which hazard ratio was reported?": np.array([1.0, 0.0]),
            "What is the hazard ratio?": np.array([0.9, float(np.sqrt(1 - 0.81))]),
        }
    )

    with patch("src.app.service.get_query_encoder", return_value=encoder):
        match = find_accepted_answer("Which hazard ratio was reported?", db_path=db)

    assert match is not None
    assert match.kind == "similar"
    assert match.answer == "The hazard ratio was 0.75."


def test_unrelated_question_returns_none(tmp_path):
    db = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("What is the hazard ratio?", "The hazard ratio was 0.75.", db_path=db)
    encoder = fake_encoder(
        {
            "How is the drug stored?": np.array([0.0, 1.0]),
            "What is the hazard ratio?": np.array([1.0, 0.0]),
        }
    )

    with patch("src.app.service.get_query_encoder", return_value=encoder):
        assert find_accepted_answer("How is the drug stored?", db_path=db) is None


def test_blank_question_returns_none(tmp_path):
    db = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("What is the hazard ratio?", "The hazard ratio was 0.75.", db_path=db)

    with patch("src.app.service.get_query_encoder") as factory:
        assert find_accepted_answer("   ", db_path=db) is None

    factory.assert_not_called()
