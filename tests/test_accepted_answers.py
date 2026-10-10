"""
Tests for the accepted-answers store. Every test writes to a temporary
DuckDB file, never to the real database in database/.
"""

from pathlib import Path

import duckdb
import pytest
import yaml

from src.app.accepted_answers import load_accepted_answers, save_accepted_answer
from src.utils.config import Config


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


def read_rows(db_file: Path):
    conn = duckdb.connect(str(db_file))
    try:
        return conn.execute(
            "SELECT id, question, answer FROM accepted_answers ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def test_saves_question_and_answer_and_returns_id(tmp_path):
    db_file = tmp_path / "accepted.duckdb"

    first_id = save_accepted_answer("What was the hazard ratio?", "It was 0.75 [Source: t.pdf, p. 2].", str(db_file))
    second_id = save_accepted_answer("Second question?", "Second answer.", str(db_file))

    assert first_id == 1
    assert second_id == 2
    assert read_rows(db_file) == [
        (1, "What was the hazard ratio?", "It was 0.75 [Source: t.pdf, p. 2]."),
        (2, "Second question?", "Second answer."),
    ]


def test_creates_missing_database_directory(tmp_path):
    db_file = tmp_path / "nested" / "dir" / "accepted.duckdb"

    save_accepted_answer("Q?", "A.", str(db_file))

    assert db_file.exists()


def test_strips_surrounding_whitespace(tmp_path):
    db_file = tmp_path / "accepted.duckdb"

    save_accepted_answer("  Q?  ", "\n A. \n", str(db_file))

    assert read_rows(db_file) == [(1, "Q?", "A.")]


@pytest.mark.parametrize("question, answer", [("", "A."), ("   ", "A."), ("Q?", ""), ("Q?", "  \n")])
def test_rejects_empty_question_or_answer(tmp_path, question, answer):
    db_file = tmp_path / "accepted.duckdb"

    with pytest.raises(ValueError):
        save_accepted_answer(question, answer, str(db_file))

    assert not db_file.exists()


def test_default_path_comes_from_config_and_not_the_rag_database(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "paths": {
                    "pdf_directory": "data/pdfs",
                    "database_directory": "database",
                    "duckdb_file": "database/medical_rag.duckdb",
                    "accepted_answers_duckdb_file": "database/accepted_answers.duckdb",
                },
            }
        ),
        encoding="utf-8",
    )
    Config.get_config(config_path)

    save_accepted_answer("Q?", "A.")

    accepted_file = tmp_path / "database" / "accepted_answers.duckdb"
    assert read_rows(accepted_file) == [(1, "Q?", "A.")]
    assert not (tmp_path / "database" / "medical_rag.duckdb").exists()


def test_load_returns_empty_list_when_file_missing(tmp_path):
    assert load_accepted_answers(db_path=str(tmp_path / "missing.duckdb")) == []


def test_load_returns_empty_list_when_table_missing(tmp_path):
    db_file = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_file)).close()

    assert load_accepted_answers(db_path=str(db_file)) == []


def test_load_returns_newest_first(tmp_path):
    db_file = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("First question?", "First answer.", db_file)
    save_accepted_answer("Second question?", "Second answer.", db_file)

    rows = load_accepted_answers(db_path=db_file)

    assert [r["id"] for r in rows] == [2, 1]
    assert rows[0]["question"] == "Second question?"
    assert rows[0]["answer"] == "Second answer."
    assert rows[0]["created_at"] is not None


def test_load_filters_by_keyword_in_question_or_answer_case_insensitively(tmp_path):
    db_file = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("Placebo question?", "Ethics answer.", db_file)
    save_accepted_answer("Other question?", "Mentions PLACEBO in the answer.", db_file)
    save_accepted_answer("Unrelated?", "Unrelated.", db_file)

    rows = load_accepted_answers(keyword="placebo", db_path=db_file)

    assert sorted(r["id"] for r in rows) == [1, 2]


def test_load_treats_blank_keyword_as_no_filter(tmp_path):
    db_file = str(tmp_path / "accepted.duckdb")
    save_accepted_answer("Q1?", "A1.", db_file)
    save_accepted_answer("Q2?", "A2.", db_file)

    assert len(load_accepted_answers(keyword="   ", db_path=db_file)) == 2
