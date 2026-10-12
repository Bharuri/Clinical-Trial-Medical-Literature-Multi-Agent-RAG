"""
Tests for the questions store. Every test writes to a temporary DuckDB file,
never to the real database in database/.
"""

from pathlib import Path

import duckdb
import pytest

from src.app.questions import load_questions, save_question
from src.utils.config import Config


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


def read_rows(db_file: Path):
    conn = duckdb.connect(str(db_file))
    try:
        return conn.execute("SELECT id, question FROM questions ORDER BY id").fetchall()
    finally:
        conn.close()


def test_saves_questions_and_returns_ids(tmp_path):
    db_file = tmp_path / "rag.duckdb"

    first_id = save_question("What was the hazard ratio?", str(db_file))
    second_id = save_question("Second question?", str(db_file))

    assert first_id == 1
    assert second_id == 2
    assert read_rows(db_file) == [(1, "What was the hazard ratio?"), (2, "Second question?")]


def test_strips_surrounding_whitespace(tmp_path):
    db_file = tmp_path / "rag.duckdb"

    save_question("  Q?  ", str(db_file))

    assert read_rows(db_file) == [(1, "Q?")]


def test_rejects_blank_question(tmp_path):
    db_file = tmp_path / "rag.duckdb"

    with pytest.raises(ValueError):
        save_question("   ", str(db_file))


def test_creates_missing_database_directory(tmp_path):
    db_file = tmp_path / "nested" / "dir" / "rag.duckdb"

    save_question("Q?", str(db_file))

    assert db_file.exists()


def test_load_returns_newest_first(tmp_path):
    db_file = tmp_path / "rag.duckdb"
    save_question("older", str(db_file))
    save_question("newer", str(db_file))

    rows = load_questions(str(db_file))

    assert [row["question"] for row in rows] == ["newer", "older"]
    assert set(rows[0]) == {"id", "created_at", "question"}


def test_load_returns_empty_when_file_missing(tmp_path):
    assert load_questions(str(tmp_path / "missing.duckdb")) == []


def test_load_returns_empty_when_table_missing(tmp_path):
    db_file = tmp_path / "rag.duckdb"
    duckdb.connect(str(db_file)).close()

    assert load_questions(str(db_file)) == []
