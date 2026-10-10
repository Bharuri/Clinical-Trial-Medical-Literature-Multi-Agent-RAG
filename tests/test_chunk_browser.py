"""
Tests for the read-only chunk browser. Every test writes to a temporary
DuckDB file, so the real RAG database is never read or changed.
"""

from pathlib import Path

import duckdb
import pytest
import yaml

from src.app.chunk_browser import list_chunk_sources, load_chunks
from src.utils.config import Config

CHUNK_TABLE_SQL = """
CREATE TABLE medical_document_chunks (
    id VARCHAR PRIMARY KEY,
    source_file VARCHAR NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type VARCHAR NOT NULL,
    section_header VARCHAR NOT NULL,
    content VARCHAR NOT NULL,
    embedding FLOAT[768] NOT NULL
)
"""

ROWS = [
    ("b-1", "trial.pdf", 2, "text", "Results", "The hazard ratio was 0.75."),
    ("a-2", "dosing.pdf", 1, "text", "Dosing", "Placebo was given daily."),
    ("a-1", "trial.pdf", 1, "table", "Baseline", "Placebo group had 42 patients."),
]


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


def make_db(db_file: Path, rows=ROWS) -> str:
    conn = duckdb.connect(str(db_file))
    try:
        conn.execute(CHUNK_TABLE_SQL)
        for row in rows:
            conn.execute(
                "INSERT INTO medical_document_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [*row, [0.0] * 768],
            )
    finally:
        conn.close()
    return str(db_file)


def test_missing_database_returns_empty(tmp_path):
    missing = str(tmp_path / "missing.duckdb")

    assert load_chunks(db_path=missing) == ([], 0)
    assert list_chunk_sources(db_path=missing) == []
    assert not (tmp_path / "missing.duckdb").exists()


def test_missing_table_returns_empty(tmp_path):
    db_file = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_file)).close()

    assert load_chunks(db_path=str(db_file)) == ([], 0)
    assert list_chunk_sources(db_path=str(db_file)) == []


def test_returns_rows_ordered_by_source_page_and_id_without_embedding(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    rows, total = load_chunks(db_path=db_path)

    assert total == 3
    assert [row["id"] for row in rows] == ["a-2", "a-1", "b-1"]
    assert set(rows[0]) == {"id", "source_file", "page_number", "chunk_type", "section_header", "content"}


def test_keyword_filter_is_case_insensitive_over_content(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    rows, total = load_chunks(keyword="PLACEBO", db_path=db_path)

    assert total == 2
    assert {row["id"] for row in rows} == {"a-1", "a-2"}


def test_source_filter_matches_exact_file_name(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    rows, total = load_chunks(source_file="trial.pdf", db_path=db_path)

    assert total == 2
    assert all(row["source_file"] == "trial.pdf" for row in rows)


def test_keyword_and_source_filters_combine(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    rows, total = load_chunks(keyword="placebo", source_file="trial.pdf", db_path=db_path)

    assert total == 1
    assert rows[0]["id"] == "a-1"


def test_blank_filters_mean_no_filter(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    _, total = load_chunks(keyword="   ", source_file="", db_path=db_path)

    assert total == 3


def test_no_match_returns_zero_total(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    assert load_chunks(keyword="nothing matches this", db_path=db_path) == ([], 0)


def test_limit_caps_rows_but_reports_full_total(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    rows, total = load_chunks(limit=2, db_path=db_path)

    assert total == 3
    assert len(rows) == 2


def test_list_sources_is_distinct_and_sorted(tmp_path):
    db_path = make_db(tmp_path / "rag.duckdb")

    assert list_chunk_sources(db_path=db_path) == ["dosing.pdf", "trial.pdf"]


def test_default_path_comes_from_config(tmp_path):
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
    (tmp_path / "database").mkdir()
    make_db(tmp_path / "database" / "medical_rag.duckdb")

    _, total = load_chunks()

    assert total == 3
    assert list_chunk_sources() == ["dosing.pdf", "trial.pdf"]
