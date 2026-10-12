"""
Storage for questions that a user adds from the Chunk Table tab.

The questions table lives in the same DuckDB file as medical_document_chunks,
so it sits next to the chunk table and is kept when ingestion rebuilds that table.
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import Config
from src.utils.connections import DuckDBConnection

TABLE_NAME = "questions"
CREATE_SEQUENCE_SQL = "CREATE SEQUENCE IF NOT EXISTS questions_id_seq"
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGINT PRIMARY KEY DEFAULT nextval('questions_id_seq'),
    question TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""
INSERT_SQL = f"INSERT INTO {TABLE_NAME} (question) VALUES (?) RETURNING id"
TABLE_EXISTS_SQL = (
    f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{TABLE_NAME}'"
)
SELECT_SQL = f"SELECT id, created_at, question FROM {TABLE_NAME} ORDER BY id DESC"


def _resolve_path(db_path: Optional[str]) -> str:
    return db_path or str(Config.get_config().get_duckdb_file())


def ensure_questions_table(db_path: Optional[str] = None) -> None:
    """Create the questions table if it does not exist yet."""
    path = _resolve_path(db_path)
    DuckDBConnection.ensure_db_directory(path)
    conn = DuckDBConnection.get_connection(path)
    try:
        conn.execute(CREATE_SEQUENCE_SQL)
        conn.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


def save_question(question: str, db_path: Optional[str] = None) -> int:
    """
    Add a question to the questions table.

    Args:
        question: The question text.
        db_path: Optional DuckDB file. Defaults to paths.duckdb_file in config.yaml.

    Returns:
        The id of the new row.

    Raises:
        ValueError: If the question is empty.
    """
    text = question.strip()
    if not text:
        raise ValueError("Cannot add an empty question.")

    path = _resolve_path(db_path)
    ensure_questions_table(path)
    conn = DuckDBConnection.get_connection(path)
    try:
        row = conn.execute(INSERT_SQL, [text]).fetchone()
    finally:
        conn.close()
    return int(row[0])


def load_questions(db_path: Optional[str] = None) -> List[Dict]:
    """
    Read stored questions, newest first.

    Args:
        db_path: Optional DuckDB file. Defaults to paths.duckdb_file in config.yaml.

    Returns:
        One dict per row. An empty list if the file or table does not exist yet.
    """
    path = Path(_resolve_path(db_path))
    if not path.exists():
        return []

    conn = DuckDBConnection.get_connection(str(path))
    try:
        if not conn.execute(TABLE_EXISTS_SQL).fetchone()[0]:
            return []
        rows = conn.execute(SELECT_SQL).fetchall()
    finally:
        conn.close()

    return [{"id": row[0], "created_at": row[1], "question": row[2]} for row in rows]
