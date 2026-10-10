"""
Storage for answers that a user has manually accepted in the UI.

The accepted answers live in their own DuckDB file, separate from the
document-chunk database, so re-running ingestion never touches them.
"""

from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import Config
from src.utils.connections import DuckDBConnection

CREATE_SEQUENCE_SQL = "CREATE SEQUENCE IF NOT EXISTS accepted_answers_id_seq"
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS accepted_answers (
    id BIGINT PRIMARY KEY DEFAULT nextval('accepted_answers_id_seq'),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""
INSERT_SQL = "INSERT INTO accepted_answers (question, answer) VALUES (?, ?) RETURNING id"
TABLE_EXISTS_SQL = (
    "SELECT count(*) FROM information_schema.tables WHERE table_name = 'accepted_answers'"
)
SELECT_SQL = """
SELECT id, created_at, question, answer
FROM accepted_answers
WHERE ? = '' OR strpos(lower(question), lower(?)) > 0 OR strpos(lower(answer), lower(?)) > 0
ORDER BY id DESC
"""


def save_accepted_answer(question: str, answer: str, db_path: Optional[str] = None) -> int:
    """
    Store a question and the answer the user accepted.

    Args:
        question: The research question as asked.
        answer: The answer text the user accepted, with its citations.
        db_path: Optional DuckDB file. Defaults to paths.accepted_answers_duckdb_file in config.yaml.

    Returns:
        The id of the stored row.

    Raises:
        ValueError: If the question or answer is empty.
    """
    if not question.strip():
        raise ValueError("Cannot accept an answer without a question.")
    if not answer.strip():
        raise ValueError("Cannot accept an empty answer.")

    path = db_path or str(Config.get_config().get_accepted_answers_file())
    DuckDBConnection.ensure_db_directory(path)
    conn = DuckDBConnection.get_connection(path)
    try:
        conn.execute(CREATE_SEQUENCE_SQL)
        conn.execute(CREATE_TABLE_SQL)
        row = conn.execute(INSERT_SQL, [question.strip(), answer.strip()]).fetchone()
    finally:
        conn.close()
    return int(row[0])


def load_accepted_answers(keyword: str = "", db_path: Optional[str] = None) -> List[Dict]:
    """
    Read stored accepted answers, newest first.

    Args:
        keyword: Optional case-insensitive text to match in the question or answer.
        db_path: Optional DuckDB file. Defaults to paths.accepted_answers_duckdb_file in config.yaml.

    Returns:
        One dict per row. An empty list if the file or table does not exist yet.
    """
    path = Path(db_path) if db_path else Config.get_config().get_accepted_answers_file()
    if not path.exists():
        return []

    search = keyword.strip()
    conn = DuckDBConnection.get_connection(str(path))
    try:
        table_count = conn.execute(TABLE_EXISTS_SQL).fetchone()[0]
        if not table_count:
            return []
        rows = conn.execute(SELECT_SQL, [search, search, search]).fetchall()
    finally:
        conn.close()

    return [
        {"id": row[0], "created_at": row[1], "question": row[2], "answer": row[3]}
        for row in rows
    ]
