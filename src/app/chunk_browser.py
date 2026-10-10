"""
Read-only access to the medical_document_chunks table for the Chunk Table tab.

The embedding column is left out of every query. It holds 1,024 floats per row,
which is not useful to read in a table view.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.config import Config
from src.utils.connections import DuckDBConnection

TABLE_NAME = "medical_document_chunks"
CHUNK_PREVIEW_LIMIT = 200

TABLE_EXISTS_SQL = (
    f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{TABLE_NAME}'"
)
SOURCES_SQL = f"SELECT DISTINCT source_file FROM {TABLE_NAME} ORDER BY source_file"
# Each filter is skipped when its parameter is blank.
FILTER_SQL = """
WHERE (? = '' OR strpos(lower(content), lower(?)) > 0)
  AND (? = '' OR source_file = ?)
"""
COUNT_SQL = f"SELECT count(*) FROM {TABLE_NAME} {FILTER_SQL}"
ROWS_SQL = f"""
SELECT id, source_file, page_number, chunk_type, section_header, content
FROM {TABLE_NAME}
{FILTER_SQL}
ORDER BY source_file, page_number, id
LIMIT ?
"""


def _fetch(sql: str, params: List, db_path: Optional[str]) -> Optional[List[Tuple]]:
    """Run a read-only query. Returns None if the database file or table does not exist yet."""
    path = Path(db_path) if db_path else Config.get_config().get_duckdb_file()
    if not path.exists():
        return None

    conn = DuckDBConnection.get_connection(str(path))
    try:
        if not conn.execute(TABLE_EXISTS_SQL).fetchone()[0]:
            return None
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def list_chunk_sources(db_path: Optional[str] = None) -> List[str]:
    """Return the distinct source file names in the chunk table, sorted. Empty if unavailable."""
    rows = _fetch(SOURCES_SQL, [], db_path)
    return [row[0] for row in rows] if rows else []


def load_chunks(
    keyword: str = "",
    source_file: str = "",
    limit: int = CHUNK_PREVIEW_LIMIT,
    db_path: Optional[str] = None,
) -> Tuple[List[Dict], int]:
    """
    Read chunks, optionally filtered by text in the content and by source file.

    Args:
        keyword: Case-insensitive text to match in the chunk content. Blank means no filter.
        source_file: Exact source file name. Blank means all sources.
        limit: Maximum number of rows to return.
        db_path: Optional DuckDB file. Defaults to paths.duckdb_file in config.yaml.

    Returns:
        The rows to show (at most limit) and the total number of matching chunks.
        ([], 0) if the database file or table does not exist yet.
    """
    search = keyword.strip()
    source = source_file.strip()
    params = [search, search, source, source]

    count_rows = _fetch(COUNT_SQL, params, db_path)
    if count_rows is None:
        return [], 0
    total = int(count_rows[0][0])
    if total == 0:
        return [], 0

    rows = _fetch(ROWS_SQL, params + [limit], db_path) or []
    return [
        {
            "id": row[0],
            "source_file": row[1],
            "page_number": row[2],
            "chunk_type": row[3],
            "section_header": row[4],
            "content": row[5],
        }
        for row in rows
    ], total
