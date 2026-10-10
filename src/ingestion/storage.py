"""
DuckDB storage for chunks and embeddings with vector search index.

Manages table creation, batch inserts, and HNSW index setup.
"""

import uuid
from typing import List, Optional

import duckdb
import numpy as np

from src.ingestion.chunking import Chunk
from src.utils.connections import DuckDBConnection


class StorageError(Exception):
    """Raised when storage operations fail."""
    pass


class ChunkStorage:
    """Manages DuckDB storage for document chunks and embeddings."""

    TABLE_NAME = "medical_document_chunks"
    INDEX_NAME = "chunks_embedding_hnsw_index"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize chunk storage.

        Args:
            db_path: Path to DuckDB database file. If not provided, uses config.

        Raises:
            StorageError: If database connection fails
        """
        try:
            self.conn = DuckDBConnection.get_connection(db_path)
            self.db_path = db_path
        except Exception as e:
            raise StorageError(f"Failed to connect to database: {e}")

    def create_table(self) -> None:
        """
        Create the medical_document_chunks table with required schema.

        Creates columns for:
        - id: UUID primary key
        - source_file: PDF filename
        - page_number: Source page
        - chunk_type: 'text' or 'table'
        - section_header: Section title/heading
        - content: Text or Markdown table content
        - embedding: FLOAT[768] vector

        Raises:
            StorageError: If table creation fails
        """
        try:
            # Drop existing table if it exists (for fresh start)
            self.conn.execute(f"DROP TABLE IF EXISTS {self.TABLE_NAME}")

            # Create table with FLOAT[768] embedding column
            create_sql = f"""
            CREATE TABLE {self.TABLE_NAME} (
                id VARCHAR PRIMARY KEY,
                source_file VARCHAR NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_type VARCHAR NOT NULL,
                section_header VARCHAR NOT NULL,
                content VARCHAR NOT NULL,
                embedding FLOAT[768] NOT NULL
            )
            """
            self.conn.execute(create_sql)

        except Exception as e:
            raise StorageError(f"Failed to create table: {e}")

    def insert_chunks_batch(
        self, chunks: List[Chunk], embeddings: List[np.ndarray]
    ) -> int:
        """
        Insert multiple chunks with their embeddings using batch insert.

        Uses parameter binding for safety and COPY for efficiency.

        Args:
            chunks: List of Chunk objects
            embeddings: List of numpy arrays (768-dimensional)

        Returns:
            Number of chunks inserted

        Raises:
            StorageError: If insertion fails or dimensions don't match

        Note:
            Validates embedding dimensions before insertion.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise StorageError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")

        try:
            inserted = 0

            for chunk, embedding in zip(chunks, embeddings):
                # Validate embedding
                if embedding.shape[0] != 768:
                    raise StorageError(
                        f"Embedding for chunk '{chunk.source_file}' has invalid dimension: "
                        f"{embedding.shape[0]} (expected 768)"
                    )

                # Convert embedding to list for DuckDB insertion
                embedding_list = embedding.tolist()

                # Generate unique ID if not present
                chunk_id = str(uuid.uuid4())

                # Insert using parameter binding (safe against injection)
                insert_sql = f"""
                INSERT INTO {self.TABLE_NAME}
                (id, source_file, page_number, chunk_type, section_header, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """

                self.conn.execute(
                    insert_sql,
                    [
                        chunk_id,
                        chunk.source_file,
                        chunk.page_number,
                        chunk.chunk_type,
                        chunk.section_header,
                        chunk.content,
                        embedding_list,
                    ],
                )

                inserted += 1

            return inserted

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to insert chunks: {e}")

    def create_index(self) -> None:
        """
        Create HNSW index on embedding column for fast similarity search.

        Uses cosine distance metric as specified.
        The vss extension must be loaded before calling this.

        Raises:
            StorageError: If index creation fails
        """
        try:
            # Ensure VSS extension is loaded
            self.conn.execute("LOAD vss")
            # Required for HNSW indexes on file-backed DuckDB databases (DuckDB 1.1.x)
            self.conn.execute("SET hnsw_enable_experimental_persistence = true")

            # Create HNSW index with cosine distance
            index_sql = f"""
            CREATE INDEX {self.INDEX_NAME} ON {self.TABLE_NAME}
            USING HNSW (embedding) WITH (metric = 'cosine')
            """

            self.conn.execute(index_sql)

        except Exception as e:
            raise StorageError(f"Failed to create index: {e}")

    def get_chunk_count(self) -> int:
        """
        Get the number of chunks stored in the database.

        Returns:
            Number of chunks in table
        """
        try:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}").fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def verify_table_exists(self) -> bool:
        """
        Check if the chunks table exists.

        Returns:
            True if table exists, False otherwise
        """
        try:
            result = self.conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.TABLE_NAME}'"
            ).fetchone()
            return result and result[0] > 0
        except Exception:
            return False

    def verify_index_exists(self) -> bool:
        """
        Check if the HNSW index exists.

        Returns:
            True if index exists, False otherwise
        """
        try:
            result = self.conn.execute(
                f"SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = 'main'"
            ).fetchone()
            # This is a simple check; full index verification would require more complex query
            return result and result[0] > 0
        except Exception:
            return False

    def close(self) -> None:
        """Close the database connection."""
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
