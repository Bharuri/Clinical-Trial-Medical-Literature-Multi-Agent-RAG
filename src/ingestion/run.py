"""
Main ingestion pipeline runner.

Independently runnable script that:
1. Discovers PDFs in configured directory
2. Extracts narrative text and tables
3. Chunks and creates metadata
4. Generates embeddings
5. Stores in DuckDB with vector index

Can be run directly:
    python -m src.ingestion.run
"""

import logging
import sys
from pathlib import Path
from typing import List

from src.ingestion.chunking import TextChunker, Chunk
from src.ingestion.embeddings import EmbeddingGenerator
from src.ingestion.pdf_extraction import PDFExtractor
from src.ingestion.storage import ChunkStorage
from src.utils.config import Config, ConfigurationError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Coordinates PDF ingestion, chunking, embedding, and storage."""

    def __init__(self):
        """
        Initialize the ingestion pipeline.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.config = Config.get_config()
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise

        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker(
            chunk_size_tokens=self.config.get_chunk_size(),
            chunk_overlap_tokens=self.config.get_chunk_overlap(),
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.get_embedding_model(),
            expected_dimensions=self.config.get_embedding_dimensions(),
        )
        self.storage = ChunkStorage()

    def run(self) -> bool:
        """
        Execute the complete ingestion pipeline.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting PDF ingestion pipeline")

            # Get PDF files
            pdf_dir = self.config.get_pdf_directory()
            logger.info(f"Looking for PDFs in: {pdf_dir}")

            if not pdf_dir.exists():
                logger.error(f"PDF directory does not exist: {pdf_dir}")
                return False

            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {pdf_dir}")
                return False

            logger.info(f"Found {len(pdf_files)} PDF files to process")

            # Create table
            logger.info("Creating DuckDB table")
            self.storage.create_table()

            # Process each PDF
            total_chunks = 0
            for pdf_file in sorted(pdf_files):
                logger.info(f"Processing: {pdf_file.name}")
                chunks_count = self._process_pdf(pdf_file)
                total_chunks += chunks_count
                logger.info(f"  Extracted {chunks_count} chunks from {pdf_file.name}")

            # Create index
            logger.info("Creating HNSW vector index")
            self.storage.create_index()

            # Verify results
            final_count = self.storage.get_chunk_count()
            logger.info(f"Ingestion complete. Total chunks stored: {final_count}")

            if final_count == 0:
                logger.warning("No chunks were stored in the database")
                return False

            return True

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return False
        finally:
            self.storage.close()

    def _process_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF file through the full pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of chunks extracted and stored
        """
        try:
            # Extract text and tables from PDF
            pdf_contents = self.pdf_extractor.extract_pdf(pdf_path)
            if not pdf_contents:
                logger.warning(f"No content extracted from {pdf_path.name}")
                return 0

            all_chunks: List[Chunk] = []
            all_embeddings: List = []

            # Process each page
            for page_content in pdf_contents:
                # Chunk narrative text
                if page_content.narrative_text:
                    text_chunks = self.text_chunker.chunk_narrative_text(
                        text=page_content.narrative_text,
                        source_file=pdf_path.name,
                        page_number=page_content.page_number,
                        section_header=page_content.section_header,
                    )
                    all_chunks.extend(text_chunks)

                # Create table chunks (one per table)
                for table_markdown in page_content.tables:
                    table_chunk = self.text_chunker.create_table_chunk(
                        table_markdown=table_markdown,
                        source_file=pdf_path.name,
                        page_number=page_content.page_number,
                        section_header=page_content.section_header,
                    )
                    all_chunks.append(table_chunk)

            if not all_chunks:
                logger.warning(f"No chunks created from {pdf_path.name}")
                return 0

            # Generate embeddings for all chunks
            texts_to_embed = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_generator.encode_texts(texts_to_embed)

            if len(embeddings) != len(all_chunks):
                logger.error(f"Embedding count ({len(embeddings)}) != chunk count ({len(all_chunks)})")
                return 0

            # Validate all embeddings
            for embedding in embeddings:
                self.embedding_generator.validate_embedding(embedding)

            # Store in DuckDB
            inserted = self.storage.insert_chunks_batch(all_chunks, embeddings)
            return inserted

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return 0


def main() -> int:
    """
    Main entry point for ingestion pipeline.

    Returns:
        0 if successful, 1 if failed
    """
    pipeline = IngestionPipeline()
    success = pipeline.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
