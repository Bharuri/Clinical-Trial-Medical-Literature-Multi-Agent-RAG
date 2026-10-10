"""
Reusable connection setup for DuckDB, Bedrock, and other services.
"""

from typing import Optional

import boto3
import duckdb

from .config import Config, ConfigurationError


class DuckDBConnection:
    """Manages DuckDB connections for the application."""

    @staticmethod
    def get_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
        """
        Get a DuckDB connection.

        Args:
            db_path: Path to the DuckDB database file. If not provided,
                    uses the path from config.yaml.

        Returns:
            DuckDB connection object.

        Raises:
            ConfigurationError: If database path is not configured.
        """
        if db_path is None:
            config = Config.get_config()
            db_path = str(config.get_duckdb_file())
        
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Load VSS extension for vector search
        try:
            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")
        except Exception as e:
            # Extension may already be loaded or unavailable
            # Continue anyway; the extension will be needed when storing vectors
            pass
        
        return conn

    @staticmethod
    def ensure_db_directory(db_path: Optional[str] = None) -> None:
        """
        Ensure the database directory exists.

        Args:
            db_path: Path to the DuckDB file. If not provided, uses config.
        """
        if db_path is None:
            config = Config.get_config()
            config.get_database_directory()  # This creates the directory
        else:
            from pathlib import Path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


class BedrockClient:
    """Manages AWS Bedrock client creation."""

    @staticmethod
    def create_client(
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str
    ) -> "boto3.client":
        """
        Create a Bedrock runtime client with provided credentials.

        Args:
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            region_name: AWS region name.

        Returns:
            Bedrock runtime client.

        Raises:
            ValueError: If credentials are invalid or missing.
        """
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS credentials (access key and secret key) are required")
        
        if not region_name:
            raise ValueError("AWS region is required")
        
        try:
            client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            return client
        except Exception as e:
            raise ValueError(f"Failed to create Bedrock client: {e}")

    @staticmethod
    def get_default_credentials_and_model() -> tuple:
        """
        Get default Bedrock model and region from configuration.

        Returns:
            Tuple of (model_id, region_name).

        Raises:
            ConfigurationError: If configuration is missing.
        """
        config = Config.get_config()
        model_id = config.get_default_bedrock_model()
        region = config.get_default_region()
        return model_id, region
