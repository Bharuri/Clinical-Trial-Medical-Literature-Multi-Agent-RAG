"""
Tests for connection setup (DuckDB and Bedrock).
"""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.utils.config import Config
from src.utils.connections import BedrockClient, DuckDBConnection


@pytest.fixture
def temp_project_root() -> Generator[Path, None, None]:
    """Create a temporary project directory with config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Create subdirectories
        (project_root / "data" / "pdfs").mkdir(parents=True)
        (project_root / "database").mkdir(parents=True)
        (project_root / "src").mkdir(parents=True)
        
        # Create a minimal config.yaml
        config_data = {
            "paths": {
                "pdf_directory": "data/pdfs",
                "database_directory": "database",
                "duckdb_file": "database/medical_rag.duckdb",
            },
            "embedding": {"model_name": "test-model", "dimensions": 1024},
            "chunking": {"chunk_size_tokens": 750, "chunk_overlap_tokens": 150},
            "retrieval": {"top_k": 5},
            "bedrock": {
                "default_model_id": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "default_region": "us-east-1",
            },
            "duckdb": {"vss_extension": "vss", "index_distance_metric": "cosine"},
        }
        
        config_path = project_root / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        yield project_root


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the config singleton before each test."""
    Config.reset()
    yield
    Config.reset()


class TestDuckDBConnection:
    """Test DuckDB connection creation and management."""

    def test_get_connection_creates_connection(self, temp_project_root: Path) -> None:
        """Should create a DuckDB connection."""
        config_path = temp_project_root / "config.yaml"
        Config(config_path)
        
        db_path = str(temp_project_root / "database" / "test.duckdb")
        conn = DuckDBConnection.get_connection(db_path)
        
        assert conn is not None
        # Verify connection is usable
        result = conn.execute("SELECT 1 AS test").fetchall()
        assert result == [(1,)]

    def test_get_connection_uses_config_path(self, temp_project_root: Path) -> None:
        """Should use DuckDB path from config if not provided."""
        config_path = temp_project_root / "config.yaml"
        Config.get_config(config_path)
        try:
            conn = DuckDBConnection.get_connection()
            assert conn is not None
            result = conn.execute("SELECT 1 AS test").fetchall()
            conn.close()
        finally:
            Config.reset()

        assert result == [(1,)]

    def test_ensure_db_directory_creates_directory(
        self, temp_project_root: Path
    ) -> None:
        """Should create database directory if it does not exist."""
        config_path = temp_project_root / "config.yaml"
        Config(config_path)
        
        new_db_path = str(temp_project_root / "new_db" / "test.duckdb")
        DuckDBConnection.ensure_db_directory(new_db_path)
        
        assert Path(new_db_path).parent.exists()

    def test_ensure_db_directory_uses_config(self, temp_project_root: Path) -> None:
        """Should use database directory from config if not provided."""
        config_path = temp_project_root / "config.yaml"
        Config(config_path)
        
        DuckDBConnection.ensure_db_directory()
        
        db_dir = temp_project_root / "database"
        assert db_dir.exists()


class TestBedrockClient:
    """Test Bedrock client creation."""

    def test_create_client_with_valid_credentials(self) -> None:
        """Should create Bedrock client with valid credentials."""
        with patch("boto3.client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            client = BedrockClient.create_client(
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="us-east-1",
            )
            
            assert client == mock_instance
            mock_client.assert_called_once_with(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
            )

    def test_create_client_missing_access_key(self) -> None:
        """Should raise error if access key is missing."""
        with pytest.raises(ValueError, match="AWS credentials.*are required"):
            BedrockClient.create_client(
                aws_access_key_id="",
                aws_secret_access_key="test-secret",
                region_name="us-east-1",
            )

    def test_create_client_missing_secret_key(self) -> None:
        """Should raise error if secret key is missing."""
        with pytest.raises(ValueError, match="AWS credentials.*are required"):
            BedrockClient.create_client(
                aws_access_key_id="test-key",
                aws_secret_access_key="",
                region_name="us-east-1",
            )

    def test_create_client_missing_region(self) -> None:
        """Should raise error if region is missing."""
        with pytest.raises(ValueError, match="AWS region is required"):
            BedrockClient.create_client(
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="",
            )

    def test_create_client_boto3_error(self) -> None:
        """Should raise error if boto3 client creation fails."""
        with patch("boto3.client", side_effect=Exception("Connection failed")):
            with pytest.raises(ValueError, match="Failed to create Bedrock client"):
                BedrockClient.create_client(
                    aws_access_key_id="test-key",
                    aws_secret_access_key="test-secret",
                    region_name="us-east-1",
                )

    def test_get_default_credentials_and_model(self, temp_project_root: Path) -> None:
        """Should return default model and region from config."""
        config_path = temp_project_root / "config.yaml"
        Config.get_config(config_path)
        try:
            model_id, region = BedrockClient.get_default_credentials_and_model()
        finally:
            Config.reset()

        assert model_id == "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert region == "us-east-1"
