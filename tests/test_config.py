"""
Tests for configuration loading and path handling.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml

from src.utils.config import Config, ConfigurationError, get_config


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
                "accepted_answers_duckdb_file": "database/accepted_answers.duckdb",
            },
            "embedding": {
                "model_name": "sentence-transformers/test-model",
                "dimensions": 768,
            },
            "chunking": {
                "chunk_size_tokens": 750,
                "chunk_overlap_tokens": 150,
            },
            "retrieval": {
                "top_k": 5,
            },
            "bedrock": {
                "default_model_id": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "default_region": "us-east-1",
                "regions": ["us-east-1", "us-west-2"],
            },
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


class TestConfigurationLoading:
    """Test configuration file loading and parsing."""

    def test_config_loads_successfully(self, temp_project_root: Path) -> None:
        """Configuration should load from valid config.yaml."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        assert config.get_embedding_model() == "sentence-transformers/test-model"
        assert config.get_embedding_dimensions() == 768

    def test_config_file_not_found(self) -> None:
        """Configuration should raise error if file does not exist."""
        with pytest.raises(ConfigurationError, match="config.yaml not found"):
            Config(Path("/nonexistent/config.yaml"))

    def test_config_invalid_yaml(self, temp_project_root: Path) -> None:
        """Configuration should raise error for invalid YAML."""
        config_path = temp_project_root / "config.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")  # Malformed YAML
        
        with pytest.raises(ConfigurationError, match="Failed to load config.yaml"):
            Config(config_path)


class TestPathResolution:
    """Test that paths are resolved correctly from project root."""

    def test_relative_paths_resolved_from_project_root(
        self, temp_project_root: Path
    ) -> None:
        """Relative paths should be resolved from project root (config.yaml location)."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        pdf_dir = config.get_pdf_directory()
        assert pdf_dir == temp_project_root / "data" / "pdfs"
        assert pdf_dir.is_absolute()

    def test_database_directory_created(self, temp_project_root: Path) -> None:
        """get_database_directory should create directory if needed."""
        config_path = temp_project_root / "config.yaml"
        new_db_dir = temp_project_root / "new_database"
        
        # Update config to use new directory that doesn't exist yet
        config_data = {
            "paths": {
                "pdf_directory": "data/pdfs",
                "database_directory": str(new_db_dir),
                "duckdb_file": "database/medical_rag.duckdb",
            },
            "embedding": {"model_name": "test", "dimensions": 768},
            "chunking": {"chunk_size_tokens": 750, "chunk_overlap_tokens": 150},
            "retrieval": {"top_k": 5},
            "bedrock": {
                "default_model_id": "test-model",
                "default_region": "us-east-1",
            },
            "duckdb": {"vss_extension": "vss", "index_distance_metric": "cosine"},
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        config = Config(config_path)
        db_dir = config.get_database_directory()
        
        assert db_dir.exists()
        assert db_dir == new_db_dir

    def test_duckdb_file_path_resolved(self, temp_project_root: Path) -> None:
        """DuckDB file path should be absolute."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        db_file = config.get_duckdb_file()
        assert db_file.is_absolute()
        assert str(db_file).endswith("medical_rag.duckdb")

    def test_accepted_answers_file_resolved(self, temp_project_root: Path) -> None:
        """Accepted-answers DuckDB path should come from config and be absolute."""
        config = Config(temp_project_root / "config.yaml")

        accepted_file = config.get_accepted_answers_file()
        assert accepted_file.is_absolute()
        assert accepted_file.name == "accepted_answers.duckdb"
        assert accepted_file != config.get_duckdb_file()

    def test_real_config_uses_top_10_and_separate_accepted_file(self) -> None:
        """The project config should retrieve 10 chunks and keep accepted answers in their own file."""
        config = Config.get_config()

        assert config.get_top_k() == 10
        assert config.get_accepted_answers_file() != config.get_duckdb_file()


class TestConfigurationValues:
    """Test retrieval of configuration values."""

    def test_get_embedding_dimensions(self, temp_project_root: Path) -> None:
        """Should return configured embedding dimensions."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        assert config.get_embedding_dimensions() == 768

    def test_embedding_dimensions_must_be_768(self, temp_project_root: Path) -> None:
        """Should raise error if embedding dimensions are not 768."""
        config_path = temp_project_root / "config.yaml"
        
        config_data = {
            "paths": {
                "pdf_directory": "data/pdfs",
                "database_directory": "database",
                "duckdb_file": "database/medical_rag.duckdb",
            },
            "embedding": {"model_name": "test", "dimensions": 512},  # Wrong dimension
            "chunking": {"chunk_size_tokens": 750, "chunk_overlap_tokens": 150},
            "retrieval": {"top_k": 5},
            "bedrock": {
                "default_model_id": "test-model",
                "default_region": "us-east-1",
            },
            "duckdb": {"vss_extension": "vss", "index_distance_metric": "cosine"},
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        config = Config(config_path)
        
        with pytest.raises(ConfigurationError, match="must be 768"):
            config.get_embedding_dimensions()

    def test_get_chunk_configuration(self, temp_project_root: Path) -> None:
        """Should return chunk size and overlap from configuration."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        assert config.get_chunk_size() == 750
        assert config.get_chunk_overlap() == 150

    def test_get_top_k(self, temp_project_root: Path) -> None:
        """Should return top_k retrieval value."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        assert config.get_top_k() == 5

    def test_get_bedrock_defaults(self, temp_project_root: Path) -> None:
        """Should return default Bedrock model and region."""
        config_path = temp_project_root / "config.yaml"
        config = Config(config_path)
        
        assert config.get_default_bedrock_model() == "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert config.get_default_region() == "us-east-1"

    def test_get_bedrock_regions(self, temp_project_root: Path) -> None:
        """Should return the regions offered in the UI."""
        config = Config(temp_project_root / "config.yaml")

        assert config.get_bedrock_regions() == ["us-east-1", "us-west-2"]

    def test_get_bedrock_regions_missing_raises(self, temp_project_root: Path) -> None:
        """Should fail clearly when no regions are configured."""
        config_path = temp_project_root / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        del data["bedrock"]["regions"]
        config_path.write_text(yaml.dump(data))

        config = Config(config_path)

        with pytest.raises(ConfigurationError, match="regions"):
            config.get_bedrock_regions()


class TestSingletonBehavior:
    """Test that configuration singleton works correctly."""

    def test_get_config_returns_singleton(self, temp_project_root: Path) -> None:
        """get_config should return the same instance on repeated calls."""
        config_path = temp_project_root / "config.yaml"
        
        config1 = Config.get_config(config_path)
        config2 = Config.get_config()
        
        assert config1 is config2

    def test_config_reset_clears_singleton(self, temp_project_root: Path) -> None:
        """Calling reset should clear the singleton."""
        config_path = temp_project_root / "config.yaml"
        
        config1 = Config.get_config(config_path)
        Config.reset()
        
        # Should create a new instance (this will fail without config.yaml in cwd)
        # But at least we can verify reset worked by checking _instance is None
        assert Config._instance is None

    def test_get_config_convenience_function(self, temp_project_root: Path) -> None:
        """Convenience function get_config should work."""
        config_path = temp_project_root / "config.yaml"
        config = get_config(config_path)
        
        assert config is not None
        assert config.get_embedding_dimensions() == 768


class TestMissingConfiguration:
    """Test error handling for missing configuration values."""

    def test_missing_pdf_directory(self, temp_project_root: Path) -> None:
        """Should raise error if PDF directory not in config."""
        config_path = temp_project_root / "config.yaml"
        
        config_data = {
            "paths": {
                "database_directory": "database",
                "duckdb_file": "database/medical_rag.duckdb",
            },
            "embedding": {"model_name": "test", "dimensions": 768},
            "chunking": {"chunk_size_tokens": 750, "chunk_overlap_tokens": 150},
            "retrieval": {"top_k": 5},
            "bedrock": {
                "default_model_id": "test-model",
                "default_region": "us-east-1",
            },
            "duckdb": {"vss_extension": "vss", "index_distance_metric": "cosine"},
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        config = Config(config_path)
        
        with pytest.raises(ConfigurationError, match="PDF directory not configured"):
            config.get_pdf_directory()

    def test_missing_embedding_model(self, temp_project_root: Path) -> None:
        """Should raise error if embedding model not in config."""
        config_path = temp_project_root / "config.yaml"
        
        config_data = {
            "paths": {
                "pdf_directory": "data/pdfs",
                "database_directory": "database",
                "duckdb_file": "database/medical_rag.duckdb",
            },
            "embedding": {"dimensions": 768},  # Missing model_name
            "chunking": {"chunk_size_tokens": 750, "chunk_overlap_tokens": 150},
            "retrieval": {"top_k": 5},
            "bedrock": {
                "default_model_id": "test-model",
                "default_region": "us-east-1",
            },
            "duckdb": {"vss_extension": "vss", "index_distance_metric": "cosine"},
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        config = Config(config_path)
        
        with pytest.raises(ConfigurationError, match="Embedding model not configured"):
            config.get_embedding_model()
