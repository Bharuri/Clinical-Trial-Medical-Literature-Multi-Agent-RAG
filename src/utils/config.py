"""
Configuration loader for the Clinical Trial & Medical Literature Multi-Agent RAG.
Loads settings from root config.yaml and validates paths.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class Config:
    """Loads and manages application configuration from root config.yaml."""

    _instance: Optional["Config"] = None
    _config: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Optional path to config.yaml. If not provided, searches
                         for it relative to the project root.

        Raises:
            ConfigurationError: If config file not found or invalid.
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self._config_path = Path(config_path).resolve()
        
        if not self._config_path.exists():
            raise ConfigurationError(f"config.yaml not found at {self._config_path}")
        
        try:
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load config.yaml: {e}")
        
        # Validate and resolve paths
        self._resolve_paths()

    @staticmethod
    def _find_config_file() -> Path:
        """
        Find config.yaml by searching up from current working directory.
        
        Search order:
        1. Current working directory
        2. Parent directories up to filesystem root
        3. If not found, default to project root relative to this module
        """
        # Start from current working directory
        current = Path.cwd()
        
        # Search up the directory tree
        for attempt in range(10):
            config_path = current / "config.yaml"
            if config_path.exists():
                return config_path
            
            parent = current.parent
            if parent == current:
                # Reached filesystem root
                break
            current = parent
        
        # Fallback: project root (assuming this file is in src/utils/)
        module_dir = Path(__file__).parent
        project_root = module_dir.parent.parent
        return project_root / "config.yaml"

    def _resolve_paths(self) -> None:
        """
        Resolve all paths in configuration to absolute paths.
        
        Relative paths are resolved from the project root (where config.yaml lives).
        """
        if "paths" not in self._config:
            self._config["paths"] = {}
        
        project_root = self._config_path.parent
        
        for key in ["pdf_directory", "database_directory", "duckdb_file", "accepted_answers_duckdb_file"]:
            if key in self._config["paths"]:
                path_str = self._config["paths"][key]
                path_obj = Path(path_str)
                
                # If relative, resolve from project root
                if not path_obj.is_absolute():
                    path_obj = project_root / path_obj
                
                self._config["paths"][key] = str(path_obj.resolve())

    def get_pdf_directory(self) -> Path:
        """Get the PDF input directory. Does not validate existence."""
        path_str = self._config.get("paths", {}).get("pdf_directory")
        if not path_str:
            raise ConfigurationError("PDF directory not configured in config.yaml")
        return Path(path_str)

    def get_database_directory(self) -> Path:
        """Get the database directory. Creates it if it does not exist."""
        path_str = self._config.get("paths", {}).get("database_directory")
        if not path_str:
            raise ConfigurationError("Database directory not configured in config.yaml")
        
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_duckdb_file(self) -> Path:
        """Get the DuckDB file path."""
        path_str = self._config.get("paths", {}).get("duckdb_file")
        if not path_str:
            raise ConfigurationError("DuckDB file path not configured in config.yaml")
        return Path(path_str)

    def get_accepted_answers_file(self) -> Path:
        """Get the DuckDB file that stores manually accepted answers."""
        path_str = self._config.get("paths", {}).get("accepted_answers_duckdb_file")
        if not path_str:
            raise ConfigurationError("Accepted answers DuckDB path not configured in config.yaml")
        return Path(path_str)

    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        model = self._config.get("embedding", {}).get("model_name")
        if not model:
            raise ConfigurationError("Embedding model not configured in config.yaml")
        return model

    def get_embedding_dimensions(self) -> int:
        """Get the expected embedding dimensions (should be 768)."""
        dims = self._config.get("embedding", {}).get("dimensions")
        if dims is None:
            raise ConfigurationError("Embedding dimensions not configured in config.yaml")
        if dims != 768:
            raise ConfigurationError(f"Embedding dimensions must be 768, got {dims}")
        return dims

    def get_chunk_size(self) -> int:
        """Get the target chunk size in tokens."""
        size = self._config.get("chunking", {}).get("chunk_size_tokens")
        if not size:
            raise ConfigurationError("Chunk size not configured in config.yaml")
        return size

    def get_chunk_overlap(self) -> int:
        """Get the target chunk overlap in tokens."""
        overlap = self._config.get("chunking", {}).get("chunk_overlap_tokens")
        if overlap is None:
            raise ConfigurationError("Chunk overlap not configured in config.yaml")
        return overlap

    def get_top_k(self) -> int:
        """Get the number of top chunks to retrieve."""
        k = self._config.get("retrieval", {}).get("top_k", 5)
        return k

    def get_default_bedrock_model(self) -> str:
        """Get the default Bedrock model ID."""
        model = self._config.get("bedrock", {}).get("default_model_id")
        if not model:
            raise ConfigurationError("Default Bedrock model not configured in config.yaml")
        return model

    def get_default_region(self) -> str:
        """Get the default AWS region."""
        region = self._config.get("bedrock", {}).get("default_region")
        if not region:
            raise ConfigurationError("Default region not configured in config.yaml")
        return region

    def get_bedrock_regions(self) -> List[str]:
        """Get the AWS regions offered in the UI."""
        regions = self._config.get("bedrock", {}).get("regions")
        if not regions:
            raise ConfigurationError("Bedrock regions not configured in config.yaml")
        return list(regions)

    def get_vss_extension(self) -> str:
        """Get the DuckDB VSS extension name."""
        ext = self._config.get("duckdb", {}).get("vss_extension", "vss")
        return ext

    def get_index_distance_metric(self) -> str:
        """Get the distance metric for the HNSW index."""
        metric = self._config.get("duckdb", {}).get("index_distance_metric", "cosine")
        return metric

    @classmethod
    def get_config(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Get the singleton configuration instance.
        
        Args:
            config_path: Optional path to config.yaml for testing.
                        If provided, a new instance is created.
        
        Returns:
            Config instance.
        """
        if config_path is not None:
            # Explicit path replaces the singleton so later calls return the same instance
            cls._instance = cls(config_path)
            return cls._instance
        
        if cls._instance is None:
            cls._instance = cls()
        
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = None
        cls._config_path = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Convenience function to get the configuration instance.
    
    Args:
        config_path: Optional path to config.yaml for testing.
    
    Returns:
        Config instance.
    """
    return Config.get_config(config_path)
