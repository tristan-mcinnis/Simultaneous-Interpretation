"""Configuration management for architecture testing."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config_dir() -> Path:
    """Return the path to the configs directory."""
    return Path(__file__).parent


def list_configs() -> list[Path]:
    """List all available configuration files."""
    return list(get_config_dir().glob("arch_*.yaml"))
