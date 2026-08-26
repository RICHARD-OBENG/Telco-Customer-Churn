"""Configuration loading: YAML/JSON files and environment variables.

Owns raw configuration access (file discovery, parsing, env var lookups) so
``setting.py`` and other modules can build typed objects without duplicating
file I/O or parsing logic.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# src/config/config_loader.py -> src/config -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"

_SUPPORTED_EXTENSIONS = (".yaml", ".yml", ".json")


class ConfigurationError(Exception):
    """Raised when a configuration file is missing, unreadable, or invalid."""


def _resolve_config_path(name: str, directory: Path) -> Path:
    """Find a config file by base or full name, trying supported extensions."""
    candidate = Path(name)
    if candidate.suffix in _SUPPORTED_EXTENSIONS:
        path = directory / candidate
        if path.is_file():
            return path
        raise ConfigurationError(f"Configuration file not found: {path}")

    for extension in _SUPPORTED_EXTENSIONS:
        path = directory / f"{name}{extension}"
        if path.is_file():
            return path

    raise ConfigurationError(
        f"No configuration file named '{name}' found in {directory} "
        f"(tried extensions: {', '.join(_SUPPORTED_EXTENSIONS)})"
    )


def _parse_file(path: Path) -> dict[str, Any]:
    """Parse a YAML or JSON file into a dictionary."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix == ".json":
                content = json.load(handle)
            else:
                content = yaml.safe_load(handle)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Failed to parse config file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Failed to read config file {path}: {exc}") from exc

    if content is None:
        return {}
    if not isinstance(content, dict):
        actual_type = type(content).__name__
        raise ConfigurationError(
            f"Config file {path} must contain a top-level mapping, got {actual_type}"
        )
    return content


@cache
def get_config(name: str, directory: str | None = None) -> dict[str, Any]:
    """Load and cache a YAML/JSON configuration file by name.

    Parameters
    ----------
    name:
        Base file name (e.g. ``"data"``) or full name with extension
        (e.g. ``"feature_config.json"``).
    directory:
        Optional override directory; defaults to the project's ``config/``.

    Returns:
        dict[str, Any]: Parsed configuration contents, or ``{}`` for an empty file.
    """
    config_dir = Path(directory) if directory else CONFIG_DIR
    path = _resolve_config_path(name, config_dir)
    return _parse_file(path)


def clear_config_cache() -> None:
    """Clear cached configuration files. Useful for tests and config reloads."""
    get_config.cache_clear()


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely walk a nested dictionary using a sequence of keys."""
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def get_env_str(key: str, default: str) -> str:
    """Return a string environment variable, falling back to ``default``."""
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """Return an integer environment variable, falling back to ``default``."""
    value = os.getenv(key)
    return int(value) if value is not None else int(default)


def get_env_float(key: str, default: float) -> float:
    """Return a float environment variable, falling back to ``default``."""
    value = os.getenv(key)
    return float(value) if value is not None else float(default)


def get_env_bool(key: str, default: bool) -> bool:
    """Return a boolean environment variable, falling back to ``default``."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
