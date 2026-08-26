"""Project-wide, strongly typed application settings.

Centralizes paths, reproducibility, data schema, model, API, and logging
settings. Values are resolved with the following precedence:
environment variables > YAML files under ``config/`` > hard-coded defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config_loader import (
    PROJECT_ROOT,
    get_config,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_str,
    get_nested,
)


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations used across the project. All paths are absolute."""

    root: Path = PROJECT_ROOT
    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_data_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    feature_engineered_dir: Path = (
        PROJECT_ROOT / "data" / "processed" / "feature_engineered"
    )
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    reports_dir: Path = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DataSettings:
    """Dataset schema and splitting configuration."""

    target_column: str = "Churn"
    id_column: str = "customerID"
    positive_label: str = "Yes"
    negative_label: str = "No"
    test_size: float = 0.15
    validation_size: float = 0.15


@dataclass(frozen=True)
class ModelSettings:
    """Model selection, reproducibility, and artifact locations.

    ``selected_model`` is configurable rather than hard-coded so the
    production model can change without rewriting application code.
    """

    selected_model: str = "catboost"
    primary_metric: str = "auc_roc"
    prediction_threshold: float = 0.5
    random_seed: int = 42
    model_file: Path = PROJECT_ROOT / "models" / "telco_churn_model.joblib"
    preprocessor_file: Path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    metadata_file: Path = PROJECT_ROOT / "models" / "model_metadata.joblib"


@dataclass(frozen=True)
class APISettings:
    """FastAPI service metadata and binding configuration."""

    title: str = "Telco Customer Churn Prediction API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass(frozen=True)
class LoggingSettings:
    """Application logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file: Path = PROJECT_ROOT / "logs" / "app.log"


@dataclass(frozen=True)
class Settings:
    """Aggregated, strongly typed application settings."""

    environment: str = "development"
    paths: PathSettings = field(default_factory=PathSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    api: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def _build_data_settings() -> DataSettings:
    """Resolve data settings from env vars, then config/data.yaml, then defaults."""
    cfg = get_config("data")
    target_name = get_nested(cfg, "columns", "target", "name", default="Churn")
    id_name = get_nested(cfg, "columns", "id", "name", default="customerID")
    target_keys = ("columns", "target")
    positive_label = get_nested(cfg, *target_keys, "positive_label", default="Yes")
    negative_label = get_nested(cfg, *target_keys, "negative_label", default="No")
    test_size = get_nested(cfg, "split", "ratios", "test", default=0.15)
    validation_size = get_nested(cfg, "split", "ratios", "validation", default=0.15)
    return DataSettings(
        target_column=get_env_str("TARGET_COLUMN", target_name),
        id_column=get_env_str("ID_COLUMN", id_name),
        positive_label=positive_label,
        negative_label=negative_label,
        test_size=get_env_float("TEST_SIZE", test_size),
        validation_size=get_env_float("VALIDATION_SIZE", validation_size),
    )


def _build_model_settings() -> ModelSettings:
    """Resolve model settings from env vars, then config/model.yaml, then defaults."""
    cfg = get_config("model")
    models_dir = PathSettings().models_dir
    primary_metric = get_nested(cfg, "evaluation", "primary_metric", default="auc_roc")
    default_model = cfg.get("default_model", "catboost")
    return ModelSettings(
        selected_model=get_env_str("SELECTED_MODEL", default_model),
        primary_metric=get_env_str("PRIMARY_METRIC", primary_metric),
        prediction_threshold=get_env_float("PREDICTION_THRESHOLD", 0.5),
        random_seed=get_env_int("RANDOM_SEED", 42),
        model_file=models_dir / "telco_churn_model.joblib",
        preprocessor_file=models_dir / "preprocessor.joblib",
        metadata_file=models_dir / "model_metadata.joblib",
    )


def _build_api_settings() -> APISettings:
    """Resolve API settings from env vars, then config/api.yaml, then defaults."""
    cfg = get_config("api")
    default_title = "Telco Customer Churn Prediction API"
    host = get_nested(cfg, "server", "host", default="0.0.0.0")
    port = get_nested(cfg, "server", "port", default=8000)
    reload = get_nested(cfg, "server", "reload", default=False)
    return APISettings(
        title=get_nested(cfg, "api", "name", default=default_title),
        version=get_nested(cfg, "api", "version", default="1.0.0"),
        host=get_env_str("API_HOST", host),
        port=get_env_int("API_PORT", port),
        reload=get_env_bool("API_RELOAD", reload),
    )


def _build_logging_settings() -> LoggingSettings:
    """Resolve logging settings from env vars, with sensible defaults."""
    logs_dir = PathSettings().logs_dir
    return LoggingSettings(
        level=get_env_str("LOG_LEVEL", "INFO"),
        log_to_file=get_env_bool("LOG_TO_FILE", True),
        log_file=logs_dir / "app.log",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the application settings singleton.

    Cached via ``lru_cache`` so the model is not loaded from disk or YAML
    re-parsed on every call. Use ``get_settings.cache_clear()`` in tests
    that need to reload settings after mutating the environment.
    """
    return Settings(
        environment=get_env_str("ENVIRONMENT", "development"),
        paths=PathSettings(),
        data=_build_data_settings(),
        model=_build_model_settings(),
        api=_build_api_settings(),
        logging=_build_logging_settings(),
    )
