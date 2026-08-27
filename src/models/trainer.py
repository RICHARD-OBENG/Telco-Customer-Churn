"""Training orchestrator for the Telco Customer Churn models.

This module trains the selected model(s) end-to-end: it dispatches to each
model's dedicated training routine (baseline, Logistic Regression, Random
Forest, XGBoost, CatBoost), which handles fitting on the prepared training
data and persisting its own model/metrics artifacts. On top of that, this
module builds a cross-model comparison, selects the best-performing model by
a chosen metric, and persists a training summary so downstream evaluation and
deployment code can identify the winning candidate.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Allow direct execution of this file as a script by adding the project source
# root (``src/``) to ``sys.path``. When imported normally, the project root is
# expected to already be on ``PYTHONPATH``.
if __name__ == "__main__":
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd

from config.config_loader import PROJECT_ROOT
from models.baseline import run_baseline_pipeline
from models.catboost import run_catboost_pipeline
from models.logistic_regression import run_logistic_regression_pipeline
from models.random_forest import run_random_forest_pipeline
from models.xgboost import run_xgboost_pipeline

logger = logging.getLogger(__name__)

ModelName = Literal[
    "baseline",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
]

# Candidate models compared for production selection; baseline is excluded by
# default since it only serves as the minimum bar, not a deployment candidate.
DEFAULT_MODEL_NAMES: tuple[ModelName, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
)

ModelResult = Any
"""Duck-typed model result: any of the ``*Result`` dataclasses exposing
``model``, ``model_name``, ``validation_metrics``, and ``test_metrics``."""


class TrainerError(Exception):
    """Raised when model training or selection cannot be completed."""


@dataclass(frozen=True, eq=False)
class TrainingSummary:
    """Outcome of training and comparing one or more models.

    Attributes:
    -----------
    results:
        Mapping from model name to its training result.
    comparison_table:
        Per-model validation/test metrics, one row per model.
    best_model_name:
        Name of the model with the highest ``selection_metric`` on
        ``selection_split``.
    selection_metric:
        Metric used to rank models (e.g. ``"f1"``).
    selection_split:
        Split used to rank models (``"validation"`` or ``"test"``).
    """

    results: dict[str, Any]
    comparison_table: pd.DataFrame
    best_model_name: str
    selection_metric: str
    selection_split: str


def _train_baseline(
    features_dir: Path | str | None,
    output_dir: Path | str | None,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
) -> Any:
    """Adapt the baseline pipeline to the trainer's uniform call signature."""
    strategy = "logistic_regression"
    if hyperparameters:
        strategy = hyperparameters.get("strategy", strategy)
    return run_baseline_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        strategy=strategy,
        random_state=random_state,
    )


def _train_logistic_regression(
    features_dir: Path | str | None,
    output_dir: Path | str | None,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
) -> Any:
    return run_logistic_regression_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


def _train_random_forest(
    features_dir: Path | str | None,
    output_dir: Path | str | None,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
) -> Any:
    return run_random_forest_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


def _train_xgboost(
    features_dir: Path | str | None,
    output_dir: Path | str | None,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
) -> Any:
    return run_xgboost_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


def _train_catboost(
    features_dir: Path | str | None,
    output_dir: Path | str | None,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
) -> Any:
    return run_catboost_pipeline(
        features_dir=features_dir,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )


_MODEL_TRAINERS: dict[str, Any] = {
    "baseline": _train_baseline,
    "logistic_regression": _train_logistic_regression,
    "random_forest": _train_random_forest,
    "xgboost": _train_xgboost,
    "catboost": _train_catboost,
}


def train_model(
    model_name: ModelName,
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Any:
    """Train a single named model and persist its artifacts.

    Parameters
    ----------
    model_name:
        One of ``"baseline"``, ``"logistic_regression"``, ``"random_forest"``,
        ``"xgboost"``, ``"catboost"``.
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
        Defaults to ``data/features/``.
    output_dir:
        Directory where the fitted model and metrics are saved.
        Defaults to ``models/``.
    hyperparameters:
        Optional parameter dictionary forwarded to the model's builder.
    random_state:
        Seed for reproducibility.

    Returns:
        The model-specific result dataclass (e.g. ``CatBoostResult``).
    """
    trainer_fn = _MODEL_TRAINERS.get(model_name)
    if trainer_fn is None:
        raise TrainerError(
            f"Unknown model_name '{model_name}'. "
            f"Supported models: {sorted(_MODEL_TRAINERS)}"
        )

    logger.info("Training model: %s", model_name)
    return trainer_fn(features_dir, output_dir, hyperparameters, random_state)


def train_models(
    model_names: list[ModelName] | None = None,
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters_map: dict[str, dict[str, Any]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train several models, skipping and logging any that fail.

    Parameters
    ----------
    model_names:
        Models to train. Defaults to :data:`DEFAULT_MODEL_NAMES`.
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
    output_dir:
        Directory where fitted models and metrics are saved.
    hyperparameters_map:
        Optional mapping from model name to its hyperparameter overrides.
    random_state:
        Seed for reproducibility.

    Returns:
        Mapping from model name to its training result. Models that raised
        an error during training are omitted.

    Raises:
        TrainerError: If every requested model fails to train.
    """
    names = list(model_names or DEFAULT_MODEL_NAMES)
    hyperparameters_map = hyperparameters_map or {}

    results: dict[str, Any] = {}
    for name in names:
        try:
            results[name] = train_model(
                model_name=name,
                features_dir=features_dir,
                output_dir=output_dir,
                hyperparameters=hyperparameters_map.get(name),
                random_state=random_state,
            )
        except Exception:
            logger.exception("Training failed for model '%s'; skipping", name)

    if not results:
        raise TrainerError(f"All requested models failed to train: {names}")

    return results


def build_comparison_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a per-model metrics comparison table.

    Returns:
        DataFrame with one row per model and columns for validation/test
        accuracy, precision, recall, F1, and ROC-AUC.
    """
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        row: dict[str, Any] = {"model": name}
        for split, metrics in (
            ("validation", result.validation_metrics),
            ("test", result.test_metrics),
        ):
            for metric_name, value in metrics.items():
                row[f"{split}_{metric_name}"] = value
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def select_best_model(
    results: dict[str, Any],
    metric: str = "f1",
    split: str = "test",
) -> tuple[str, Any]:
    """Select the model with the highest metric value on the given split.

    Parameters
    ----------
    results:
        Mapping from model name to its training result.
    metric:
        Metric key to rank on (e.g. ``"f1"``, ``"roc_auc"``).
    split:
        Which split's metrics to use: ``"validation"`` or ``"test"``.

    Returns:
        Tuple of ``(best_model_name, best_result)``.
    """
    if not results:
        raise TrainerError("Cannot select a best model from empty results")
    if split not in ("validation", "test"):
        raise TrainerError(f"split must be 'validation' or 'test', got {split!r}")

    def _metric_value(result: Any) -> float:
        metrics = (
            result.validation_metrics if split == "validation" else result.test_metrics
        )
        value = metrics.get(metric)
        if value is None:
            raise TrainerError(f"Metric '{metric}' not found for split '{split}'")
        return float(value)

    best_name = max(results, key=lambda name: _metric_value(results[name]))
    return best_name, results[best_name]


def run_training_pipeline(
    model_names: list[ModelName] | None = None,
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters_map: dict[str, dict[str, Any]] | None = None,
    selection_metric: str = "f1",
    selection_split: str = "test",
    random_state: int = 42,
) -> TrainingSummary:
    """Train candidate models, compare them, and select the best one.

    Parameters
    ----------
    model_names:
        Models to train. Defaults to :data:`DEFAULT_MODEL_NAMES`.
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
        Defaults to ``data/features/``.
    output_dir:
        Directory where fitted models, metrics, and the training summary are
        saved. Defaults to ``models/``.
    hyperparameters_map:
        Optional mapping from model name to its hyperparameter overrides.
    selection_metric:
        Metric used to select the best model (e.g. ``"f1"``, ``"roc_auc"``).
    selection_split:
        Split used to select the best model (``"validation"`` or ``"test"``).
    random_state:
        Seed for reproducibility.

    Returns:
        :class:`TrainingSummary` with all results, the comparison table, and
        the selected best model.
    """
    resolved_output_dir = (
        Path(output_dir) if output_dir else PROJECT_ROOT / "models"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    results = train_models(
        model_names=model_names,
        features_dir=features_dir,
        output_dir=resolved_output_dir,
        hyperparameters_map=hyperparameters_map,
        random_state=random_state,
    )

    comparison_table = build_comparison_table(results)
    best_name, _ = select_best_model(
        results, metric=selection_metric, split=selection_split
    )

    summary = TrainingSummary(
        results=results,
        comparison_table=comparison_table,
        best_model_name=best_name,
        selection_metric=selection_metric,
        selection_split=selection_split,
    )

    _persist_training_summary(resolved_output_dir, summary)
    logger.info(
        "Best model by %s (%s): %s",
        selection_metric,
        selection_split,
        best_name,
    )
    return summary


def _persist_training_summary(
    output_dir: Path,
    summary: TrainingSummary,
) -> None:
    """Save the comparison table and best-model pointer to disk."""
    try:
        summary.comparison_table.to_csv(output_dir / "training_summary.csv")

        payload = {
            "selection_metric": summary.selection_metric,
            "selection_split": summary.selection_split,
            "best_model_name": summary.best_model_name,
            "trained_models": list(summary.results.keys()),
            "comparison": summary.comparison_table.reset_index().to_dict(
                orient="records"
            ),
        }
        with (output_dir / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2)

        logger.info("Saved training summary to %s", output_dir)
    except OSError as exc:
        raise TrainerError(f"Failed to persist training summary: {exc}") from exc


def main() -> None:
    """CLI entry point to train and compare all candidate models."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    summary = run_training_pipeline()
    logger.info("Comparison table:\n%s", summary.comparison_table)
    logger.info("Selected best model: %s", summary.best_model_name)


if __name__ == "__main__":
    main()
