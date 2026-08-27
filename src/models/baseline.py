"""Baseline model for the Telco Customer Churn project.

This module trains a simple but sensible baseline classifier and reports
performance metrics on the validation and test sets. The baseline establishes
the minimum performance that more sophisticated models should beat.

The default baseline is a Logistic Regression classifier with balanced class
weights and a StandardScaler for the numerical features. It uses the same
preprocessed feature matrices produced by the feature pipeline so that
comparisons with later models are fair.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow direct execution of this file as a script by adding the project source
# root (``src/``) to ``sys.path``. When imported normally, the project root is
# expected to already be on ``PYTHONPATH``.
if __name__ == "__main__":
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)


class BaselineModelError(Exception):
    """Raised when the baseline model cannot be trained or evaluated."""


@dataclass(frozen=True)
class BaselineResult:
    """Results from training and evaluating a baseline model.

    Attributes:
    -----------
    model:
        The fitted baseline estimator.
    model_name:
        Human-readable name of the baseline strategy.
    validation_metrics:
        Metrics computed on the validation split.
    test_metrics:
        Metrics computed on the test split.
    """

    model: Any
    model_name: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


def _encode_target(y: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert a binary target into {0, 1} integers."""
    arr = np.asarray(y).ravel()
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        unique = sorted(set(arr[~pd.isna(arr)]))
        if len(unique) != 2:
            raise BaselineModelError(
                f"Target must be binary, got {len(unique)} unique values"
            )
        mapping = {unique[0]: 0, unique[1]: 1}
        arr = np.array([mapping.get(v, np.nan) for v in arr])
    return arr.astype(int)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    """Return a dictionary of classification metrics for binary problems."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def train_baseline_model(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    strategy: str = "logistic_regression",
    random_state: int = 42,
) -> Any:
    """Train a baseline classifier on the training data.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training target vector.
    strategy:
        Baseline strategy. ``"logistic_regression"`` fits a balanced logistic
        regression. ``"most_frequent"`` predicts the majority class.
    random_state:
        Seed for reproducibility.

    Returns:
        Fitted baseline estimator.
    """
    y = _encode_target(y_train)

    if x_train.empty:
        raise BaselineModelError("Cannot train baseline on an empty training set")

    if strategy == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif strategy == "most_frequent":
        model = DummyClassifier(strategy="most_frequent")
    else:
        raise BaselineModelError(f"Unknown baseline strategy: {strategy}")

    model.fit(x_train, y)
    logger.info("Trained baseline model: %s", strategy)
    return model


def evaluate_baseline_model(
    model: Any,
    x_validation: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    model_name: str = "baseline",
) -> BaselineResult:
    """Evaluate a fitted baseline model on validation and test sets.

    Parameters
    ----------
    model:
        Fitted baseline estimator.
    x_validation:
        Validation feature matrix.
    y_validation:
        Validation target vector.
    x_test:
        Test feature matrix.
    y_test:
        Test target vector.
    model_name:
        Name used for reporting.

    Returns:
        :class:`BaselineResult` containing metrics for both splits.
    """
    y_val = _encode_target(y_validation)
    y_tst = _encode_target(y_test)

    val_pred = model.predict(x_validation)
    val_proba = _predict_proba_positive(model, x_validation)

    test_pred = model.predict(x_test)
    test_proba = _predict_proba_positive(model, x_test)

    val_metrics = _compute_metrics(y_val, val_pred, val_proba)
    test_metrics = _compute_metrics(y_tst, test_pred, test_proba)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    _log_classification_report("Validation", y_val, val_pred)
    _log_classification_report("Test", y_tst, test_pred)

    return BaselineResult(
        model=model,
        model_name=model_name,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def _predict_proba_positive(
    model: Any,
    x: pd.DataFrame,
) -> np.ndarray | None:
    """Return predicted probabilities for the positive class if available."""
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is None:
        return None
    proba = np.asarray(predict_proba(x), dtype=float)
    return proba[:, 1]


def _log_classification_report(
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Log a detailed classification report at INFO level."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
    logger.info("Classification report — %s:\n%s", split_name, report)


def run_baseline_pipeline(
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    strategy: str = "logistic_regression",
    random_state: int = 42,
) -> BaselineResult:
    """Load feature matrices, train a baseline, evaluate it, and save artifacts.

    Parameters
    ----------
    features_dir:
        Directory containing ``X_train_feature_engineered.csv`` and the
        corresponding target CSVs. Defaults to ``data/features/``.
    output_dir:
        Directory where the fitted baseline model and metrics will be saved.
        Defaults to ``models/``.
    strategy:
        Baseline strategy (``"logistic_regression"`` or ``"most_frequent"``).
    random_state:
        Seed for reproducibility.

    Returns:
        :class:`BaselineResult` with the fitted model and metrics.
    """
    features_dir = (
        Path(features_dir)
        if features_dir
        else PROJECT_ROOT / "data" / "features"
    )
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = pd.read_csv(features_dir / "X_train_feature_engineered.csv")
    x_validation = pd.read_csv(
        features_dir / "X_validation_feature_engineered.csv"
    )
    x_test = pd.read_csv(features_dir / "X_test_feature_engineered.csv")

    y_train = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "y_train.csv"
    ).iloc[:, 0]
    y_validation = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "y_validation.csv"
    ).iloc[:, 0]
    y_test = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "y_test.csv"
    ).iloc[:, 0]

    model = train_baseline_model(
        x_train=x_train,
        y_train=y_train,
        strategy=strategy,
        random_state=random_state,
    )

    result = evaluate_baseline_model(
        model=model,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        model_name=strategy,
    )

    _persist_baseline_artifacts(output_dir, result)
    return result


def _persist_baseline_artifacts(
    output_dir: Path,
    result: BaselineResult,
) -> None:
    """Save the baseline model and its metrics to disk."""
    try:
        joblib.dump(result.model, output_dir / "baseline_model.joblib")

        metrics_path = output_dir / "baseline_metrics.json"
        metrics = {
            "model_name": result.model_name,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        logger.info("Saved baseline artifacts to %s", output_dir)
    except OSError as exc:
        raise BaselineModelError(
            f"Failed to persist baseline artifacts: {exc}"
        ) from exc


def load_baseline_model(
    output_dir: Path | str,
) -> Any:
    """Load a previously saved baseline model from disk."""
    output_dir = Path(output_dir)
    return joblib.load(output_dir / "baseline_model.joblib")


def main() -> None:
    """CLI entry point to train and evaluate the baseline model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_baseline_pipeline()
    logger.info(
        "Baseline '%s' — validation F1: %.4f, test F1: %.4f",
        result.model_name,
        result.validation_metrics["f1"],
        result.test_metrics["f1"],
    )


if __name__ == "__main__":
    main()
