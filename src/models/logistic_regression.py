"""Logistic Regression classifier for the Telco Customer Churn project.

This module defines, trains, and evaluates a Logistic Regression classifier.
Logistic Regression is a natural baseline for churn prediction because it is
interpretable, fast to train, and provides well-calibrated probability
estimates. The module reuses the feature matrices produced by the feature
pipeline and persists the fitted model and metrics.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config_loader import PROJECT_ROOT, get_config, get_nested

logger = logging.getLogger(__name__)


class LogisticRegressionError(Exception):
    """Raised when the Logistic Regression model cannot be trained or evaluated."""


@dataclass(frozen=True)
class LogisticRegressionResult:
    """Results from training and evaluating the Logistic Regression classifier.

    Attributes:
    -----------
    model:
        The fitted Logistic Regression estimator.
    model_name:
        Human-readable name of the model.
    validation_metrics:
        Metrics computed on the validation split.
    test_metrics:
        Metrics computed on the test split.
    """

    model: LogisticRegression
    model_name: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


def _encode_target(y: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert a binary target into {0, 1} integers."""
    arr = np.asarray(y).ravel()
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        unique = sorted(set(arr[~pd.isna(arr)]))
        if len(unique) != 2:
            raise LogisticRegressionError(
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


def build_logistic_regression(
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LogisticRegression:
    """Build a Logistic Regression classifier from config or defaults.

    Parameters
    ----------
    hyperparameters:
        Optional dictionary of scikit-learn ``LogisticRegression`` parameters.
        If ``None``, values are read from ``config/model.yaml``.
    random_state:
        Seed for reproducibility.

    Returns:
        Unfitted ``LogisticRegression`` estimator.
    """
    if hyperparameters is None:
        cfg = get_config("model")
        hyperparameters = get_nested(
            cfg, "hyperparameters", "logistic_regression", default={}
        )

    params: dict[str, Any] = dict(hyperparameters or {})
    params.setdefault("max_iter", 1000)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", random_state)

    # n_jobs is deprecated in scikit-learn >= 1.8 for solvers that do not use it.
    params.pop("n_jobs", None)

    return LogisticRegression(**params)


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a Logistic Regression classifier on the training data.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training target vector.
    hyperparameters:
        Optional parameter dictionary. See :func:`build_logistic_regression`.
    random_state:
        Seed for reproducibility.

    Returns:
        Fitted ``LogisticRegression`` estimator.
    """
    if x_train.empty:
        raise LogisticRegressionError(
            "Cannot train logistic regression on an empty training set"
        )

    y = _encode_target(y_train)
    model = build_logistic_regression(
        hyperparameters=hyperparameters,
        random_state=random_state,
    )
    model.fit(x_train, y)
    logger.info("Trained Logistic Regression classifier")
    return model


def evaluate_logistic_regression(
    model: LogisticRegression,
    x_validation: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> LogisticRegressionResult:
    """Evaluate a fitted Logistic Regression classifier.

    Parameters
    ----------
    model:
        Fitted estimator.
    x_validation:
        Validation feature matrix.
    y_validation:
        Validation target vector.
    x_test:
        Test feature matrix.
    y_test:
        Test target vector.

    Returns:
        :class:`LogisticRegressionResult` with metrics for both splits.
    """
    y_val = _encode_target(y_validation)
    y_tst = _encode_target(y_test)

    val_pred = model.predict(x_validation)
    val_proba = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)

    test_pred = model.predict(x_test)
    test_proba = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)

    val_metrics = _compute_metrics(y_val, val_pred, val_proba)
    test_metrics = _compute_metrics(y_tst, test_pred, test_proba)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    _log_classification_report("Validation", y_val, val_pred)
    _log_classification_report("Test", y_tst, test_pred)

    return LogisticRegressionResult(
        model=model,
        model_name="logistic_regression",
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
    )


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


def run_logistic_regression_pipeline(
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> LogisticRegressionResult:
    """Load features, train, evaluate, and save a Logistic Regression model.

    Parameters
    ----------
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
        Defaults to ``data/features/``.
    output_dir:
        Directory where the fitted model and metrics are saved.
        Defaults to ``models/``.
    hyperparameters:
        Optional parameter dictionary passed to ``LogisticRegression``.
    random_state:
        Seed for reproducibility.

    Returns:
        :class:`LogisticRegressionResult` with the fitted model and metrics.
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

    model = train_logistic_regression(
        x_train=x_train,
        y_train=y_train,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    result = evaluate_logistic_regression(
        model=model,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )

    _persist_logistic_regression_artifacts(output_dir, result)
    return result


def _persist_logistic_regression_artifacts(
    output_dir: Path,
    result: LogisticRegressionResult,
) -> None:
    """Save the fitted Logistic Regression model and its metrics to disk."""
    try:
        joblib.dump(result.model, output_dir / "logistic_regression_model.joblib")

        metrics_path = output_dir / "logistic_regression_metrics.json"
        metrics = {
            "model_name": result.model_name,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        logger.info("Saved Logistic Regression artifacts to %s", output_dir)
    except OSError as exc:
        raise LogisticRegressionError(
            f"Failed to persist Logistic Regression artifacts: {exc}"
        ) from exc


def load_logistic_regression_model(
    output_dir: Path | str,
) -> LogisticRegression:
    """Load a previously saved Logistic Regression model from disk."""
    output_dir = Path(output_dir)
    return joblib.load(output_dir / "logistic_regression_model.joblib")


def main() -> None:
    """CLI entry point to train and evaluate the Logistic Regression model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_logistic_regression_pipeline()
    logger.info(
        "Logistic Regression — validation F1: %.4f, test F1: %.4f",
        result.validation_metrics["f1"],
        result.test_metrics["f1"],
    )


if __name__ == "__main__":
    # Allow direct execution from the project root.
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    main()
