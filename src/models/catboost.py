"""CatBoost classifier for the Telco Customer Churn project.

This module defines, trains, and evaluates a CatBoost gradient-boosted tree
classifier. CatBoost is particularly well-suited to this dataset because it
handles categorical features natively (no manual one-hot/ordinal encoding
required), which matters here since most Telco churn features are
categorical. Given comparable or superior performance to Logistic Regression,
Random Forest, and XGBoost, CatBoost is the selected production candidate.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Direct execution puts this file's own directory on ``sys.path[0]``, which
# shadows the real third-party ``catboost`` package (same module name). Drop
# it before importing catboost, then add ``src/`` for the ``config`` import.
if __name__ == "__main__":
    _SCRIPT_DIR = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == _SCRIPT_DIR:
        sys.path.pop(0)
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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


class CatBoostError(Exception):
    """Raised when the CatBoost model cannot be trained or evaluated."""


@dataclass(frozen=True)
class CatBoostResult:
    """Results from training and evaluating the CatBoost classifier.

    Attributes:
    -----------
    model:
        The fitted CatBoost estimator.
    model_name:
        Human-readable name of the model.
    validation_metrics:
        Metrics computed on the validation split.
    test_metrics:
        Metrics computed on the test split.
    feature_importances:
        Mapping from feature name to importance, sorted descending.
    categorical_features:
        Column names treated as categorical during training.
    """

    model: CatBoostClassifier
    model_name: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    categorical_features: list[str] = field(default_factory=list)


def _encode_target(y: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert a binary target into {0, 1} integers."""
    arr = np.asarray(y).ravel()
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        unique = sorted(set(arr[~pd.isna(arr)]))
        if len(unique) != 2:
            raise CatBoostError(
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


def _detect_categorical_features(x: pd.DataFrame) -> list[str]:
    """Detect columns CatBoost should treat as categorical (non-numeric dtypes)."""
    return x.select_dtypes(include=["object", "category"]).columns.tolist()


def build_catboost(
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> CatBoostClassifier:
    """Build a CatBoost classifier from config or defaults.

    Parameters
    ----------
    hyperparameters:
        Optional dictionary of ``CatBoostClassifier`` parameters. If ``None``,
        values are read from ``config/model.yaml``.
    random_state:
        Seed for reproducibility.

    Returns:
        Unfitted ``CatBoostClassifier`` estimator.
    """
    if hyperparameters is None:
        cfg = get_config("model")
        hyperparameters = get_nested(cfg, "hyperparameters", "catboost", default={})

    params: dict[str, Any] = dict(hyperparameters or {})
    params.setdefault("iterations", 500)
    params.setdefault("depth", 6)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("l2_leaf_reg", 3.0)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "AUC")
    params.setdefault("auto_class_weights", "Balanced")
    params.setdefault("random_seed", random_state)
    params.setdefault("verbose", False)

    return CatBoostClassifier(**params)


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    hyperparameters: dict[str, Any] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = 42,
) -> tuple[CatBoostClassifier, list[str]]:
    """Train a CatBoost classifier on the training data.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training target vector.
    hyperparameters:
        Optional parameter dictionary. See :func:`build_catboost`.
    categorical_features:
        Column names to treat as categorical. If ``None``, non-numeric
        columns in ``x_train`` are detected automatically.
    random_state:
        Seed for reproducibility.

    Returns:
        Tuple of the fitted ``CatBoostClassifier`` and the categorical
        feature names used during training.
    """
    if x_train.empty:
        raise CatBoostError("Cannot train CatBoost on an empty training set")

    y = _encode_target(y_train)
    cat_features = (
        categorical_features
        if categorical_features is not None
        else _detect_categorical_features(x_train)
    )

    model = build_catboost(hyperparameters=hyperparameters, random_state=random_state)
    model.fit(x_train, y, cat_features=cat_features or None)
    logger.info(
        "Trained CatBoost classifier with %d categorical feature(s): %s",
        len(cat_features),
        cat_features,
    )
    return model, cat_features


def _compute_feature_importances(
    model: CatBoostClassifier,
    feature_names: list[str],
) -> dict[str, float]:
    """Return CatBoost feature importances sorted descending."""
    raw_importances: list[float] = list(
        np.asarray(model.get_feature_importance(), dtype=float).ravel()
    )
    importances = {
        name: float(importance)
        for name, importance in zip(feature_names, raw_importances, strict=True)
    }
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def evaluate_catboost(
    model: CatBoostClassifier,
    x_validation: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    categorical_features: list[str] | None = None,
) -> CatBoostResult:
    """Evaluate a fitted CatBoost classifier.

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
    categorical_features:
        Column names treated as categorical during training, for reporting.

    Returns:
        :class:`CatBoostResult` with metrics and feature importances.
    """
    y_val = _encode_target(y_validation)
    y_tst = _encode_target(y_test)

    val_pred = model.predict(x_validation)
    val_proba = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)

    test_pred = model.predict(x_test)
    test_proba = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)

    val_metrics = _compute_metrics(y_val, np.asarray(val_pred).ravel(), val_proba)
    test_metrics = _compute_metrics(y_tst, np.asarray(test_pred).ravel(), test_proba)
    feature_importances = _compute_feature_importances(
        model, list(x_validation.columns)
    )

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    _log_classification_report("Validation", y_val, np.asarray(val_pred).ravel())
    _log_classification_report("Test", y_tst, np.asarray(test_pred).ravel())

    return CatBoostResult(
        model=model,
        model_name="catboost",
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        feature_importances=feature_importances,
        categorical_features=categorical_features or [],
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


def run_catboost_pipeline(
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = 42,
) -> CatBoostResult:
    """Load features, train, evaluate, and save a CatBoost model.

    Parameters
    ----------
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
        Defaults to ``data/features/``.
    output_dir:
        Directory where the fitted model and metrics are saved.
        Defaults to ``models/``.
    hyperparameters:
        Optional parameter dictionary passed to ``CatBoostClassifier``.
    categorical_features:
        Column names to treat as categorical. If ``None``, detected
        automatically from non-numeric dtypes in the training data.
    random_state:
        Seed for reproducibility.

    Returns:
        :class:`CatBoostResult` with the fitted model and metrics.
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

    model, resolved_cat_features = train_catboost(
        x_train=x_train,
        y_train=y_train,
        hyperparameters=hyperparameters,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    result = evaluate_catboost(
        model=model,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        categorical_features=resolved_cat_features,
    )

    _persist_catboost_artifacts(output_dir, result)
    return result


def _persist_catboost_artifacts(
    output_dir: Path,
    result: CatBoostResult,
) -> None:
    """Save the fitted CatBoost model and its metrics to disk."""
    try:
        joblib.dump(result.model, output_dir / "catboost_model.joblib")

        metrics_path = output_dir / "catboost_metrics.json"
        metrics = {
            "model_name": result.model_name,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "feature_importances": result.feature_importances,
            "categorical_features": result.categorical_features,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        logger.info("Saved CatBoost artifacts to %s", output_dir)
    except OSError as exc:
        raise CatBoostError(
            f"Failed to persist CatBoost artifacts: {exc}"
        ) from exc


def load_catboost_model(
    output_dir: Path | str,
) -> CatBoostClassifier:
    """Load a previously saved CatBoost model from disk."""
    output_dir = Path(output_dir)
    model = joblib.load(output_dir / "catboost_model.joblib")
    if not isinstance(model, CatBoostClassifier):
        raise CatBoostError("Loaded artifact is not a CatBoostClassifier instance")
    return model


def main() -> None:
    """CLI entry point to train and evaluate the CatBoost model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_catboost_pipeline()
    logger.info(
        "CatBoost — validation F1: %.4f, test F1: %.4f",
        result.validation_metrics["f1"],
        result.test_metrics["f1"],
    )


if __name__ == "__main__":
    main()
