"""XGBoost classifier for the Telco Customer Churn project.

This module defines, trains, and evaluates an XGBoost gradient-boosted tree
classifier. XGBoost is provided as a stronger, higher-capacity comparison
point against the Logistic Regression and Random Forest models. It reuses
the feature matrices produced by the feature pipeline and persists the
fitted model, metrics, and feature importances.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Direct execution puts this file's own directory on ``sys.path[0]``, which
# shadows the real third-party ``xgboost`` package (same module name). Drop it
# before importing xgboost, then add ``src/`` for the ``config`` import below.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config.config_loader import PROJECT_ROOT, get_config, get_nested

logger = logging.getLogger(__name__)


class XGBoostError(Exception):
    """Raised when the XGBoost model cannot be trained or evaluated."""


@dataclass(frozen=True)
class XGBoostResult:
    """Results from training and evaluating the XGBoost classifier.

    Attributes:
    -----------
    model:
        The fitted XGBoost estimator.
    model_name:
        Human-readable name of the model.
    validation_metrics:
        Metrics computed on the validation split.
    test_metrics:
        Metrics computed on the test split.
    feature_importances:
        Mapping from feature name to gain-based importance, sorted descending.
    """

    model: XGBClassifier
    model_name: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)


def _encode_target(y: pd.Series | np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convert a binary target into {0, 1} integers."""
    arr = np.asarray(y).ravel()
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        unique = sorted(set(arr[~pd.isna(arr)]))
        if len(unique) != 2:
            raise XGBoostError(
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


def _resolve_scale_pos_weight(
    value: Any,
    y_train: pd.Series | np.ndarray,
) -> float | None:
    """Resolve the ``scale_pos_weight`` hyperparameter to a numeric ratio.

    ``"balanced"`` computes ``negative_count / positive_count`` from the
    training labels; numeric values are passed through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str) and value == "balanced":
        y = _encode_target(y_train)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        if positive_count == 0:
            raise XGBoostError("Cannot compute scale_pos_weight: no positive samples")
        return negative_count / positive_count
    return float(value)


def build_xgboost(
    hyperparameters: dict[str, Any] | None = None,
    y_train: pd.Series | np.ndarray | None = None,
    random_state: int = 42,
) -> XGBClassifier:
    """Build an XGBoost classifier from config or defaults.

    Parameters
    ----------
    hyperparameters:
        Optional dictionary of ``XGBClassifier`` parameters. If ``None``,
        values are read from ``config/model.yaml``.
    y_train:
        Training labels, required to resolve a ``"balanced"``
        ``scale_pos_weight`` into a numeric ratio.
    random_state:
        Seed for reproducibility.

    Returns:
        Unfitted ``XGBClassifier`` estimator.
    """
    if hyperparameters is None:
        cfg = get_config("model")
        hyperparameters = get_nested(cfg, "hyperparameters", "xgboost", default={})

    params: dict[str, Any] = dict(hyperparameters or {})
    params.setdefault("n_estimators", 500)
    params.setdefault("max_depth", 6)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.8)
    params.setdefault("eval_metric", "logloss")
    params.setdefault("random_state", random_state)
    params.setdefault("n_jobs", -1)

    # use_label_encoder was removed in recent xgboost releases.
    params.pop("use_label_encoder", None)

    scale_pos_weight = params.pop("scale_pos_weight", None)
    if scale_pos_weight is not None:
        if y_train is None:
            raise XGBoostError(
                "y_train is required to resolve scale_pos_weight='balanced'"
            )
        resolved_y_train = y_train
        params["scale_pos_weight"] = _resolve_scale_pos_weight(
            scale_pos_weight, resolved_y_train
        )

    return XGBClassifier(**params)


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> XGBClassifier:
    """Train an XGBoost classifier on the training data.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training target vector.
    hyperparameters:
        Optional parameter dictionary. See :func:`build_xgboost`.
    random_state:
        Seed for reproducibility.

    Returns:
        Fitted ``XGBClassifier`` estimator.
    """
    if x_train.empty:
        raise XGBoostError("Cannot train XGBoost on an empty training set")

    y = _encode_target(y_train)
    model = build_xgboost(
        hyperparameters=hyperparameters,
        y_train=y_train,
        random_state=random_state,
    )
    model.fit(x_train, y)
    logger.info("Trained XGBoost classifier")
    return model


def _compute_feature_importances(
    model: XGBClassifier,
    feature_names: list[str],
) -> dict[str, float]:
    """Return gain-based feature importances sorted descending."""
    importances = {
        name: float(importance)
        for name, importance in zip(
            feature_names, model.feature_importances_, strict=True
        )
    }
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def evaluate_xgboost(
    model: XGBClassifier,
    x_validation: pd.DataFrame,
    y_validation: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> XGBoostResult:
    """Evaluate a fitted XGBoost classifier.

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
        :class:`XGBoostResult` with metrics and feature importances.
    """
    y_val = _encode_target(y_validation)
    y_tst = _encode_target(y_test)

    val_pred = model.predict(x_validation)
    val_proba = np.asarray(model.predict_proba(x_validation)[:, 1], dtype=float)

    test_pred = model.predict(x_test)
    test_proba = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)

    val_metrics = _compute_metrics(y_val, val_pred, val_proba)
    test_metrics = _compute_metrics(y_tst, test_pred, test_proba)
    feature_importances = _compute_feature_importances(
        model, list(x_validation.columns)
    )

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    _log_classification_report("Validation", y_val, val_pred)
    _log_classification_report("Test", y_tst, test_pred)

    return XGBoostResult(
        model=model,
        model_name="xgboost",
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        feature_importances=feature_importances,
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


def run_xgboost_pipeline(
    features_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> XGBoostResult:
    """Load features, train, evaluate, and save an XGBoost model.

    Parameters
    ----------
    features_dir:
        Directory containing ``X_*_feature_engineered.csv`` files.
        Defaults to ``data/features/``.
    output_dir:
        Directory where the fitted model and metrics are saved.
        Defaults to ``models/``.
    hyperparameters:
        Optional parameter dictionary passed to ``XGBClassifier``.
    random_state:
        Seed for reproducibility.

    Returns:
        :class:`XGBoostResult` with the fitted model and metrics.
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

    model = train_xgboost(
        x_train=x_train,
        y_train=y_train,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    result = evaluate_xgboost(
        model=model,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )

    _persist_xgboost_artifacts(output_dir, result)
    return result


def _persist_xgboost_artifacts(
    output_dir: Path,
    result: XGBoostResult,
) -> None:
    """Save the fitted XGBoost model and its metrics to disk."""
    try:
        joblib.dump(result.model, output_dir / "xgboost_model.joblib")

        metrics_path = output_dir / "xgboost_metrics.json"
        metrics = {
            "model_name": result.model_name,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "feature_importances": result.feature_importances,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        logger.info("Saved XGBoost artifacts to %s", output_dir)
    except OSError as exc:
        raise XGBoostError(
            f"Failed to persist XGBoost artifacts: {exc}"
        ) from exc


def load_xgboost_model(
    output_dir: Path | str,
) -> XGBClassifier:
    """Load a previously saved XGBoost model from disk."""
    output_dir = Path(output_dir)
    model = joblib.load(output_dir / "xgboost_model.joblib")
    if not isinstance(model, XGBClassifier):
        raise XGBoostError("Loaded artifact is not an XGBClassifier instance")
    return model


def main() -> None:
    """CLI entry point to train and evaluate the XGBoost model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_xgboost_pipeline()
    logger.info(
        "XGBoost — validation F1: %.4f, test F1: %.4f",
        result.validation_metrics["f1"],
        result.test_metrics["f1"],
    )


if __name__ == "__main__":
    main()
