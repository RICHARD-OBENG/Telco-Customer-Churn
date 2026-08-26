"""Feature selection for the Telco Customer Churn pipeline.

This module decides which features should be retained for modeling. It supports
multiple complementary strategies:

* **Correlation analysis** — drop numerical features that are nearly perfectly
  correlated with another feature.
* **Multicollinearity (VIF)** — flag numerical features with high variance
  inflation factor.
* **Statistical tests** — univariate mutual information, chi-square for
  categorical features, and ANOVA F-test for numerical features.
* **Model-based selection** — feature importance from tree ensembles and
  coefficient magnitude from regularized linear models.

The recommended entry point is :func:`select_features`, which returns a list of
retained feature names and a structured report. All methods operate on the
*training split only* to avoid data leakage.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Literal

# Allow direct execution of this file as a script by adding the project source
# root (``src/``) to ``sys.path``. When the module is imported normally, the
# project root is expected to already be on ``PYTHONPATH``.
if __name__ == "__main__":
    _SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

from config.config_loader import get_config, get_nested
from feature.preprocessing import FeatureSchema, build_preprocessor, transform_features

logger = logging.getLogger(__name__)

SelectorStrategy = Literal[
    "correlation",
    "vif",
    "mutual_info",
    "chi2",
    "anova",
    "model_importance",
    "all",
]


class FeatureSelectionError(Exception):
    """Raised when feature selection input or configuration is invalid."""


@dataclass(frozen=True)
class SelectionReport:
    """Results produced by a feature selection run."""

    selected_features: list[str]
    dropped_features: list[str]
    correlation_drops: list[str] = field(default_factory=list)
    vif_drops: list[str] = field(default_factory=list)
    mutual_info_scores: dict[str, float] = field(default_factory=dict)
    chi2_scores: dict[str, float] = field(default_factory=dict)
    anova_scores: dict[str, float] = field(default_factory=dict)
    model_importances: dict[str, float] = field(default_factory=dict)
    method_thresholds: dict[str, Any] = field(default_factory=dict)


def _encode_target(y: pd.Series) -> np.ndarray:
    """Encode a binary target into {0, 1} integers."""
    encoded = y.copy()
    if encoded.dtype == object or encoded.dtype.name == "category":
        labels = sorted(encoded.dropna().unique())
        if len(labels) != 2:
            raise FeatureSelectionError(
                f"Target must be binary, got {len(labels)} unique values"
            )
        mapping = {labels[0]: 0, labels[1]: 1}
        encoded = encoded.map(mapping)
    return np.asarray(encoded.astype(int), dtype=int)


def correlation_selection(
    x_train: pd.DataFrame,
    numerical_features: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Drop one feature from each pair of highly correlated numerical features.

    Parameters
    ----------
    x_train:
        Training features.
    numerical_features:
        Numerical column names to consider.
    threshold:
        Absolute correlation above which one feature is dropped.

    Returns:
        List of numerical feature names to drop.
    """
    if threshold < 0 or threshold > 1:
        raise FeatureSelectionError("correlation threshold must be between 0 and 1")

    available = [col for col in numerical_features if col in x_train.columns]
    if len(available) < 2:
        return []

    corr_matrix = x_train[available].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop: set[str] = set()
    for column in upper_triangle.columns:
        highly_correlated = upper_triangle[column][upper_triangle[column] > threshold]
        for other_feature in highly_correlated.index:
            if column not in to_drop and other_feature not in to_drop:
                # Drop the feature that is less unique / more redundant.
                # Simple heuristic: drop the one with the higher mean absolute
                # correlation to all other numerical features.
                column_redundancy = corr_matrix[column].drop(column).mean()
                other_redundancy = corr_matrix[other_feature].drop(other_feature).mean()
                feature_to_drop = (
                    column if column_redundancy >= other_redundancy else other_feature
                )
                to_drop.add(feature_to_drop)
                logger.info(
                    "Dropping '%s' due to high correlation (%.3f) with '%s'",
                    feature_to_drop,
                    upper_triangle[column][other_feature],
                    other_feature,
                )

    return sorted(to_drop)


def vif_selection(
    x_train: pd.DataFrame,
    numerical_features: list[str],
    threshold: float = 10.0,
) -> list[str]:
    """Iteratively drop numerical features with high variance inflation factor.

    Parameters
    ----------
    x_train:
        Training features.
    numerical_features:
        Numerical column names to consider.
    threshold:
        VIF value above which a feature is dropped. A common rule of thumb is 5
        or 10.

    Returns:
        List of numerical feature names to drop.
    """
    if threshold <= 0:
        raise FeatureSelectionError("VIF threshold must be positive")

    available = [col for col in numerical_features if col in x_train.columns]
    if len(available) < 2:
        return []

    df = x_train[available].dropna().copy()
    if df.empty:
        return []

    to_drop: list[str] = []
    remaining = list(df.columns)

    while remaining:
        # Add constant term for VIF computation; it is dropped afterward.
        x_const = pd.concat(
            [pd.Series(1, index=df.index, name="const"), df[remaining]], axis=1
        )
        vif_values = [
            variance_inflation_factor(x_const.values, i + 1)
            for i in range(len(remaining))
        ]
        vif_series = pd.Series(vif_values, index=remaining)

        max_vif_feature = str(vif_series.idxmax())
        max_vif_value = float(vif_series.max())

        if max_vif_value <= threshold:
            break

        to_drop.append(max_vif_feature)
        remaining.remove(max_vif_feature)
        logger.info(
            "Dropping '%s' due to high VIF (%.2f)", max_vif_feature, max_vif_value
        )

    return to_drop


def mutual_information_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    numerical_features: list[str],
    categorical_features: list[str],
    k: int | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Score features using mutual information with the binary target.

    Numerical features are passed directly; categorical features are label
    encoded before scoring.

    Parameters
    ----------
    x_train:
        Training features.
    y_train:
        Training target.
    numerical_features:
        Numerical column names.
    categorical_features:
        Categorical column names.
    k:
        Number of top features to return. If None, all scores are returned.
    random_state:
        Seed for mutual information estimation.

    Returns:
        Mapping from feature name to mutual information score, sorted descending.
    """
    y = _encode_target(pd.Series(y_train))
    scores: dict[str, float] = {}

    for col in numerical_features:
        if col not in x_train.columns:
            continue
        x_col = np.asarray(
            x_train[col].fillna(x_train[col].median()), dtype=float
        ).reshape(-1, 1)
        score = mutual_info_classif(
            x_col, y, discrete_features=False, random_state=random_state
        )[0]
        scores[col] = float(score)

    for col in categorical_features:
        if col not in x_train.columns:
            continue
        le = LabelEncoder()
        x_col = le.fit_transform(x_train[col].astype(str))
        scores[col] = float(mutual_info_score(x_col, y))

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    if k is not None:
        sorted_scores = dict(list(sorted_scores.items())[:k])
    return sorted_scores


def chi2_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    categorical_features: list[str],
    k: int | None = None,
) -> dict[str, float]:
    """Score categorical features using the chi-square independence test.

    Parameters
    ----------
    x_train:
        Training features.
    y_train:
        Training target.
    categorical_features:
        Categorical column names.
    k:
        Number of top features to return. If None, all scores are returned.

    Returns:
        Mapping from feature name to chi-square statistic, sorted descending.
    """
    y = pd.Series(np.asarray(y_train).ravel())
    scores: dict[str, float] = {}

    for col in categorical_features:
        if col not in x_train.columns:
            continue
        contingency = pd.crosstab(x_train[col].astype(str), y.astype(str))
        if contingency.size == 0:
            continue
        chi2, _, _, _ = chi2_contingency(contingency)
        scores[col] = float(np.asarray(chi2).item())

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    if k is not None:
        sorted_scores = dict(islice(sorted_scores.items(), k))
    return sorted_scores


def anova_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    numerical_features: list[str],
    k: int | None = None,
) -> dict[str, float]:
    """Score numerical features using ANOVA F-test.

    Parameters
    ----------
    x_train:
        Training features.
    y_train:
        Training target.
    numerical_features:
        Numerical column names.
    k:
        Number of top features to return. If None, all scores are returned.

    Returns:
        Mapping from feature name to F-statistic, sorted descending.
    """
    y = _encode_target(pd.Series(y_train))
    available = [col for col in numerical_features if col in x_train.columns]
    if not available:
        return {}

    x_numeric = x_train[available].fillna(x_train[available].median())
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(x_numeric, y)

    scores = {
        feature: float(score)
        for feature, score in zip(available, np.asarray(selector.scores_), strict=True)
    }
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    if k is not None:
        sorted_scores = dict(islice(sorted_scores.items(), k))
    return sorted_scores


def model_based_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    numerical_features: list[str],
    categorical_features: list[str],
    model_family: Literal["linear", "tree", "xgboost"] = "tree",
    random_state: int = 42,
) -> dict[str, float]:
    """Compute feature importance using a fast model-based proxy.

    Parameters
    ----------
    x_train:
        Training features.
    y_train:
        Training target.
    numerical_features:
        Numerical column names.
    categorical_features:
        Categorical column names.
    model_family:
        Model type used to derive importance: ``"tree"`` uses a Random Forest,
        ``"xgboost"`` uses XGBoost, and ``"linear"`` uses logistic regression
        coefficient magnitudes.
    random_state:
        Seed for the estimator.

    Returns:
        Mapping from feature name to importance score, sorted descending.
    """
    from sklearn.ensemble import RandomForestClassifier

    y = _encode_target(pd.Series(y_train))

    # Determine which requested features actually exist in x_train. If the input
    # is already preprocessed (prefixed columns), use those directly; otherwise
    # fall back to the configured schema for raw data.
    available_cols = set(x_train.columns)
    available_numerical = [c for c in numerical_features if c in available_cols]
    available_categorical = [c for c in categorical_features if c in available_cols]

    if not available_numerical and not available_categorical:
        # Input may already be preprocessed; treat numeric columns as numerical
        # and any remaining object/category columns as categorical.
        available_numerical = x_train.select_dtypes(include="number").columns.tolist()
        available_categorical = x_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    schema = FeatureSchema(
        numerical_features=tuple(available_numerical),
        categorical_features=tuple(available_categorical),
    )

    if model_family == "linear":
        preprocessor = build_preprocessor(schema, model_family="linear")
    elif model_family in ("tree", "xgboost"):
        preprocessor = build_preprocessor(schema, model_family="tree")
    else:
        raise FeatureSelectionError(f"Unsupported model_family: {model_family}")

    x_processed = transform_features(preprocessor.fit(x_train), x_train)

    if model_family == "linear":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_processed, y)
        importances = np.abs(model.coef_[0])
    elif model_family == "tree":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_processed, y)
        importances = model.feature_importances_
    else:  # xgboost
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(x_processed, y)
        importances = model.feature_importances_

    feature_names = list(preprocessor.get_feature_names_out())
    scores = {
        name: float(importance)
        for name, importance in zip(feature_names, importances, strict=True)
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    strategies: list[SelectorStrategy] | None = None,
    config_name: str = "feature_config",
    random_state: int = 42,
) -> SelectionReport:
    """Run configured feature selection methods and return retained features.

    This is the main entry point for the module. It applies a sequence of
    filters and scoring methods to the training data only.

    Parameters
    ----------
    x_train:
        Training feature matrix (raw, pre-engineered features).
    y_train:
        Training target vector.
    numerical_features:
        Override list of numerical column names. If None, loaded from config.
    categorical_features:
        Override list of categorical column names. If None, loaded from config.
    strategies:
        List of selection methods to apply. Defaults to all methods.
    config_name:
        Name of the feature configuration file used to read engineered features.
    random_state:
        Seed for stochastic selectors.

    Returns:
        :class:`SelectionReport` containing selected/dropped features and scores.
    """
    if x_train.empty:
        raise FeatureSelectionError("Cannot select features from an empty training set")

    schema = FeatureSchema.from_config()
    numerical = list(numerical_features or schema.numerical_features)
    categorical = list(categorical_features or schema.categorical_features)

    # Always include engineered features present in the dataframe.
    feature_cfg = get_config(config_name)
    engineered = get_nested(
        feature_cfg, "engineered_features", default=[]
    ) or get_nested(feature_cfg, "engineered", default=[])
    for entry in engineered:
        if isinstance(entry, dict):
            feature_name = entry.get("name")
            feature_type = entry.get("type")
        else:
            feature_name = entry
            feature_type = None

        if (
            isinstance(feature_name, str)
            and feature_name in x_train.columns
            and feature_name not in numerical + categorical
        ):
            if feature_type in ("numerical", "binary"):
                numerical.append(feature_name)
            else:
                categorical.append(feature_name)

    all_candidate_features = [
        col for col in numerical + categorical if col in x_train.columns
    ]
    if not all_candidate_features:
        raise FeatureSelectionError("No candidate features found in training data")

    strategies = strategies or ["all"]
    if "all" in strategies:
        strategies = [
            "correlation",
            "vif",
            "mutual_info",
            "chi2",
            "anova",
            "model_importance",
        ]

    correlation_drops: list[str] = []
    vif_drops: list[str] = []
    mutual_info_scores: dict[str, float] = {}
    chi2_scores: dict[str, float] = {}
    anova_scores: dict[str, float] = {}
    model_importances: dict[str, float] = {}
    method_thresholds: dict[str, Any] = {}

    if "correlation" in strategies:
        threshold = 0.95
        method_thresholds["correlation"] = threshold
        correlation_drops = correlation_selection(x_train, numerical, threshold)

    if "vif" in strategies:
        threshold = 10.0
        method_thresholds["vif"] = threshold
        # Run VIF on the subset that survived correlation filtering.
        remaining_numerical = [c for c in numerical if c not in correlation_drops]
        vif_drops = vif_selection(x_train, remaining_numerical, threshold)

    if "mutual_info" in strategies:
        mutual_info_scores = mutual_information_selection(
            x_train,
            y_train,
            numerical,
            categorical,
            random_state=random_state,
        )

    if "chi2" in strategies:
        chi2_scores = chi2_selection(x_train, y_train, categorical)

    if "anova" in strategies:
        anova_scores = anova_selection(x_train, y_train, numerical)

    if "model_importance" in strategies:
        model_importances = model_based_importance(
            x_train,
            y_train,
            numerical,
            categorical,
            model_family="tree",
            random_state=random_state,
        )

    dropped = sorted(set(correlation_drops + vif_drops))
    selected = [c for c in all_candidate_features if c not in dropped]

    logger.info(
        "Feature selection complete: %d retained, %d dropped (%s)",
        len(selected),
        len(dropped),
        ", ".join(strategies),
    )

    return SelectionReport(
        selected_features=selected,
        dropped_features=dropped,
        correlation_drops=correlation_drops,
        vif_drops=vif_drops,
        mutual_info_scores=mutual_info_scores,
        chi2_scores=chi2_scores,
        anova_scores=anova_scores,
        model_importances=model_importances,
        method_thresholds=method_thresholds,
    )


def get_top_features(
    report: SelectionReport,
    method: Literal["mutual_info", "chi2", "anova", "model_importance"],
    top_n: int = 10,
) -> list[str]:
    """Return the top ``n`` feature names for a given scoring method."""
    score_map = {
        "mutual_info": report.mutual_info_scores,
        "chi2": report.chi2_scores,
        "anova": report.anova_scores,
        "model_importance": report.model_importances,
    }
    scores = score_map.get(method, {})
    if not scores:
        return []
    sorted_features = sorted(
        scores.items(), key=lambda item: item[1], reverse=True
    )
    return [feature for feature, _ in sorted_features][:top_n]
