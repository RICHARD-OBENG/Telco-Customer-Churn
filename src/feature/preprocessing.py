"""Convert raw features into model-ready features.

Builds reproducible scikit-learn preprocessing pipelines that handle missing
values, categorical encoding, and numerical scaling. A single pipeline
definition is shared across models to avoid duplicated preprocessing logic;
callers select a ``model_family`` to get the right encoding/scaling strategy.

The returned ``ColumnTransformer`` must be fit only on training data, then
reused as-is (via joblib persistence) for validation, test, and inference
data to prevent training/inference skew.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from config.config_loader import get_config, get_nested

logger = logging.getLogger(__name__)

ModelFamily = Literal["linear", "tree", "catboost"]


class PreprocessingError(Exception):
    """Raised when preprocessing configuration or input data is invalid."""


@dataclass(frozen=True)
class FeatureSchema:
    """Column groups used to build the preprocessing pipeline."""

    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    @classmethod
    def from_config(cls) -> FeatureSchema:
        """Load numerical/categorical column names from ``config/data.yaml``."""
        cfg = get_config("data")
        numerical = get_nested(cfg, "features", "numerical", default=[])
        categorical = get_nested(cfg, "features", "categorical", default=[])
        if not numerical and not categorical:
            raise PreprocessingError(
                "No numerical or categorical features declared in config/data.yaml"
            )
        return cls(
            numerical_features=tuple(numerical),
            categorical_features=tuple(categorical),
        )


def build_preprocessor(
    schema: FeatureSchema,
    model_family: ModelFamily = "tree",
) -> ColumnTransformer:
    """Build an unfitted preprocessing pipeline for the given model family.

    Parameters
    ----------
    schema:
        Numerical/categorical column groups to transform.
    model_family:
        ``"linear"`` scales numerical features and one-hot encodes
        categoricals (needed by e.g. Logistic Regression). ``"tree"``
        skips scaling and ordinal-encodes categoricals, since sklearn/XGBoost/
        LightGBM tree models are scale-invariant but need numeric input.
        ``"catboost"`` skips scaling and leaves categoricals as raw strings
        (missing values filled with a placeholder), since CatBoost handles
        categorical columns natively.

    Returns:
        ColumnTransformer: Unfitted transformer. Must be fit only on training data.
    """
    if model_family not in ("linear", "tree", "catboost"):
        raise PreprocessingError(f"Unsupported model_family: {model_family}")

    numerical_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_family == "linear":
        numerical_steps.append(("scaler", StandardScaler()))
    numerical_pipeline = Pipeline(steps=numerical_steps)

    if model_family == "linear":
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )
    elif model_family == "tree":
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", ordinal_encoder),
            ]
        )
    else:  # catboost: keep raw categories, only fill missing values
        missing_value_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        categorical_pipeline = Pipeline(steps=[("imputer", missing_value_imputer)])

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if schema.numerical_features:
        transformers.append(
            ("numerical", numerical_pipeline, list(schema.numerical_features))
        )
    if schema.categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, list(schema.categorical_features))
        )

    if not transformers:
        raise PreprocessingError(
            "No features available to build a preprocessing pipeline"
        )

    logger.info(
        "Built '%s' preprocessing pipeline: %d numerical, %d categorical features",
        model_family,
        len(schema.numerical_features),
        len(schema.categorical_features),
    )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit a preprocessor in place, strictly on training data.

    Never call this with validation, test, or inference data — doing so
    would leak information from those sets into the fitted transformer.
    """
    if x_train.empty:
        raise PreprocessingError("Cannot fit preprocessor on an empty training set")
    preprocessor.fit(x_train)
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    x: pd.DataFrame,
) -> pd.DataFrame:
    """Apply an already-fitted preprocessor to produce model-ready features."""
    check_is_fitted(preprocessor)
    transformed = preprocessor.transform(x)
    feature_names = preprocessor.get_feature_names_out()
    result = pd.DataFrame(transformed, columns=feature_names, index=x.index)
    return _coerce_numeric_columns(result)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric dtypes lost when numerical/categorical outputs mix.

    ``ColumnTransformer`` stacks all transformer outputs into a single
    array; if any transformer emits raw strings (the "catboost" family),
    numpy upcasts the whole array to ``object``. This converts back any
    column whose values are fully numeric, leaving genuine text untouched.
    """
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().all():
            df[column] = converted
    return df
