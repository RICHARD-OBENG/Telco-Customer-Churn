"""End-to-end feature pipeline for the Telco Customer Churn model.

This module combines feature engineering, preprocessing, and feature selection
into a single reproducible workflow. It produces the final feature matrices
used by downstream model training, plus the fitted preprocessor and a feature
selection report so that the exact steps can be replayed for validation, test,
and inference data.

The pipeline is deliberately split into three phases to prevent data leakage:

1. **Feature engineering** — transforms raw input columns into new derived
   columns using only per-row information.
2. **Feature selection** — decides which raw and engineered columns to keep,
   fitted strictly on the training split.
3. **Preprocessing** — builds and fits a scikit-learn ``ColumnTransformer`` on
   the training split using the selected columns, then transforms all splits.

Artifacts are persisted to disk via joblib/CSV:

* Model artifacts (``.joblib``) go to ``models/``.
* Final feature matrices (``.csv``) go to ``data/features/``.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
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
import pandas as pd
from sklearn.compose import ColumnTransformer

from config.config_loader import PROJECT_ROOT, get_config, get_nested
from feature.preprocessing import FeatureSchema, build_preprocessor, transform_features
from feature.selection import SelectionReport, SelectorStrategy, select_features

logger = logging.getLogger(__name__)


class FeaturePipelineError(Exception):
    """Raised when the feature pipeline cannot be built or executed."""


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Outputs produced by the full feature pipeline.

    Attributes:
    -----------
    x_train:
        Final preprocessed training feature matrix.
    x_validation:
        Final preprocessed validation feature matrix.
    x_test:
        Final preprocessed test feature matrix.
    preprocessor:
        Fitted ``ColumnTransformer`` trained on the selected training features.
    selection_report:
        Report from the feature selection step (scores, dropped/retained lists).
    selected_features:
        List of column names that survived the selection step.
    """

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    preprocessor: ColumnTransformer
    selection_report: SelectionReport
    selected_features: list[str]


def _load_split_data(
    processed_dir: Path,
    splits: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load train/validation/test feature and target splits from CSV."""
    x_train = pd.read_csv(processed_dir / splits["x_train"])
    x_validation = pd.read_csv(processed_dir / splits["x_validation"])
    x_test = pd.read_csv(processed_dir / splits["x_test"])

    y_train_raw = pd.read_csv(processed_dir / splits["y_train"]).iloc[:, 0]
    y_validation_raw = pd.read_csv(
        processed_dir / splits["y_validation"]
    ).iloc[:, 0]
    y_test_raw = pd.read_csv(processed_dir / splits["y_test"]).iloc[:, 0]

    y_train = pd.Series(y_train_raw)
    y_validation = pd.Series(y_validation_raw)
    y_test = pd.Series(y_test_raw)

    return (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from the preprocessed Telco customer churn data.

    The input is expected to contain one-hot encoded categorical columns with
    the prefixes produced by ``build_preprocessor`` (e.g.
    ``categorical__contract_Month-to-month``) and numerical columns with the
    ``numerical__`` prefix. All new columns are derived row-wise so that the
    same function can be applied to any split without leakage.

    New features created:

    * ``total_services`` — count of active add-on services.
    * ``has_security_service`` — indicator for any security/backup/support.
    * ``has_streaming_service`` — indicator for any streaming service.
    * ``is_month_to_month`` — indicator for month-to-month contract.
    * ``is_long_term_contract`` — indicator for one/two-year contract.
    * ``is_electronic_payment`` — indicator for electronic check payment.
    * ``tenure_group`` — binned tenure category.
    """
    df = df.copy()

    service_yes_columns = [
        col
        for col in df.columns
        if any(
            col.startswith(f"categorical__{service}_Yes")
            for service in [
                "phoneservice",
                "multiplelines",
                "onlinesecurity",
                "onlinebackup",
                "deviceprotection",
                "techsupport",
                "streamingtv",
                "streamingmovies",
            ]
        )
    ]
    df["total_services"] = df[service_yes_columns].sum(axis=1)

    security_yes_columns = [
        col
        for col in df.columns
        if col.startswith(
            (
                "categorical__onlinesecurity_Yes",
                "categorical__onlinebackup_Yes",
                "categorical__deviceprotection_Yes",
                "categorical__techsupport_Yes",
            )
        )
    ]
    df["has_security_service"] = (
        df[security_yes_columns].eq(1).any(axis=1).astype(int)
    )

    streaming_columns = [
        col
        for col in df.columns
        if col.startswith(
            (
                "categorical__streamingtv_Yes",
                "categorical__streamingmovies_Yes",
            )
        )
    ]
    df["has_streaming_service"] = (
        df[streaming_columns].eq(1).any(axis=1).astype(int)
    )

    contract_month_to_month = "categorical__contract_Month-to-month"
    df["is_month_to_month"] = (
        df[contract_month_to_month].eq(1).astype(int)
        if contract_month_to_month in df.columns
        else 0
    )

    long_term_columns = [
        "categorical__contract_One year",
        "categorical__contract_Two year",
    ]
    available_long_term = [c for c in long_term_columns if c in df.columns]
    df["is_long_term_contract"] = (
        df[available_long_term].eq(1).any(axis=1).astype(int)
        if available_long_term
        else 0
    )

    payment_column = "categorical__paymentmethod_Electronic check"
    df["is_electronic_payment"] = (
        df[payment_column].eq(1).astype(int)
        if payment_column in df.columns
        else 0
    )

    tenure_column = "numerical__tenure"
    if tenure_column in df.columns:
        bins = [-1, 6, 12, 24, 48, float("inf")]
        labels = [
            "0-6_months",
            "7-12_months",
            "13-24_months",
            "25-48_months",
            "49+_months",
        ]
        df["tenure_group"] = pd.cut(
            df[tenure_column],
            bins=bins,
            labels=labels,
        )
    else:
        df["tenure_group"] = pd.NA

    return df


def _build_preprocessor_schema(
    selected_features: list[str],
    x_train_selected: pd.DataFrame,
) -> FeatureSchema:
    """Infer numerical/categorical split for the selected features."""
    schema = FeatureSchema.from_config()

    selected_numerical = [
        c for c in schema.numerical_features if c in selected_features
    ]
    selected_categorical = [
        c for c in schema.categorical_features if c in selected_features
    ]

    # Engineered features default to categorical unless explicitly numerical/binary.
    feature_cfg = get_config("feature_config")
    engineered_meta = get_nested(feature_cfg, "engineered_features", default=[])
    engineered_type_map: dict[str, str] = {}
    if isinstance(engineered_meta, list):
        for entry in engineered_meta:
            if isinstance(entry, dict):
                engineered_type_map[entry.get("name", "")] = entry.get("type", "")

    for feature in selected_features:
        if feature not in selected_numerical + selected_categorical:
            if engineered_type_map.get(feature) in ("numerical", "binary"):
                selected_numerical.append(feature)
            else:
                selected_categorical.append(feature)

    # Fallback: classify by dtype when config provides no guidance.
    if not selected_numerical and not selected_categorical:
        selected_numerical = x_train_selected.select_dtypes(
            include=["number"]
        ).columns.tolist()
        selected_categorical = [
            c for c in selected_features if c not in selected_numerical
        ]

    return FeatureSchema(
        numerical_features=tuple(selected_numerical),
        categorical_features=tuple(selected_categorical),
    )


def run_feature_pipeline(
    processed_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    features_dir: Path | str | None = None,
    model_family: str = "tree",
    selection_strategies: list[SelectorStrategy] | None = None,
    random_state: int = 42,
) -> FeaturePipelineResult:
    """Run the complete feature engineering + selection + preprocessing pipeline.

    Parameters
    ----------
    processed_dir:
        Directory containing ``X_train.csv``, ``X_validation.csv``,
        ``X_test.csv``, ``y_train.csv``, ``y_validation.csv``, ``y_test.csv``.
        Defaults to the value in ``config/data.yaml``.
    output_dir:
        Directory for model artifacts (``.joblib``). Defaults to ``models/``.
    features_dir:
        Directory for final feature matrix CSVs. Defaults to ``data/features/``.
    model_family:
        Preprocessor family passed to ``build_preprocessor`` (``"linear"``,
        ``"tree"``, or ``"catboost"``).
    selection_strategies:
        List of selection strategies to apply. ``None`` runs all available
        strategies.
    random_state:
        Seed for reproducible selection and model-based importance.

    Returns:
        :class:`FeaturePipelineResult` with final matrices and artifacts.
    """
    if model_family not in ("linear", "tree", "catboost"):
        raise FeaturePipelineError(f"Unsupported model_family: {model_family}")

    cfg = get_config("data")
    paths_cfg = get_nested(cfg, "paths", "processed", default={})

    resolved_processed_dir = (
        Path(processed_dir)
        if processed_dir
        else PROJECT_ROOT / get_nested(paths_cfg, "dir", default="data/processed")
    )
    resolved_output_dir = (
        Path(output_dir) if output_dir else PROJECT_ROOT / "models"
    )
    resolved_features_dir = (
        Path(features_dir)
        if features_dir
        else PROJECT_ROOT / "data" / "features"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_features_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "x_train": "X_train.csv",
        "x_validation": "X_validation.csv",
        "x_test": "X_test.csv",
        "y_train": "y_train.csv",
        "y_validation": "y_validation.csv",
        "y_test": "y_test.csv",
    }

    logger.info("Loading processed splits from %s", resolved_processed_dir)
    x_train, x_validation, x_test, y_train, _, _ = _load_split_data(
        resolved_processed_dir, splits
    )

    logger.info("Engineering features for all splits")
    x_train_fe = engineer_features(x_train)
    x_validation_fe = engineer_features(x_validation)
    x_test_fe = engineer_features(x_test)

    logger.info("Running feature selection on the training split")
    selection_report = select_features(
        x_train=x_train_fe,
        y_train=y_train,
        strategies=selection_strategies,
        random_state=random_state,
    )
    selected_features = selection_report.selected_features

    if not selected_features:
        raise FeaturePipelineError("No features were selected; cannot build pipeline")

    logger.info(
        "Selected %d features after dropping %s",
        len(selected_features),
        selection_report.dropped_features,
    )

    x_train_selected = x_train_fe[selected_features]
    x_validation_selected = x_validation_fe[selected_features]
    x_test_selected = x_test_fe[selected_features]

    preprocessor_schema = _build_preprocessor_schema(
        selected_features, x_train_selected
    )
    preprocessor = build_preprocessor(preprocessor_schema, model_family=model_family)

    logger.info("Fitting preprocessor on selected training features")
    preprocessor.fit(x_train_selected)

    x_train_final = transform_features(preprocessor, x_train_selected)
    x_validation_final = transform_features(preprocessor, x_validation_selected)
    x_test_final = transform_features(preprocessor, x_test_selected)

    logger.info(
        "Final feature matrices — train: %s, validation: %s, test: %s",
        x_train_final.shape,
        x_validation_final.shape,
        x_test_final.shape,
    )

    _persist_artifacts(
        output_dir=resolved_output_dir,
        features_dir=resolved_features_dir,
        preprocessor=preprocessor,
        selection_report=selection_report,
        x_train=x_train_final,
        x_validation=x_validation_final,
        x_test=x_test_final,
    )

    return FeaturePipelineResult(
        x_train=x_train_final,
        x_validation=x_validation_final,
        x_test=x_test_final,
        preprocessor=preprocessor,
        selection_report=selection_report,
        selected_features=selected_features,
    )


def _persist_artifacts(
    output_dir: Path,
    features_dir: Path,
    preprocessor: ColumnTransformer,
    selection_report: SelectionReport,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Save ``.joblib`` artifacts to ``models/`` and CSVs to ``data/features/``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
        joblib.dump(selection_report, output_dir / "selection_report.joblib")
        joblib.dump(
            selection_report.selected_features,
            output_dir / "selected_features.joblib",
        )

        x_train.to_csv(features_dir / "X_train_feature_engineered.csv", index=False)
        x_validation.to_csv(
            features_dir / "X_validation_feature_engineered.csv", index=False
        )
        x_test.to_csv(features_dir / "X_test_feature_engineered.csv", index=False)

        logger.info(
            "Saved model artifacts to %s and feature matrices to %s",
            output_dir,
            features_dir,
        )
    except OSError as exc:
        raise FeaturePipelineError(
            f"Failed to persist pipeline artifacts: {exc}"
        ) from exc


def load_pipeline_artifacts(
    output_dir: Path | str,
    features_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load persisted pipeline artifacts.

    ``features_dir`` defaults to ``PROJECT_ROOT / "data/features"``.
    """
    output_dir = Path(output_dir)
    resolved_features_dir = (
        Path(features_dir)
        if features_dir
        else PROJECT_ROOT / "data" / "features"
    )

    artifacts: dict[str, Any] = {
        "preprocessor": joblib.load(output_dir / "preprocessor.joblib"),
        "selection_report": joblib.load(output_dir / "selection_report.joblib"),
        "selected_features": joblib.load(output_dir / "selected_features.joblib"),
        "x_train": pd.read_csv(
            resolved_features_dir / "X_train_feature_engineered.csv"
        ),
        "x_validation": pd.read_csv(
            resolved_features_dir / "X_validation_feature_engineered.csv"
        ),
        "x_test": pd.read_csv(
            resolved_features_dir / "X_test_feature_engineered.csv"
        ),
    }
    return artifacts


def main() -> None:
    """CLI entry point to run the feature pipeline from the project root."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_feature_pipeline()
    logger.info(
        "Feature pipeline complete. Retained %d features: %s",
        len(result.selected_features),
        result.selected_features,
    )


if __name__ == "__main__":
    main()
