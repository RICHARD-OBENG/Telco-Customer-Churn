"""Feature package: engineering and preprocessing for model-ready data."""

from feature.pipeline import (
    FeaturePipelineError,
    FeaturePipelineResult,
    engineer_features,
    load_pipeline_artifacts,
    run_feature_pipeline,
)
from feature.preprocessing import (
    FeatureSchema,
    PreprocessingError,
    build_preprocessor,
    fit_preprocessor,
    transform_features,
)
from feature.selection import (
    FeatureSelectionError,
    SelectionReport,
    SelectorStrategy,
    anova_selection,
    chi2_selection,
    correlation_selection,
    get_top_features,
    model_based_importance,
    mutual_information_selection,
    select_features,
    vif_selection,
)

__all__ = [
    # Pipeline
    "FeaturePipelineError",
    "FeaturePipelineResult",
    "engineer_features",
    "load_pipeline_artifacts",
    "run_feature_pipeline",
    # Preprocessing
    "FeatureSchema",
    "PreprocessingError",
    "build_preprocessor",
    "fit_preprocessor",
    "transform_features",
    # Selection
    "FeatureSelectionError",
    "SelectionReport",
    "SelectorStrategy",
    "anova_selection",
    "chi2_selection",
    "correlation_selection",
    "get_top_features",
    "model_based_importance",
    "mutual_information_selection",
    "select_features",
    "vif_selection",
]
