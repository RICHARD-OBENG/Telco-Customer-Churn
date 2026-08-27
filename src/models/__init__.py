"""Model package: training, evaluation, and registry for churn classifiers."""

from models.baseline import (
    BaselineModelError,
    BaselineResult,
    evaluate_baseline_model,
    load_baseline_model,
    run_baseline_pipeline,
    train_baseline_model,
)
from models.catboost import (
    CatBoostError,
    CatBoostResult,
    build_catboost,
    evaluate_catboost,
    load_catboost_model,
    run_catboost_pipeline,
    train_catboost,
)
from models.logistic_regression import (
    LogisticRegressionError,
    LogisticRegressionResult,
    build_logistic_regression,
    evaluate_logistic_regression,
    load_logistic_regression_model,
    run_logistic_regression_pipeline,
    train_logistic_regression,
)
from models.random_forest import (
    RandomForestError,
    RandomForestResult,
    build_random_forest,
    evaluate_random_forest,
    load_random_forest_model,
    run_random_forest_pipeline,
    train_random_forest,
)
from models.xgboost import (
    XGBoostError,
    XGBoostResult,
    build_xgboost,
    evaluate_xgboost,
    load_xgboost_model,
    run_xgboost_pipeline,
    train_xgboost,
)

__all__ = [
    # Baseline
    "BaselineModelError",
    "BaselineResult",
    "evaluate_baseline_model",
    "load_baseline_model",
    "run_baseline_pipeline",
    "train_baseline_model",
    # Logistic Regression
    "LogisticRegressionError",
    "LogisticRegressionResult",
    "build_logistic_regression",
    "evaluate_logistic_regression",
    "load_logistic_regression_model",
    "run_logistic_regression_pipeline",
    "train_logistic_regression",
    # Random Forest
    "RandomForestError",
    "RandomForestResult",
    "build_random_forest",
    "evaluate_random_forest",
    "load_random_forest_model",
    "run_random_forest_pipeline",
    "train_random_forest",
    # XGBoost
    "XGBoostError",
    "XGBoostResult",
    "build_xgboost",
    "evaluate_xgboost",
    "load_xgboost_model",
    "run_xgboost_pipeline",
    "train_xgboost",
    # CatBoost
    "CatBoostError",
    "CatBoostResult",
    "build_catboost",
    "evaluate_catboost",
    "load_catboost_model",
    "run_catboost_pipeline",
    "train_catboost",
]
