# Telco Customer Churn Prediction — Project Overview

---

## Executive Summary

This project delivers a production-grade machine learning system that predicts the likelihood of telecom customer churn. The solution ingests raw customer demographic, service, and account data, transforms it through a validated preprocessing pipeline, trains and compares multiple classification algorithms, and exposes the best-performing model through a containerized FastAPI inference service. The goal is to enable proactive retention strategies that reduce churn-related revenue loss.

---

## Business Problem

Customer churn is one of the most significant cost drivers in the telecom industry. Acquiring a new customer is typically more expensive than retaining an existing one, and silent attrition erodes recurring revenue. Without an early warning system, retention teams react only after cancellation, missing the window for effective intervention.

The business problem is therefore: **how can we accurately identify customers at risk of leaving before they churn, and provide interpretable risk scores that retention teams can act on?**

---

## Business Objectives

- Predict individual churn probability for each active customer.
- Rank customers by churn risk to prioritize retention outreach.
- Provide transparent, explainable predictions for regulatory and business trust.
- Deliver predictions through a low-latency API suitable for batch scoring and real-time integration.
- Maintain reproducibility, testability, and observability across the ML lifecycle.

---

## Project Scope

### In Scope

- Data ingestion, extraction, and validation pipeline.
- Exploratory data analysis and feature engineering.
- Training, evaluation, and selection of classification models.
- Hyperparameter tuning and class imbalance handling.
- Model explainability using SHAP values.
- Experiment tracking with MLflow.
- FastAPI service for model inference.
- Docker containerization for portable deployment.
- Unit tests, linting, type checking, and CI-ready tooling.

### Out of Scope

- Automated model retraining loop in production.
- Live A/B testing platform.
- Customer-facing dashboard or CRM integration.
- Real-time streaming data ingestion.

These may be added in future phases.

---

## Success Criteria

| Criterion | Target |
|---|---|
| Model AUC-ROC | ≥ 0.85 on hold-out test set |
| Recall for churn class | ≥ 0.60 to capture most at-risk customers |
| Precision for churn class | ≥ 0.65 to control campaign cost |
| API latency (p95) | < 200 ms per prediction |
| Prediction throughput | ≥ 100 requests/second per container instance |
| Pipeline reproducibility | Full reproduction from `poetry.lock` and committed code |
| Code coverage | ≥ 80% unit test coverage |

---

## Machine Learning Workflow

The ML workflow follows a structured, iterative pipeline:

1. **Data Ingestion** — download raw Telco churn dataset from configured source.
2. **Data Extraction & Validation** — extract archives, validate schema, check for missing values and data drift.
3. **Exploratory Data Analysis** — profile distributions, correlations, class balance, and feature quality.
4. **Feature Engineering** — encode categoricals, scale numerics, handle missing values, create interaction and derived features.
5. **Modeling** — train and compare Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
6. **Evaluation** — assess accuracy, precision, recall, F1-score, AUC-ROC, and calibration.
7. **Explainability** — generate SHAP values to understand drivers of individual predictions.
8. **Selection & Registration** — register the best model and its metrics in MLflow.
9. **Serving** — deploy the model via FastAPI with OpenAPI documentation.
10. **Monitoring** — log predictions and track performance degradation over time.

---

## Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core development language |
| Packaging | Poetry | Reproducible dependency and environment management |
| Data Processing | Pandas, NumPy, SciPy | Tabular data manipulation and computation |
| Visualization | Matplotlib, Seaborn, Plotly, Missingno | EDA, reporting, and profiling |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost | Model training and evaluation |
| Imbalance Handling | Imbalanced-learn | Resampling and cost-sensitive learning |
| Feature Engineering | Feature-engine | sklearn-compatible feature transformers |
| Explainability | SHAP | Model interpretation and local explanations |
| Experiment Tracking | MLflow | Artifact, metric, and model versioning |
| API Framework | FastAPI, Uvicorn | High-performance model serving |
| Containerization | Docker | Consistent deployment across environments |
| Code Quality | Ruff, Black, isort, mypy, pytest, Bandit | Linting, formatting, typing, testing, security |
| Documentation | Sphinx / MkDocs Material | Hosted project documentation |

---

## Repository Structure

```text
Telco_Customer_Churn/
├── config/                 # Configuration files and data source definitions
├── data/                   # Raw, extracted, and processed datasets
├── docs/                   # Project documentation
│   ├── README.md
│   └── project_overview.md
├── notebook/               # Jupyter notebooks for EDA and experiments
├── src/                    # Production source code
│   ├── data/               # Data download, extract, validate, pipeline modules
│   │   ├── 01_downloader.py
│   │   ├── 02_extractor.py
│   │   ├── 03_validator.py
│   │   └── 04_pipeline.py
│   └── utils/              # Shared utilities and logging
│       └── logger.py
├── pyproject.toml          # Poetry project metadata and tool configuration
├── poetry.lock             # Deterministic dependency lock file
├── README.md               # Public project overview
└── LICENSE                 # Apache 2.0 license
```

---

## Expected Business Impact

- **Reduced churn rate** by enabling targeted retention campaigns on high-risk customers.
- **Improved marketing efficiency** through prioritized outreach rather than blanket campaigns.
- **Higher customer lifetime value** via timely interventions that extend tenure.
- **Increased operational transparency** through explainable model predictions and documented workflows.
- **Faster experimentation** with a reproducible pipeline, experiment tracking, and modular codebase.

The system is designed to be extended into a full MLOps workflow with scheduled retraining, drift detection, and production monitoring in subsequent releases.
