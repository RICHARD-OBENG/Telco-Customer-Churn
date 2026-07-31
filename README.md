# Telco Customer Churn Prediction

A production-grade machine learning system that predicts whether a telecom customer is likely to churn based on demographics, services, account information, and usage behavior.

---

## Business Problem

Customer churn directly reduces recurring revenue and increases acquisition costs for telecom providers. Identifying customers at risk of leaving before they cancel enables targeted retention campaigns, proactive support, and improved customer lifetime value. This project builds an end-to-end ML pipeline that transforms raw customer data into actionable churn risk scores.

---

## Objectives

- Build a reproducible data pipeline for ingestion, validation, and preprocessing.
- Train and compare multiple classification models (logistic regression, tree-based ensembles, gradient boosting).
- Optimize for recall and precision to support retention campaign targeting.
- Track experiments, artifacts, and model versions with MLflow.
- Package the trained model behind a FastAPI service for real-time inference.
- Deploy with Docker for consistent production environments.

---

## Key Features

- Modular, maintainable pipeline under `src/`.
- Automated data validation and cleaning steps.
- Feature engineering and selection for structured tabular data.
- Model comparison across scikit-learn, XGBoost, LightGBM, and CatBoost.
- SHAP-based model explainability for transparent predictions.
- Experiment tracking and artifact logging with MLflow.
- REST API for single-record and batch churn predictions.
- Containerized deployment with Docker.
- CI-ready tooling: pytest, Ruff, Black, isort, mypy, Bandit.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly, Missingno |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn, Feature-engine, SHAP |
| Experiment Tracking | MLflow |
| API & Serving | FastAPI, Uvicorn |
| Packaging | Poetry, Docker |
| Code Quality | Ruff, Black, isort, mypy, pytest, Bandit, Pylint |
| Documentation | Sphinx, MkDocs Material |
| Notebooks | JupyterLab, ipykernel |

---

## Project Structure

```text
Telco_Customer_Churn/
├── config/                 # Configuration files and data URLs
├── data/                   # Raw and processed data
├── docs/                   # Documentation
├── notebook/               # Exploratory data analysis and experiments
├── src/                    # Source code
│   ├── data/               # Download, extract, validate, pipeline modules
│   └── utils/              # Shared utilities and logger
├── pyproject.toml          # Poetry project configuration
├── poetry.lock             # Locked dependency graph
├── README.md               # Project overview
└── LICENSE                 # Apache 2.0
```

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/RICHARD-OBENG/telco-customer-churn.git
cd telco-customer-churn

# Install Poetry (if not already installed)
pipx install poetry

# Install all dependencies
poetry install

# Install with development, testing, and notebook groups
poetry install --with dev,test,lint,format,type,docs,notebook

# Activate the virtual environment
poetry shell
```

---

## Quick Start

Run the data pipeline to download, extract, and validate the Telco dataset:

```bash
poetry run python src/data/01_downloader.py
poetry run python src/data/02_extractor.py
poetry run python src/data/03_validator.py
```

Launch JupyterLab for exploratory analysis:

```bash
poetry run jupyter lab
```

Start the FastAPI prediction service:

```bash
poetry run uvicorn src.api.app:app --reload
```

The API documentation will be available at `http://localhost:8000/docs`.

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.65 | ~0.55 | ~0.60 | ~0.84 |
| Random Forest | ~0.81 | ~0.67 | ~0.52 | ~0.59 | ~0.83 |
| XGBoost | ~0.83 | ~0.72 | ~0.58 | ~0.64 | ~0.86 |
| LightGBM | ~0.83 | ~0.71 | ~0.60 | ~0.65 | ~0.87 |
| CatBoost | ~0.84 | ~0.74 | ~0.61 | ~0.67 | ~0.88 |

*Values are illustrative and updated after each training run. Final metrics are tracked in MLflow and stored with the best model artifact.*

---

## Deployment Overview

The trained model is exposed through a FastAPI application and can be containerized with Docker:

```bash
# Build the Docker image
docker build -t telco-churn-api:latest .

# Run the container
docker run -p 8000:8000 telco-churn-api:latest
```

The `/predict` endpoint accepts JSON payloads and returns churn probability and binary risk classification.

---

## Documentation Links

- Project documentation: [https://richard-obeng.github.io/telco-customer-churn/](https://richard-obeng.github.io/telco-customer-churn/)
- API docs (local): [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow tracking (local): [http://localhost:5000](http://localhost:5000)

---

## Future Improvements

- Add automated hyperparameter tuning with Optuna.
- Implement model monitoring and drift detection in production.
- Add batch inference pipeline for campaign list scoring.
- Integrate DVC for data and model versioning.
- Expand feature set with usage behavior and customer service interaction data.
- Add A/B testing framework for retention campaign optimization.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Author

**Richard Obeng**
- Email: [richardkwabenaobeng17@gmail.com](mailto:richardkwabenaobeng17@gmail.com)
- GitHub: [https://github.com/RICHARD-OBENG](https://github.com/RICHARD-OBENG)
