# Config

This folder contains the centralized configuration files for the **Telco Customer Churn Prediction** project. All runtime, training, data, model, serving, logging, and path settings are defined here to keep the codebase modular, reproducible, and environment-agnostic.

---

## Purpose

The `config/` directory provides a single source of truth for project settings. Separating configuration from code makes it easier to:

- Reproduce experiments across different machines and environments.
- Update settings without modifying source code.
- Maintain consistent behavior between development, testing, staging, and production.
- Avoid hard-coded values, secrets, and environment-specific assumptions in the codebase.

---

## Configuration Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | Global project settings including metadata, reproducibility, environment, experiment tracking, serving, monitoring, and environment-specific overrides. |
| [data.yaml](data.yaml) | Data-related configuration: paths, schema, feature types, train/validation/test splits, preprocessing, feature engineering, and data validation rules. |
| [model.yaml](model.yaml) | Model configuration: default model, supported algorithms, hyperparameters, search spaces, feature selection, evaluation metrics, cross-validation, and model registry settings. |
| [training.yaml](training.yaml) | Training runtime configuration: random seeds, batch/epoch settings, cross-validation, hyperparameter tuning, early stopping, optimization, checkpoints, and experiment tracking. |
| [api.yaml](api.yaml) | FastAPI service configuration: server binding, workers, CORS, timeouts, request limits, authentication placeholders, health checks, logging, and monitoring. |
| [logging.yaml](logging.yaml) | Python logging configuration: log levels, formatters, console/file/rotating handlers, and per-module loggers with environment-specific overrides. |
| [paths.yaml](paths.yaml) | Centralized relative path definitions for data, models, artifacts, notebooks, reports, logs, deployment, and documentation. |

---

## Configuration Hierarchy

The configuration files work together in the following hierarchy:

```text
paths.yaml       → Defines where all files and directories live.
data.yaml        → Describes the dataset, features, and preprocessing.
model.yaml       → Defines models, hyperparameters, metrics, and registry.
training.yaml    → Controls how models are trained and validated.
api.yaml         → Configures the inference service.
logging.yaml     → Configures logging behavior across modules.
config.yaml      → Provides global defaults and environment overrides.
```

A typical flow looks like this:

1. The pipeline loads `paths.yaml` to locate directories and files.
2. It reads `data.yaml` to understand the schema, features, and preprocessing rules.
3. The training script loads `model.yaml` and `training.yaml` to configure models and training runs.
4. After training, artifacts are saved to paths defined in `paths.yaml` and registered using settings from `model.yaml`.
5. The FastAPI service loads `api.yaml`, `model.yaml`, and `paths.yaml` to serve predictions.
6. Logging behavior is controlled by `logging.yaml` and referenced globally.

---

## Folder Structure

```text
config/
├── README.md          # This file
├── config.yaml        # Global project configuration
├── data.yaml          # Data and preprocessing configuration
├── model.yaml         # Model and hyperparameter configuration
├── training.yaml      # Training runtime configuration
├── api.yaml           # FastAPI inference service configuration
├── logging.yaml       # Python logging configuration
└── paths.yaml         # Centralized path definitions
```

---

## Best Practices

### Avoid Hard-Coded Values
All configurable values should live in these YAML files. Source code should load settings from `config/` rather than embedding magic numbers, paths, or strings.

### Keep Secrets Out of Configuration
Never store passwords, API keys, tokens, or credentials in these files. Use environment variables or a secrets manager instead. Placeholder examples:

```yaml
# config.yaml
experiment_tracking:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  artifact_location: "${MLFLOW_ARTIFACT_LOCATION}"
```

### Use Environment-Specific Overrides
Configuration files include sections for `development`, `testing`, `staging`, and `production`. Load the appropriate section based on the `ENVIRONMENT` variable rather than maintaining separate files per environment.

### Version Control Your Configuration
Commit these files to version control so that every experiment and deployment is reproducible. Exception: local override files or `.env` files containing secrets should be ignored.

### Validate Configuration at Runtime
Loaders should validate configuration files against expected schemas. Invalid values should fail fast with clear error messages.

### Document Changes
When adding new features, hyperparameters, or endpoints, update the relevant YAML file and summarize the change in the project changelog.

---

## Maintainability

- Each file has a single responsibility, making it easy to locate and update specific settings.
- Comments explain every section so new team members can understand intent quickly.
- Consistent naming conventions and structure reduce cognitive load across files.

## Scalability

- Relative paths in `paths.yaml` allow the project to run on any system without modification.
- Modular configuration makes it straightforward to add new models, data sources, or environments.
- API and logging configs can scale from local development to containerized production deployments.

## Reproducibility

- Random seeds are declared explicitly in `config.yaml`, `data.yaml`, `model.yaml`, and `training.yaml`.
- Cross-validation, splitting, and hyperparameter search settings are version-controlled.
- MLflow experiment tracking and artifact locations are configured consistently across files.

---

## Notes

- These files are designed to be consumed by Python loaders such as PyYAML, OmegaConf, or Hydra.
- Environment variables can be interpolated at runtime using libraries like OmegaConf or python-dotenv.
- The `config/` folder should remain free of sensitive values. Use `.env` files or a secret store for credentials.

---

## Document Control

| Property | Value |
|---|---|
| Version | 1.0 |
| Author | Richard Obeng |
| Last Updated | 2026-07-31 |
| Review Cycle | Per release or quarterly |
