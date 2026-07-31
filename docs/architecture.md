# Telco Customer Churn Prediction — Architecture Documentation

---

## 1. High-Level Architecture

The system is a modular, production-grade machine learning platform for predicting telecom customer churn. It follows a pipeline-based architecture separating data ingestion, preprocessing, model training, inference, and monitoring concerns.

```mermaid
flowchart TB
    subgraph Sources["Data Sources"]
        A[IBM Telco Dataset<br/>CSV Archive]
    end

    subgraph Ingestion["Data Ingestion"]
        B[Downloader]
        C[Extractor]
        D[Validator]
    end

    subgraph Processing["Preprocessing & Feature Engineering"]
        E[Cleaning & Imputation]
        F[Encoding & Scaling]
        G[Feature Engineering]
    end

    subgraph Training["Model Training"]
        H[Cross-Validation]
        I[Hyperparameter Tuning]
        J[Model Registry<br/>MLflow]
    end

    subgraph Serving["Inference Serving"]
        K[FastAPI Service]
        L[(Model Artifact)]
    end

    subgraph Observability["Observability"]
        M[Prometheus Metrics]
        N[Structured Logs]
        O[MLflow Tracking]
    end

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    J --> L --> K
    K --> M
    K --> N
    I --> O
    J --> O
```

---

## 2. Data Flow

Data moves through the system in distinct stages, each producing validated, versioned artifacts.

```mermaid
flowchart LR
    A[Raw CSV Archive] -->|download| B[Raw Data Store]
    B -->|extract| C[Extracted CSV]
    C -->|validate| D{Schema OK?}
    D -->|No| E[Alert / Halt]
    D -->|Yes| F[Cleaned Dataset]
    F -->|engineer| G[Feature Store / Training Data]
    G -->|train| H[Trained Model]
    H -->|register| I[MLflow Model Registry]
    I -->|deploy| J[FastAPI Inference Service]
    J -->|predict| K[Churn Risk Scores]
```

### Stages
1. **Download**: Fetch raw archive from configured source URL.
2. **Extract**: Unzip and place CSV in the raw data directory.
3. **Validate**: Enforce schema, types, ranges, and missing-value thresholds.
4. **Clean**: Impute missing values, fix types, remove identifiers.
5. **Engineer**: Create derived features and encode for modeling.
6. **Train**: Fit and compare candidate models; select best performer.
7. **Register**: Log model, metrics, and preprocessing artifacts to MLflow.
8. **Serve**: Load model into FastAPI and expose prediction endpoints.
9. **Monitor**: Track predictions, latency, errors, and drift.

---

## 3. Data Ingestion

### Components
| Module | File | Responsibility |
|---|---|---|
| Downloader | `src/data/01_downloader.py` | Download raw dataset archive from configured URL |
| Extractor | `src/data/02_extractor.py` | Decompress archive into `data/raw/extracted/` |
| Validator | `src/data/03_validator.py` | Validate schema, types, ranges, and duplicates |
| Pipeline | `src/data/04_pipeline.py` | Orchestrate ingestion end-to-end |

### Ingestion Flow

```mermaid
sequenceDiagram
    participant Config as config/data_url.yaml
    participant Downloader as 01_downloader.py
    participant Extractor as 02_extractor.py
    participant Validator as 03_validator.py
    participant Raw as data/raw/

    Config->>Downloader: Provide source URL
    Downloader->>Raw: Download archive
    Raw->>Extractor: Archive path
    Extractor->>Raw: Extract CSV
    Raw->>Validator: Extracted CSV
    Validator->>Validator: Check schema & quality
    Validator-->>Raw: Pass / Fail
```

### Design Decisions
- Source URLs are externalized to `config/data_url.yaml` for environment flexibility.
- Validation runs before any transformation to fail fast on data quality issues.
- Logs capture row counts, validation status, and failure reasons.

---

## 4. Data Preprocessing

Preprocessing prepares raw data for modeling while preserving reproducibility.

### Steps
1. **Identifier removal**: Drop `customerID`.
2. **Type conversion**: Convert `TotalCharges` to float; coerce invalid values to NaN.
3. **Missing value imputation**:
   - If `tenure == 0`, set `TotalCharges = 0.0`.
   - Otherwise, impute `TotalCharges = tenure × MonthlyCharges`.
4. **Whitespace cleanup**: Strip categorical strings.
5. **Target encoding**: Map `Churn` to `{Yes: 1, No: 0}`.
6. **Consistency checks**: Assert non-negative charges and valid tenure range.

### Preprocessing Pipeline

```mermaid
flowchart LR
    A[Raw DataFrame] --> B[Drop ID]
    B --> C[Convert Types]
    C --> D[Impute Missing]
    D --> E[Clean Strings]
    E --> F[Encode Target]
    F --> G[Validate Consistency]
    G --> H[Clean Dataset]
```

---

## 5. Feature Engineering

Engineered features capture domain knowledge and improve model discrimination.

### Engineered Features
| Feature | Formula / Rule |
|---|---|
| `avg_monthly_charge` | `TotalCharges / tenure` when `tenure > 0` |
| `tenure_group` | Binned: `0-12`, `13-24`, `25-48`, `49-60`, `61+` |
| `has_internet` | `InternetService != 'No'` |
| `has_phone` | `PhoneService == 'Yes'` |
| `num_addons` | Count of security/backup/support/protection add-ons |
| `is_month_to_month` | `Contract == 'Month-to-month'` |
| `is_electronic_check` | `PaymentMethod == 'Electronic check'` |

### Encoding Strategy
| Model Type | Encoding |
|---|---|
| Linear models | One-hot encoding + `StandardScaler` |
| Tree ensembles | Native categorical handling or label encoding |
| CatBoost | Native categorical features passed directly |

### Feature Engineering Flow

```mermaid
flowchart LR
    A[Clean Dataset] --> B[Create Derived Features]
    B --> C[Bin Tenure]
    C --> D[Encode Categoricals]
    D --> E[Scale Numerics]
    E --> F[Training-Ready Matrix]
```

---

## 6. Model Training Pipeline

The training pipeline is modular, reproducible, and experiment-tracked.

### Pipeline Steps

```mermaid
flowchart TB
    A[Training-Ready Data] --> B[Stratified Split<br/>70/15/15]
    B --> C[Cross-Validation<br/>5 folds]
    C --> D[Train Candidate Models]
    D --> E[Hyperparameter Tuning<br/>Optuna]
    E --> F[Evaluate on Validation Set]
    F --> G{Meets Success Criteria?}
    G -->|No| H[Adjust Features / Model]
    H --> C
    G -->|Yes| I[Train Final Model]
    I --> J[Log to MLflow]
    J --> K[Register Production Model]
```

### Candidate Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Key Controls
- Stratified splits and cross-validation preserve class distribution.
- Class weights and threshold tuning address imbalance.
- Random seeds are fixed across all libraries.
- Hyperparameters, metrics, artifacts, and code version are logged to MLflow.

---

## 7. Inference Pipeline

The inference pipeline loads the registered model and serves predictions via REST API.

### Inference Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Service
    participant Preproc as Preprocessing Pipeline
    participant Model as CatBoost Model
    participant SHAP as SHAP Explainer

    Client->>API: POST /predict or /predict/batch
    API->>API: Validate JSON schema
    API->>Preproc: Transform raw features
    Preproc->>Model: Engineered features
    Model-->>API: churn_probability
    API->>SHAP: Generate local explanation
    SHAP-->>API: top feature contributions
    API-->>Client: probability, prediction, explanation
```

### Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /predict` | Single-record prediction with explanation |
| `POST /predict/batch` | Batch scoring up to `MAX_BATCH_SIZE` records |
| `GET /health` | Service and model load status |
| `GET /ready` | Readiness for load balancers |
| `GET /metrics` | Prometheus metrics |

### Latency Targets
- p95 single prediction: < 100 ms
- p95 batch prediction (≤ 100 records): < 200 ms

---

## 8. Deployment Architecture

The service is deployed as a stateless container behind a load balancer.

```mermaid
flowchart TB
    subgraph Client["Clients"]
        A[Web App]
        B[CRM]
        C[Batch Scoring Job]
    end

    subgraph Edge["Edge"]
        D[Load Balancer / Ingress]
    end

    subgraph Cluster["Container Orchestration"]
        E[FastAPI Replica 1]
        F[FastAPI Replica 2]
        G[FastAPI Replica N]
    end

    subgraph Storage["Storage & Registry"]
        H[(MLflow Model Registry)]
        I[(Model Artifact Store)]
    end

    subgraph Observability["Observability"]
        J[Prometheus]
        K[Grafana]
        L[Log Aggregator]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    D --> F
    D --> G
    H --> I
    I -.-> E
    I -.-> F
    I -.-> G
    E --> J
    E --> L
    F --> J
    F --> L
    G --> J
    G --> L
    J --> K
```

### Deployment Options
| Environment | Tool | Notes |
|---|---|---|
| Local | Poetry + Uvicorn | Development and debugging |
| Single host | Docker Compose | Staging and small production |
| Production | Kubernetes | Auto-scaling, rolling updates, high availability |

---

## 9. Monitoring Architecture

Monitoring covers system health, model performance, and data quality.

```mermaid
flowchart LR
    A[FastAPI Service] -->|metrics| B[Prometheus]
    A -->|logs| C[Log Aggregator]
    B --> D[Grafana Dashboards]
    C --> E[Alerting Rules]
    A -->|prediction schema| F[Drift Detector]
    F --> G[Retraining Trigger]
```

### Monitored Signals
| Layer | Signal | Tool |
|---|---|---|
| System | CPU, memory, disk, network | Prometheus + node-exporter |
| Application | Request rate, latency, errors | Prometheus metrics |
| Model | Prediction distribution, AUC degradation | MLflow + drift detector |
| Data | Schema violations, missing values, drift | Validator + Great Expectations |
| Business | Recall, precision, campaign ROI | BI / reporting layer |

### Alerting Rules
| Condition | Severity |
|---|---|
| `/health` returns `model_loaded: false` | Critical |
| p95 latency > 300 ms | Warning |
| 5xx error rate > 1% | Critical |
| Input schema validation failure rate > 0.5% | Warning |
| Prediction distribution shift detected | Warning |

---

## 10. Folder Structure Overview

```text
Telco_Customer_Churn/
├── config/                     # Configuration files
│   └── data_url.yaml           # Data source URLs
├── data/                       # Data artifacts
│   ├── raw/                    # Downloaded and extracted raw data
│   └── processed/              # Cleaned and engineered datasets
├── docs/                       # Project documentation
│   ├── README.md
│   ├── project_overview.md
│   ├── data_documentation.md
│   ├── model_documentation.md
│   ├── architecture.md
│   └── deployment_guide.md
├── notebook/                   # EDA and experimentation notebooks
├── src/                        # Production source code
│   ├── api/                    # FastAPI application
│   │   ├── app.py
│   │   ├── models.py
│   │   └── routers/
│   ├── data/                   # Data pipeline modules
│   │   ├── 01_downloader.py
│   │   ├── 02_extractor.py
│   │   ├── 03_validator.py
│   │   └── 04_pipeline.py
│   ├── features/               # Feature engineering and selection
│   │   ├── build_features.py
│   │   └── preprocess.py
│   ├── models/                 # Model training, tuning, and registry
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── registry.py
│   └── utils/                  # Shared utilities
│       └── logger.py
├── tests/                      # Unit and integration tests
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Local multi-service deployment
├── pyproject.toml              # Poetry project and tool config
├── poetry.lock                 # Locked dependency graph
└── README.md                   # Public project overview
```

---

## 11. Design Principles

| Principle | Application |
|---|---|
| **Modularity** | Each pipeline stage is a separate module with a single responsibility. |
| **Reproducibility** | Fixed random seeds, locked dependencies, versioned data, and MLflow tracking. |
| **Testability** | Unit tests for each component; integration tests for pipeline and API. |
| **Observability** | Structured logs, Prometheus metrics, and health/readiness endpoints. |
| **Scalability** | Stateless service design supports horizontal scaling. |
| **Maintainability** | Clear naming, type hints, docstrings, and separation of concerns. |
| **Portability** | Docker containerization ensures consistent runtime across environments. |
| **Security** | Non-root container user, read-only model mounts, input validation, and no secrets in images. |
| **Extensibility** | New models, features, and data sources can be added without rewriting core logic. |

---

## 12. Document Control

| Property | Value |
|---|---|
| Version | 1.0 |
| Author | Richard Obeng |
| Last Updated | 2026-07-31 |
| Review Cycle | Per release or quarterly |
| Related Documents | `project_overview.md`, `model_documentation.md`, `deployment_quide.md` |
