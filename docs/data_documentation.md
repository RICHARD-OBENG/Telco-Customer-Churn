# Telco Customer Churn — Data Documentation

---

## Dataset Source

The project uses the publicly available **IBM Telco Customer Churn** dataset, a synthetic but realistic dataset widely used for churn prediction benchmarks. It contains customer demographic, account, and service subscription information along with a churn label.

| Property | Value |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Rows | 7,043 |
| Columns | 21 |
| File | `data/raw/extracted/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Format | CSV |

---

## Dataset Overview

Each row represents a unique telecom customer. The dataset combines:

- **Demographic information** — gender, senior citizen status, partner, dependents.
- **Account information** — tenure, contract type, billing preferences, payment method, charges.
- **Service subscriptions** — phone, internet, online security, backup, tech support, streaming.
- **Target label** — whether the customer left within the last month.

The data is structured, tabular, and relatively clean, making it suitable for classical supervised learning algorithms as well as gradient-boosted tree models.

---

## Data Schema

| Column | Data Type | Description |
|---|---|---|
| `customerID` | string | Unique customer identifier |
| `gender` | categorical | Customer gender: `Male`, `Female` |
| `SeniorCitizen` | integer | 1 if customer is 65 or older, 0 otherwise |
| `Partner` | categorical | Whether the customer has a partner: `Yes`, `No` |
| `Dependents` | categorical | Whether the customer has dependents: `Yes`, `No` |
| `tenure` | integer | Number of months the customer has stayed |
| `PhoneService` | categorical | Whether the customer has phone service: `Yes`, `No` |
| `MultipleLines` | categorical | Multiple phone lines: `Yes`, `No`, `No phone service` |
| `InternetService` | categorical | Internet provider: `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | categorical | Online security add-on: `Yes`, `No`, `No internet service` |
| `OnlineBackup` | categorical | Online backup add-on: `Yes`, `No`, `No internet service` |
| `DeviceProtection` | categorical | Device protection add-on: `Yes`, `No`, `No internet service` |
| `TechSupport` | categorical | Tech support add-on: `Yes`, `No`, `No internet service` |
| `StreamingTV` | categorical | TV streaming service: `Yes`, `No`, `No internet service` |
| `StreamingMovies` | categorical | Movie streaming service: `Yes`, `No`, `No internet service` |
| `Contract` | categorical | Contract term: `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | categorical | Whether billing is paperless: `Yes`, `No` |
| `PaymentMethod` | categorical | Payment method used |
| `MonthlyCharges` | float | Monthly recurring charge |
| `TotalCharges` | float | Total amount charged to date |
| `Churn` | categorical | Target: `Yes` if customer churned, `No` otherwise |

---

## Feature Descriptions

### Demographic Features

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`: basic customer profile attributes used to segment churn risk.

### Behavioral / Tenure Features

- `tenure`: highly predictive of churn; new customers are generally more likely to leave.
- `MonthlyCharges`: current monthly spend.
- `TotalCharges`: lifetime spend; should be derived from `tenure × MonthlyCharges` and used to detect inconsistencies.

### Service Subscription Features

Phone and internet-related features encode the breadth and type of services subscribed. Many are dependent on whether the customer has internet service. These are handled through grouped encoding or dependency-aware feature construction.

### Contract & Billing Features

- `Contract`: month-to-month contracts show significantly higher churn.
- `PaperlessBilling` and `PaymentMethod`: electronic check payments correlate with higher churn in this dataset.

---

## Target Variable

- **Name**: `Churn`
- **Type**: Binary classification target
- **Values**: `Yes` (churned), `No` (retained)
- **Encoding**: Mapped to `1` for churn and `0` for retained.
- **Class balance**: Approximately 27% churn, 73% retained — imbalanced, requiring stratified sampling and class-aware modeling.

---

## Missing Value Handling

The IBM Telco dataset contains a small number of blank strings in `TotalCharges`, typically for customers with zero tenure. These are treated as missing values.

| Step | Action |
|---|---|
| Detection | Identify empty strings and non-numeric entries in `TotalCharges` |
| Imputation | Fill with `0.0` when `tenure == 0`, otherwise fill with `tenure × MonthlyCharges` |
| Validation | Assert no missing values remain after imputation |

No other columns contain missing values in the source data.

---

## Data Cleaning

1. **Identifier removal**: Drop `customerID` before modeling; it carries no predictive signal.
2. **Type conversion**: Convert `TotalCharges` from string/object to float.
3. **Whitespace stripping**: Trim leading/trailing whitespace from categorical values.
4. **Consistency checks**: Ensure `TotalCharges` is approximately equal to `tenure × MonthlyCharges`.
5. **Target encoding**: Convert `Churn` to binary integers.

---

## Feature Engineering

The following engineered features are created to improve model performance:

| Feature | Description |
|---|---|
| `avg_monthly_charge` | `TotalCharges / tenure` (for tenure > 0) |
| `tenure_group` | Binned tenure into segments: `0-12`, `13-24`, `25-48`, `49-60`, `61+` |
| `has_internet` | Binary indicator for any internet service |
| `has_phone` | Binary indicator for phone service |
| `num_addons` | Count of subscribed security/backup/support/protection services |
| `is_month_to_month` | Binary indicator for month-to-month contract |
| `is_electronic_check` | Binary indicator for electronic check payment method |

Categorical variables are encoded using one-hot or ordinal encoding depending on model requirements. Numerical variables are standardized for linear models and left raw for tree-based models.

---

## Data Validation

Data validation is implemented in `src/data/03_validator.py` and enforces:

- All expected columns are present.
- Column data types match the schema.
- `MonthlyCharges` and `TotalCharges` are non-negative.
- `tenure` is within a valid range.
- No duplicate `customerID` values.
- Target variable contains only valid labels.
- Missing value count is below an acceptable threshold.

Validation failures raise descriptive errors so pipeline issues are caught early.

---

## Train, Validation, and Test Split

The dataset is split using stratified sampling to preserve the churn class distribution across sets.

| Set | Proportion | Purpose |
|---|---|---|
| Training | 70% | Model fitting and feature learning |
| Validation | 15% | Hyperparameter tuning and model selection |
| Test | 15% | Final, unbiased performance evaluation |

- `random_state` is fixed for reproducibility.
- `stratify` is set on the target variable.
- Cross-validation is performed on the training set during model development.

---

## Data Quality Considerations

| Risk | Mitigation |
|---|---|
| Class imbalance | Stratified splits, class weights, SMOTE or threshold tuning |
| Leakage from `TotalCharges` | Derive carefully; avoid using post-hoc information |
| Categorical dependency | Group internet-dependent features before encoding |
| Synthetic data limitations | Validate assumptions before production deployment |
| Concept drift | Monitor feature distributions and model performance over time |
| Reproducibility | Pin random seeds, version data with DVC, and lock dependencies |

This documentation should be updated whenever the dataset, schema, or preprocessing logic changes.
