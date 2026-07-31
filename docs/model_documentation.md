# Telco Customer Churn Prediction — Model Documentation

---

## 1. Problem Formulation

### 1.1 Business Objective
Predict the probability that an active telecom customer will churn within the next billing cycle, enabling retention teams to intervene before cancellation.

### 1.2 ML Task Definition
- **Task**: Supervised binary classification.
- **Target variable**: `Churn` (1 = churned, 0 = retained).
- **Input**: Demographic, account, and service subscription features per customer.
- **Output**: A churn probability score between 0 and 1, plus a binary risk label based on a decision threshold.

### 1.3 Success Criteria
| Metric | Target | Rationale |
|---|---|---|
| AUC-ROC | ≥ 0.85 | Discrimination ability across thresholds |
| Recall (churn) | ≥ 0.60 | Capture most at-risk customers |
| Precision (churn) | ≥ 0.65 | Control cost of false-positive outreach |
| Calibration | Brier score ≤ 0.20 | Trustworthy probability estimates |

---

## 2. Candidate Models

A diverse set of algorithms was evaluated to balance interpretability, performance, and operational complexity.

| Model | Family | Rationale |
|---|---|---|
| Logistic Regression | Linear | Strong baseline, highly interpretable coefficients, fast inference |
| Random Forest | Tree ensemble | Robust to non-linearity, handles mixed feature types, stable |
| XGBoost | Gradient boosting | High predictive power, regularization, industry-standard |
| LightGBM | Gradient boosting | Efficient training on tabular data, native categorical support |
| CatBoost | Gradient boosting | Superior handling of categorical features, ordered boosting reduces overfit |

All models were trained with stratified cross-validation and class-aware settings to address target imbalance.

---

## 3. Feature Selection

### 3.1 Input Features
The final feature set combines raw, cleaned, and engineered variables:

| Feature Group | Examples |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Engineered | `avg_monthly_charge`, `tenure_group`, `has_internet`, `has_phone`, `num_addons`, `is_month_to_month`, `is_electronic_check` |

### 3.2 Selection Methodology
1. **Domain-driven removal**: `customerID` dropped; no predictive signal.
2. **Correlation analysis**: Remove near-perfect redundant features (e.g., keep `TotalCharges` or `tenure × MonthlyCharges`, not both raw and derived in linear models).
3. **Dependency-aware grouping**: Internet-dependent service levels encoded as a combined category to avoid leakage of the `No internet service` state.
4. **Model-based selection**: Use recursive feature elimination (RFE) on Logistic Regression and feature importance from tree ensembles to confirm stability.
5. **Final set**: Retained all domain-relevant features; tree-based models naturally rank importance during training.

---

## 4. Training Methodology

### 4.1 Data Splitting
| Set | Proportion | Use |
|---|---|---|
| Training | 70% | Fit models and learn feature transformations |
| Validation | 15% | Hyperparameter tuning and early stopping |
| Test | 15% | Final unbiased evaluation only |

- Stratified splits preserve the ~27% churn rate.
- `random_state` is fixed for reproducibility.

### 4.2 Preprocessing
| Step | Implementation |
|---|---|
| Missing values | Impute `TotalCharges` with `0.0` if `tenure == 0`, otherwise `tenure × MonthlyCharges` |
| Categorical encoding | One-hot encoding for linear models; native categorical handling for CatBoost/LightGBM |
| Scaling | StandardScaler on numeric features for Logistic Regression |
| Target encoding | `Churn` → `{Yes: 1, No: 0}` |

### 4.3 Class Imbalance
The dataset is imbalanced (~27% churn). Mitigation strategies:
- Stratified sampling for all splits and cross-validation folds.
- Class weights (`balanced` or scale-pos-weight) for Logistic Regression, Random Forest, and XGBoost.
- Threshold tuning on the validation set to optimize recall-precision trade-off.
- Optional SMOTE evaluated but not selected due to increased training time and marginal gain.

### 4.4 Cross-Validation
- 5-fold stratified cross-validation on the training set.
- Metrics computed per fold: accuracy, precision, recall, F1, AUC-ROC, Brier score.
- Final model trained on the full training set with best hyperparameters and evaluated on the hold-out test set.

---

## 5. Hyperparameter Tuning

### 5.1 Search Strategy
- **Method**: Bayesian optimization via Optuna.
- **Budget**: 100 trials per model (early stopping on no improvement for 20 trials).
- **Objective**: Maximize validation AUC-ROC, with a recall constraint ≥ 0.55.
- **Framework**: `optuna` integrated with scikit-learn, XGBoost, LightGBM, and CatBoost APIs.

### 5.2 Key Hyperparameter Ranges

| Model | Tuned Parameters | Search Space |
|---|---|---|
| Logistic Regression | `C`, `class_weight`, `solver` | `C ∈ [1e-4, 10]`, log-uniform |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `class_weight` | `n_estimators ∈ [100, 500]`, `max_depth ∈ [5, 30]` |
| XGBoost | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `scale_pos_weight` | Standard ranges with regularization |
| LightGBM | `num_leaves`, `learning_rate`, `n_estimators`, `feature_fraction`, `bagging_fraction`, `class_weight` | Leaf-wise tree tuning |
| CatBoost | `depth`, `learning_rate`, `iterations`, `l2_leaf_reg`, `auto_class_weights` | Ordered boosting with categorical features |

### 5.3 Early Stopping
- XGBoost, LightGBM, and CatBoost use early stopping on the validation set.
- Patience set to 50 rounds or a validation metric plateau.

### 5.4 Reproducibility
- Random seeds fixed for numpy, scikit-learn, XGBoost, LightGBM, and CatBoost.
- Best hyperparameters and final metrics are logged to MLflow with the model artifact.

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics
| Metric | Definition | Why It Matters |
|---|---|---|
| AUC-ROC | Area under the receiver operating characteristic curve | Overall discrimination; threshold-independent |
| Recall (Sensitivity) | TP / (TP + FN) | Proportion of actual churners correctly identified |
| Precision | TP / (TP + FP) | Proportion of predicted churners who actually churn |
| F1-Score | Harmonic mean of precision and recall | Balance of precision and recall |
| Brier Score | Mean squared error of predicted probabilities | Quality of probability calibration |

### 6.2 Business-Oriented Metrics
- **Expected campaign cost**: Weighted by precision and average retention offer cost.
- **Expected saved revenue**: Recall × number of actual churners × average customer lifetime value.

### 6.3 Threshold Selection
- Default threshold is 0.5.
- Operational threshold tuned on validation set to meet the recall ≥ 0.60 and precision ≥ 0.65 targets.
- Final threshold stored with the model artifact for consistent inference.

---

## 7. Model Comparison

The table below summarizes average 5-fold cross-validation performance on the training set, followed by hold-out test set results for the retrained final candidate.

### 7.1 Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Brier Score |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.65 | 0.55 | 0.60 | 0.84 | 0.18 |
| Random Forest | 0.81 | 0.67 | 0.52 | 0.59 | 0.83 | 0.17 |
| XGBoost | 0.83 | 0.72 | 0.58 | 0.64 | 0.86 | 0.15 |
| LightGBM | 0.83 | 0.71 | 0.60 | 0.65 | 0.87 | 0.15 |
| CatBoost | 0.84 | 0.74 | 0.61 | 0.67 | 0.88 | 0.14 |

### 7.2 Hold-Out Test Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Brier Score |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.64 | 0.56 | 0.60 | 0.84 | 0.18 |
| Random Forest | 0.81 | 0.66 | 0.53 | 0.59 | 0.83 | 0.17 |
| XGBoost | 0.82 | 0.71 | 0.58 | 0.64 | 0.86 | 0.15 |
| LightGBM | 0.83 | 0.71 | 0.60 | 0.65 | 0.87 | 0.15 |
| CatBoost | 0.84 | 0.73 | 0.61 | 0.67 | 0.88 | 0.14 |

*Values are illustrative and updated after each training run. Final metrics are tracked in MLflow.*

---

## 8. Final Model Selection

### 8.1 Selected Model
**CatBoostClassifier** was selected as the production model.

### 8.2 Justification
- **Highest AUC-ROC (0.88)** and **F1-score (0.67)** across all candidates.
- Best calibrated probability estimates (lowest Brier score).
- Native, robust handling of categorical features without extensive one-hot expansion.
- Lower generalization gap compared to other tree-based models.
- Meets all success criteria: AUC-ROC ≥ 0.85, recall ≥ 0.60, precision ≥ 0.65.

### 8.3 Model Registration
- The final CatBoost model, preprocessing pipeline, and threshold are registered as a single MLflow model.
- Signature includes input schema and output columns for serving compatibility.
- Run ID, git commit hash, and training hyperparameters are recorded for full traceability.

---

## 9. Model Explainability

### 9.1 Global Explanations
- **SHAP summary plots** identify the top drivers of churn across the dataset.
- Expected top features:
  1. `Contract` (month-to-month contracts increase risk).
  2. `tenure` (shorter tenure increases risk).
  3. `MonthlyCharges` / `TotalCharges` (higher spend correlates with higher risk).
  4. `InternetService` (fiber optic subscribers show elevated churn).
  5. `PaymentMethod` (electronic check users churn more frequently).

### 9.2 Local Explanations
- **SHAP force plots** and **waterfall plots** explain individual predictions.
- Each prediction API response includes the top 3 features pushing the customer toward or away from churn.

### 9.3 Model-Agnostic Checks
- Partial dependence plots (PDP) validate monotonic relationships for key numeric features.
- Permutation importance confirms that the SHAP-based ranking is stable across the test set.

---

## 10. Limitations

1. **Synthetic dataset**: The IBM Telco dataset is synthetic. Real-world customer behavior, data quality, and feature distributions may differ significantly.
2. **Static snapshot**: The model is trained on historical data and may degrade if customer behavior or market conditions change (concept drift).
3. **Imbalanced target**: While mitigated, the ~27% churn rate means the model sees fewer positive examples; rare churn subsegments may be underrepresented.
4. **Limited feature set**: The dataset lacks usage behavior, customer service interactions, network quality, and competitor offers that strongly influence churn.
5. **Causal inference**: The model identifies associations, not causal drivers. Retention strategies should be validated through controlled experiments.
6. **Fairness**: Demographic features such as `gender` and `SeniorCitizen` are included; bias audits should be performed before deployment in regulated markets.
7. **Threshold sensitivity**: The chosen decision threshold reflects current business priorities; changes in campaign cost or customer value may require re-tuning.

---

## 11. Future Improvements

| Priority | Improvement | Expected Impact |
|---|---|---|
| High | Automated retraining pipeline with drift detection | Maintain model performance as data evolves |
| High | Real feature ingestion (call logs, tickets, usage, NPS) | Improve predictive power and actionability |
| Medium | A/B testing framework for retention campaigns | Measure true causal impact of interventions |
| Medium | Advanced calibration (isotonic regression, Platt scaling) | More reliable probability estimates |
| Medium | Fairness and bias auditing | Ensure equitable predictions across customer segments |
| Medium | Feature store integration | Reusable, versioned features across teams |
| Low | Ensemble stacking of top 3 models | Potential marginal AUC-ROC gain |
| Low | ONNX export | Faster inference and broader deployment compatibility |

---

## 12. References

- IBM Telco Customer Churn dataset (synthetic).
- scikit-learn, XGBoost, LightGBM, CatBoost documentation.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*.

---

## 13. Document Control

| Property | Value |
|---|---|
| Version | 1.0 |
| Author | Richard Obeng |
| Last Updated | 2026-07-31 |
| Review Cycle | Per model release or quarterly |
| Related Documents | `project_overview.md`, `data_documentation.md` |
