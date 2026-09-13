<div align="center">

# Customer Churn Prediction Using Machine Learning

### An end-to-end telecom churn classification workflow with feature engineering, model comparison, feature selection, and hyperparameter tuning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-2EA44F)](#results--model-comparison)

</div>

---

## About

This project predicts whether a telecommunications customer is likely to churn. It compares six classification models, evaluates two feature-engineering strategies, investigates feature importance and feature selection, and tunes Gradient Boosting with Grid Search.

The complete implementation is available in [`customer_churn_prediction.ipynb`](customer_churn_prediction.ipynb).

---

## Project Objectives

- Establish baseline performance using multiple classification models.
- Create compact and extended domain-based features.
- Compare models using consistent train-test splits and metrics.
- Analyze important transformed features.
- Evaluate Filter, Wrapper, and Embedded feature-selection methods.
- Tune the strongest Gradient Boosting configuration.
- Identify the best-performing modelling approach for churn prediction.

---

## Business Problem

Customer churn can reduce recurring revenue and increase customer-acquisition costs. A churn prediction model can help identify customers who may need targeted retention attention.

This is a binary classification problem. `Churn = 1` represents a customer who leaves the service.

---

## Dataset

The project uses the Telco Customer Churn dataset:

- **Records:** 7,043
- **Original columns:** 21
- **Target:** `Churn`
- **Type:** Binary classification
- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

`customerID` is removed before modelling. `TotalCharges` is converted from `object` to numeric, and `Churn` is encoded as `0` and `1`.

### Dataset Features / Important Features

The dataset contains customer demographics, tenure, services, contract information, payment behaviour, monthly charges, and total charges.

The most influential features identified by the Random Forest analysis were:

1. `charge_per_tenure_year`
2. `MonthlyCharges`
3. `TotalCharges`
4. `service_tenure_interaction`
5. `tenure`
6. `is_month_to_month`
7. `Contract_Month-to-month`
8. `is_long_term_contract`
9. `num_add_services`
10. `PaymentMethod_Electronic check`

---

## Project Workflow

```text
Data Loading
    -> Data Cleaning & Preprocessing
    -> Baseline Model Comparison
    -> Compact Feature Engineering
    -> Extended Feature Engineering
    -> Feature Set Comparison
    -> Feature Importance
    -> Feature Selection
    -> Hyperparameter Tuning
    -> Final Evaluation
    -> Final Comparison & Findings
```

---

## Models Used

| Model | Role |
| --- | --- |
| Logistic Regression | Baseline linear classifier and RFE estimator |
| Random Forest Classifier | Baseline model and feature-importance analysis |
| Gradient Boosting Classifier | Model selected for hyperparameter tuning |
| XGBoost | Baseline tree-based classifier |
| LightGBM | Baseline gradient-boosting classifier |
| Support Vector Machine | Baseline margin-based classifier |

---

## Data Cleaning & Preprocessing

- Converted `TotalCharges` to numeric values.
- Imputed invalid `TotalCharges` values with the median.
- Encoded `Churn` as `0` and `1`.
- Removed rows with missing target values.
- Removed `customerID` because it is an identifier.
- Standardized numerical features using `StandardScaler`.
- One-hot encoded categorical features using `OneHotEncoder`.
- Used scikit-learn `Pipeline` and `ColumnTransformer` objects.
- Used an 80/20 stratified train-test split with `random_state=42`.

---

## Feature Engineering

### Compact Feature Engineering

The compact feature set includes:

- `tenure_group`
- `num_add_services`
- `monthly_charge_ratio`
- Simplified service-related categorical values

### Extended Feature Engineering

The extended feature set adds domain-based signals related to:

- Tenure and customer lifecycle
- Service usage
- Customer relationships
- Payment behaviour
- Contract type
- Monthly and total charges
- Feature interactions

Important engineered features include `charge_per_tenure_year`, `service_tenure_interaction`, `is_month_to_month`, `is_long_term_contract`, and `num_add_services`.

The `high_monthly_charge` threshold is learned from training data only to avoid test-set leakage.

---

## Feature Importance

Random Forest was used to estimate the relative importance of transformed features. Billing, tenure, service usage, and contract-related variables were among the strongest signals in the analysis.

---

## Feature Selection

Three feature-selection strategies were evaluated:

| Method | Implementation |
| --- | --- |
| **Filter** | Mutual Information with `SelectKBest` |
| **Wrapper** | Recursive Feature Elimination (`RFE`) |
| **Embedded** | L1 Logistic Regression with `SelectFromModel` |

The best feature-selected configuration was **Embedded Feature Selection + Gradient Boosting**. However, the full extended feature set performed better overall, so it was retained for hyperparameter tuning.

---

## Model Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churn.
- **Recall:** Proportion of actual churners identified.
- **F1-score:** Balance between Churn precision and recall.
- **ROC-AUC:** Ranking quality across classification thresholds.
- **PR-AUC:** Precision-recall performance for the imbalanced churn target.
- **Confusion Matrix:** Counts of correct and incorrect predictions by class.

---

## Cross-Validation

Hyperparameter tuning used **5-fold cross-validation** on the training data. The held-out test set was reserved for final evaluation.

---

## Hyperparameter Tuning

Tuning was performed **only on Gradient Boosting**, using the full extended feature set and `GridSearchCV` with `scoring="f1"`.

| Parameter | Values |
| --- | --- |
| `n_estimators` | `[100, 150, 200]` |
| `learning_rate` | `[0.05, 0.1, 0.15]` |
| `max_depth` | `[2, 3, 4]` |
| `min_samples_split` | `[2, 5]` |
| `min_samples_leaf` | `[1, 2]` |

The search evaluated **108 combinations** across **540 fits**.

**Best parameters:** `learning_rate=0.05`, `max_depth=2`, `min_samples_leaf=1`, `min_samples_split=2`, and `n_estimators=200`.

**Best cross-validation F1-score:** `0.586`

---

## Results / Model Comparison

| Stage | Best Configuration | Accuracy | Precision | Recall | Churn F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Logistic Regression | 80.6% | 65.7% | 55.9% | 0.604 |
| Compact Feature Engineering | Logistic Regression | 80.3% | 66.4% | 51.9% | 0.583 |
| Extended Feature Engineering | Gradient Boosting | **81.2%** | **68.5%** | 54.0% | **0.604** |
| Feature Selection | Embedded + Gradient Boosting | 80.6% | 67.0% | 52.7% | 0.590 |
| Hyperparameter Tuning | Gradient Boosting | 80.8% | 67.8% | 52.9% | 0.595 |

### Best Performing Approach

**Gradient Boosting + Extended Feature Engineering**

| Metric | Test Result |
| --- | ---: |
| Accuracy | **81.2%** |
| Precision | **68.5%** |
| Recall | **54.0%** |
| Churn F1-score | **0.604** |
| ROC-AUC | **0.846** |
| PR-AUC | **0.659** |

The tuned model achieved `80.8%` accuracy and a `0.595` Churn F1-score. It did not outperform the original extended-feature Gradient Boosting model.

---

## Key Findings

- Feature engineering provided the strongest performance improvement.
- Extended feature engineering produced the highest test accuracy.
- The extended model's Churn F1-score of `0.604` is tied with baseline Logistic Regression.
- Feature selection reduced performance compared with the full extended feature set.
- Hyperparameter tuning improved neither final accuracy nor Churn F1-score.
- The full extended feature set was retained as the final modelling configuration.

---

## Business Insights

The feature-importance results suggest that churn predictions are strongly associated with billing levels, tenure, service adoption, contract type, and electronic-check payment behaviour. These signals can support targeted retention analysis, but they should be validated with business experiments before being used for customer decisions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

---

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
git clone https://github.com/bhavyasree-22/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook
```

Open `customer_churn_prediction.ipynb` and run the cells from top to bottom. The dataset path is relative to the project directory.

---

## Future Enhancements

- Repeat evaluation across multiple stratified folds or an external holdout dataset.
- Optimize the classification threshold for retention-focused business costs.
- Add model calibration and explainability analysis.
- Package the preprocessing and model pipeline for reproducible inference.
- Add deployment and monitoring workflows.

---

## Conclusion

This project demonstrates a complete machine-learning workflow for telecom churn prediction. The final comparison shows that feature engineering, rather than hyperparameter tuning or feature selection, provided the strongest improvement. The overall best approach was Gradient Boosting with the extended feature set, achieving `81.2%` test accuracy and a `0.604` Churn F1-score.

---

## Author

**Bhavya Sree Gubba**  
B.Tech CSE (AI & ML), VIT-AP University

**Profiles:** [GitHub](https://github.com/bhavyasree-22) · [LinkedIn](https://www.linkedin.com/in/bhavya-sree-22122006bs/)
