# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. The project focuses on turning raw customer, service, contract, payment, and billing data into a reliable modelling workflow that can be evaluated and explained.

The complete workflow is implemented in [customer_churn_prediction.ipynb](customer_churn_prediction.ipynb).

## Business Context

Customer churn creates direct revenue loss and increases the cost of acquiring replacement customers. A churn model can help a telecom provider identify at-risk customers early and prioritize retention activity.

This project treats churn prediction as a binary classification problem. The positive class is `Churn = 1`, representing a customer who leaves the service.

The project emphasizes both overall classification quality and the ability to identify churners. Accuracy alone is not sufficient because churn is an imbalanced target, so Churn-class precision, recall, F1-score, ROC-AUC, and PR-AUC are also reported.

## Objectives

- Clean and prepare the Telco Customer Churn dataset.
- Establish a baseline using multiple classification algorithms.
- Create compact and extended domain-based features.
- Compare model performance across feature sets.
- Analyze transformed feature importance.
- Compare Filter, Wrapper, and Embedded feature-selection strategies.
- Tune the strongest Gradient Boosting configuration with 5-fold Grid Search.
- Evaluate threshold-based and ranking-based metrics on an unseen test set.
- Produce a consolidated comparison table for every modelling stage.

## Dataset

The project uses the IBM Telco Customer Churn dataset stored at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains 7,043 customer records and includes:

- Demographic attributes such as gender, senior-citizen status, partner status, and dependents.
- Account attributes such as tenure, contract type, billing method, and payment method.
- Service attributes such as phone service, internet service, online security, backup, device protection, and streaming services.
- Billing attributes such as monthly charges and total charges.
- The target variable, `Churn`.

`customerID` is removed because it is an identifier rather than a predictive feature. `TotalCharges` is converted from text to numeric values, and invalid values are imputed with the median.

## Modelling Workflow

```text
Data Loading
	-> Data Cleaning
	-> Baseline Model Comparison
	-> Compact Feature Engineering
	-> Extended Feature Engineering
	-> Feature Importance
	-> Feature Selection
	-> Hyperparameter Tuning
	-> Final Evaluation
	-> Unified Comparison and Findings
```

### 1. Baseline Models

The cleaned dataset is evaluated without engineered features using:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Support Vector Machine

Numerical features are standardized with `StandardScaler`. Categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")`. Each model is evaluated inside a scikit-learn `Pipeline` and `ColumnTransformer`.

### 2. Compact Feature Engineering

The compact feature set introduces a small number of interpretable transformations:

- `tenure_group`: groups customers by lifecycle stage.
- `num_add_services`: counts additional services used by a customer.
- `monthly_charge_ratio`: relates monthly charges to customer tenure.
- Service categories are simplified by combining `No phone service` and `No internet service` with the corresponding `No` category.

### 3. Extended Feature Engineering

The extended feature set adds broader customer-behaviour and relationship signals:

- Total number of services.
- New-customer indicator.
- Family relationship indicator.
- Electronic-payment indicator.
- Month-to-month contract indicator.
- Long-term contract indicator.
- Monthly-charge threshold indicator.
- Charge relative to tenure.
- Service-tenure interaction.
- Contract-tenure combination.

The `high_monthly_charge` threshold is learned from the training partition only and then applied to the test partition. This prevents the test set from influencing feature construction.

### 4. Feature Importance

A Random Forest model is used to inspect the importance of transformed features. The analysis highlights billing, tenure, service usage, and contract-related variables as important contributors to churn predictions.

### 5. Feature Selection

Three strategies are compared:

- **Filter:** Mutual Information with `SelectKBest`.
- **Wrapper:** Recursive Feature Elimination with Logistic Regression.
- **Embedded:** L1-regularized Logistic Regression with `SelectFromModel`.

Each selector is placed inside a modelling pipeline so that selection is learned from training data rather than from the test set.

### 6. Hyperparameter Tuning

Gradient Boosting is tuned with `GridSearchCV` using 5-fold cross-validation. The search optimizes Churn-class F1-score across:

- Number of estimators.
- Learning rate.
- Maximum tree depth.
- Minimum samples required to split a node.
- Minimum samples required at a leaf.

## Evaluation Metrics

### Threshold-Based Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churners who actually churn.
- **Recall:** Proportion of actual churners identified by the model.
- **Churn F1-score:** Harmonic mean of churn precision and churn recall.

### Ranking-Based Metrics

- **ROC-AUC:** Measures how well the model ranks positive examples above negative examples across thresholds.
- **PR-AUC:** Summarizes the precision-recall trade-off and is especially useful for an imbalanced churn target.

For classifiers that expose probabilities, positive-class probabilities are used. For the linear Support Vector Machine, the decision-function margin is used as the ranking score.

## Results

The best accuracy and Churn F1-score came from Gradient Boosting with the leakage-aware extended feature set. Its Churn F1-score is tied with the baseline Logistic Regression result, so the main improvement is accuracy and ranking quality rather than a unique F1-score improvement.

### Best Extended Feature-Engineering Model

| Metric | Result |
| --- | ---: |
| Model | Gradient Boosting |
| Accuracy | 0.812 |
| Precision | 0.685 |
| Recall | 0.540 |
| Churn F1-score | 0.604 |
| ROC-AUC | 0.846 |
| PR-AUC | 0.659 |

### Tuned Model

| Metric | Result |
| --- | ---: |
| Model | Tuned Gradient Boosting |
| Accuracy | 0.808 |
| Precision | 0.678 |
| Recall | 0.529 |
| Churn F1-score | 0.595 |
| ROC-AUC | 0.846 |
| PR-AUC | 0.666 |

Tuning improved PR-AUC, which indicates better ranking quality for the imbalanced target, but it did not improve accuracy or Churn F1-score over the extended feature-engineered model.

### Stage-Level Summary

| Stage | Model | Accuracy | Churn F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Logistic Regression | 0.806 | 0.604 | 0.842 | 0.633 |
| Extended features | Gradient Boosting | 0.812 | 0.604 | 0.846 | 0.659 |
| Feature selection | Embedded + Gradient Boosting | 0.806 | 0.590 | 0.845 | 0.659 |
| Tuned model | Gradient Boosting | 0.808 | 0.595 | 0.846 | 0.666 |

## Project Structure

```text
Customer-Churn-Prediction/
├── customer_churn_prediction.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended. Create an isolated environment from the project directory:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Notebook

Open the notebook in VS Code with the Jupyter extension or launch Jupyter directly:

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Run the cells from top to bottom. The notebook expects the dataset path to remain relative to the project directory. Running cells out of order can leave stale variables in the kernel, so restarting the kernel and running all cells is recommended after code changes.

## Reproducibility

- Train-test split: 80/20.
- Split random state: `42`.
- Stratification: enabled for the churn target.
- Hyperparameter tuning: 5-fold cross-validation.
- Feature construction threshold: learned from training data only.
- Model preprocessing: contained inside scikit-learn pipelines.

The notebook records intermediate comparison tables, final metrics, feature importance results, and the unified modelling-stage table.

## Resume-Ready Summary

Built an end-to-end telecom customer churn prediction workflow using six classification models, leakage-aware domain feature engineering, Filter/Wrapper/Embedded feature-selection experiments, and 5-fold Grid Search. Achieved 81.2% test accuracy with Gradient Boosting and extended features, and evaluated imbalanced-class performance using Churn F1-score, ROC-AUC, and PR-AUC.

### Resume Bullet Options

- Built a telecom customer churn classification pipeline using six machine-learning models, scikit-learn preprocessing pipelines, and domain-driven feature engineering.
- Achieved 81.2% test accuracy and 0.604 Churn F1-score with Gradient Boosting and an extended feature set; evaluated ranking performance with ROC-AUC and PR-AUC.
- Compared Filter, Wrapper, and Embedded feature selection and tuned Gradient Boosting with 5-fold Grid Search, improving PR-AUC to 0.666.
- Prevented test-set leakage by learning a monthly-charge threshold from training data only and applying it consistently to validation data.

## Limitations and Future Work

- Results use one stratified train-test split. Repeated cross-validation or an external holdout dataset would provide a stronger estimate of generalization.
- The churn class is imbalanced. Future work should evaluate threshold optimization, calibration, class weighting, and business-cost-sensitive metrics.
- The project does not yet include deployment, monitoring, model versioning, or a production inference API.
- A future version could add explainability with SHAP or permutation-based local explanations.
- Retention experiments would be needed to measure whether model-driven interventions reduce actual churn.

## License and Dataset Note

This repository is intended for educational and portfolio use. The dataset is included locally in the `data/` directory; consult its original source and terms before redistributing it.
