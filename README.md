# 🏥 Multiple Disease Prediction System
### A production-grade clinical decision-support web application

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-0073B7?style=flat)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
<img width="1910" height="966" alt="image" src="https://github.com/user-attachments/assets/c212fecf-7673-422b-bf37-26e93964d767" />


## 📌 Overview

This project is a **multi-disease clinical prediction platform** built with Python and Streamlit. It predicts the likelihood of **8 diseases** from patient data using trained machine learning models, and also supports **symptom-based general disease diagnosis**.

The system goes beyond a simple model wrapper — it features a **rigorous ML pipeline** with ensemble learning, calibrated probabilities, SMOTE-based class balancing, and automated PDF reporting.

> Built as part of a data science internship at **Innomatics Research Labs**.

---

## 🎯 Diseases Covered

| Module | Dataset | Model |
|---|---|---|
| **Diabetes** | PIMA Indians Diabetes Dataset | Stacking / Soft-Voting Ensemble |
| **Heart Disease** | UCI Heart Disease | Scikit-learn Classifier |
| **Parkinson's Disease** | UCI Parkinson's | Scikit-learn Classifier |
| **Breast Cancer** | Wisconsin Breast Cancer | Scikit-learn Classifier |
| **Liver Disease** | ILPD Dataset | Scikit-learn Classifier |
| **Hepatitis C** | UCI Hepatitis C | Scikit-learn Classifier |
| **Chronic Kidney Disease** | CKD Dataset | Scikit-learn Classifier |
| **Lung Cancer** | Survey-based Dataset | Scikit-learn Classifier |
| **Symptom Checker** | 132-symptom dataset | XGBoost |

---

## 🧠 ML Pipeline — Diabetes Module (PIMA)

The diabetes prediction module showcases a **production-level ML pipeline** with the following stages:

```
Raw CSV → Feature Engineering → Preprocessing → Model Training → Ensemble → Evaluation → PDF Report
```

### 1. Feature Engineering
Custom domain-informed features added on top of raw PIMA features:
```python
glucose_x_bmi   = Glucose × BMI          # metabolic risk interaction
glucose_by_age  = Glucose / (Age + ε)    # age-normalized glucose load
bmi_sq          = BMI²                    # non-linear obesity signal
```

### 2. Preprocessing
- **Median imputation** for missing values (biologically plausible zeros replaced)
- **StandardScaler** normalization
- **KFold Target Encoding** for categorical variables (leak-safe, OOF)
- **SelectKBest** (ANOVA F-score) for feature selection

### 3. Class Imbalance — SMOTETomek
```python
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)
```
Combines oversampling of the minority class (SMOTE) with removal of noisy borderline samples (Tomek links).

### 4. Model Benchmarking
11 classifiers evaluated via **5-fold stratified cross-validation**:

| Model | CV Accuracy |
|---|---|
| **ExtraTrees** | **85.2%** ✅ |
| **CatBoost** | 84.4% |
| **GradientBoosting** | 84.0% |
| **MLP Neural Network** | 83.7% |
| RandomForest | 83.6% |
| HistGradientBoosting | 83.2% |
| XGBoost | 82.8% |
| KNN | 79.7% |
| SVC | 78.8% |
| DecisionTree | 78.0% |
| GaussianNB | 74.2% |

### 5. Ensemble Strategy
Top-4 models selected and combined using two strategies:

**Soft Voting with Optimized Weights**
```python
from scipy.optimize import minimize
# Numerically minimize negative CV accuracy to find optimal per-model weights
res = minimize(ensemble_obj, x0=init, bounds=bounds, constraints=cons, method='SLSQP')
```

**Stacking with Logistic Regression meta-learner**
```python
stacking = StackingClassifier(
    estimators=top_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5,
    stack_method='predict_proba'
)
```

Final model chosen is whichever achieves higher holdout accuracy.

### 6. Probability Calibration
All base models wrapped with `CalibratedClassifierCV` (sigmoid method) for reliable probability outputs — important for clinical settings.

### 7. Automated PDF Report
Pipeline auto-generates a structured PDF (`model_report.pdf`) with:
- CV score comparison chart
- Holdout set metrics (blend vs stack)
- Model selection rationale

---

## 🗂️ Project Structure

```
Disease_prediction_app/
│
├── Frontend/                     # Streamlit web application
│   ├── app.py                    # Main multi-page app
│   ├── code/
│   │   ├── DiseaseModel.py       # Symptom-based XGBoost predictor
│   │   ├── helper.py             # Data loading & preprocessing utilities
│   │   └── train.py              # Symptom classifier training
│   ├── models/                   # Serialized .sav model files (8 diseases)
│   └── data/                     # CSVs: symptoms, severity, precautions
│
└── code/PIMA/                    # End-to-end diabetes ML pipeline
    ├── config.yml                # Experiment configuration (YAML-driven)
    ├── run_pipeline.py           # Pipeline entrypoint
    ├── data_prep.py              # KFold target encoder + numeric preprocessing
    ├── feature_engineer.py       # Domain feature generation
    ├── models.py                 # Model factory (11 classifiers)
    ├── training.py               # Full training loop + ensemble + calibration
    ├── evaluation.py             # Metrics + PDF report generation
    └── artifacts/
        ├── final_model.joblib    # Saved final model
        ├── preproc.joblib        # Saved preprocessing pipeline
        ├── cv_scores.json        # Benchmark results
        └── model_report.pdf      # Auto-generated PDF report
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Anaconda (recommended)

### Setup

```bash
# Clone the repo
git clone https://github.com/KAVYADESHINI/IPL_ANALYSIS.git
cd Disease_prediction_app

# Create environment
conda create -n disease_app python=3.10
conda activate disease_app

# Install dependencies
conda install numpy pandas scikit-learn -c conda-forge
pip install streamlit joblib xgboost catboost imbalanced-learn matplotlib
```

### Run the Web App

```bash
cd Frontend
streamlit run app.py
```

### Retrain the Diabetes Pipeline

```bash
cd code/PIMA
python run_pipeline.py
# Artifacts saved to code/PIMA/artifacts/
```

---

## ⚙️ Configuration

The pipeline is fully config-driven via `config.yml` — no hardcoding needed:

```yaml
data_path: "pima_diabetes.csv"
target: "Outcome"
test_size: 0.2
random_state: 42
cv_folds: 5
top_n: 4            # Number of top models to ensemble
n_trials: 30        # Optuna tuning trials (if enabled)
use_gpu: false
save_pickle: true
```

---

## 📊 Key Design Decisions

| Decision | Rationale |
|---|---|
| SMOTETomek over plain SMOTE | Removes noisy borderline samples; reduces false positives |
| KFold Target Encoding | Prevents target leakage in categorical features during CV |
| Calibrated probabilities | Clinically, probability estimates matter — not just class labels |
| SLSQP weight optimization | Data-driven ensemble weighting outperforms uniform voting |
| Stack vs Blend comparison | Automatically selects best strategy per run |
| YAML config | Reproducible experiments; easy hyperparameter sweeps |

---

## 🔮 Future Improvements

- [ ] Add Optuna-based hyperparameter tuning (scaffolded in config)
- [ ] SHAP explainability for each prediction
- [ ] Patient history tracking (database integration)
- [ ] REST API with FastAPI for model serving
- [ ] Docker containerization for deployment
- [ ] Unit tests for preprocessing pipeline

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `XGBoost` · `CatBoost` · `imbalanced-learn` · `pandas` · `NumPy` · `SciPy` · `joblib` · `matplotlib`

---

## 👩‍💻 Author

**Kavya Deshini**  
Data Science Intern @ Innomatics Research Labs  
[GitHub](https://github.com/KAVYADESHINI) · [LinkedIn](#)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

> *"The goal is to turn data into information, and information into insight."* — Carly Fiorina
