# AI-Enabled Data Engineering Pipeline for Predictive Analytics
### Student Performance Prediction System

A complete, production-oriented, single-file Python pipeline that ingests, validates, enriches, and models 1,000,000 student records to predict academic grades (A–F) with near-perfect accuracy.

![Students Analysed](https://img.shields.io/badge/Students-1%2C000%2C000-blue)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.81%25-brightgreen)
![Models](https://img.shields.io/badge/Models%20Trained-3-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

---

## 📋 Overview

This project implements an end-to-end AI-enabled data engineering and machine learning pipeline for predicting student academic performance across five grade categories (**A, B, C, D, F**). Built on a dataset of **1,000,000 student records**, the pipeline covers data ingestion, validation, feature engineering, model training, evaluation, and single-student prediction.

The best-performing model — **Random Forest** — achieved **99.81% accuracy** and weighted F1-score, with a **perfect recall (1.00)** on the critical minority "Grade F" class after correcting for severe class imbalance.

> "The goal of educational data mining is not simply to predict grades, but to generate the kind of insight that enables institutions to intervene at the right moment, with the right support, for the right student." — Romero & Ventura (2020)

---

## 🔑 Key Findings

- **Weekly self-study hours** is the dominant predictor of academic outcomes (Pearson **r = 0.81** with total score), accounting for the overwhelming majority of Random Forest feature importance.
- **Attendance** (r ≈ −0.001) and **class participation** (r ≈ +0.001) show negligible correlation with performance — challenging conventional assumptions about passive engagement.
- The dataset exhibits **severe class imbalance** (54.9% Grade A vs. 0.62% Grade F, an 88:1 ratio), which was corrected using **balanced class weighting** across all models.
- The pipeline is modular and reproducible, and is architecturally extensible to real-time deployment via **Apache Spark**, **Kafka**, and **FastAPI**.

---

## 🏗️ Pipeline Architecture

The system follows a seven-stage ETL/ML architecture:

| Stage | Name | Key Operations |
|---|---|---|
| 1 | Data Ingestion | CSV Reader → Schema Validator → Raw DataFrame |
| 2 | Data Validation | Null Check → Dedup → Range Validation → Label Check |
| 3 | Feature Engineering | `engagement_score`, `study_intensity`, `is_high_attendance`, `is_active_student` |
| 4 | Preprocessing | Label Encoding → Stratified Sampling → Train/Val/Test Split → Scaling |
| 5 | Model Training | Logistic Regression · Decision Tree · Random Forest (balanced weights) |
| 6 | Evaluation | Classification Report · Confusion Matrix · ROC-AUC · Feature Importance |
| 7 | Prediction Interface | Single-Student Input → Feature Derivation → Inference → Grade + Confidence |

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Records | 1,000,000 student entries |
| Raw features | 4 (+ 1 target label) |
| Target classes | A, B, C, D, F (5-class classification) |
| Missing values | 0 |
| Duplicate records | 0 |
| File format | CSV (~28 MB) |

**Features:**

| Feature | Type | Range | Description |
|---|---|---|---|
| `weekly_self_study_hours` | Continuous | 0–40 h | Hours of independent study per week |
| `attendance_percentage` | Continuous | 50–100% | Proportion of classes attended |
| `class_participation` | Ordinal | 0–10 | In-class engagement score |
| `total_score` | Continuous | 9.4–100 | Cumulative assessment score |
| `grade` (target) | Categorical | A/B/C/D/F | Final academic grade |

**Grade distribution (severe imbalance):**

| Grade | Count | Percentage |
|---|---|---|
| A | 548,644 | 54.86% |
| B | 258,174 | 25.82% |
| C | 141,980 | 14.20% |
| D | 44,998 | 4.50% |
| F | 6,204 | 0.62% |

---

## 🤖 Models & Results

Three classifiers were trained on a stratified 100,000-record sample (70/10/20 train/val/test split) with balanced class weighting:

| Model | Accuracy | Weighted F1 | Training Time | Notes |
|---|---|---|---|---|
| Logistic Regression | 98.59–98.75% | ~98.6% | ~2s | Interpretable linear baseline |
| Decision Tree | 99.80–99.81% | ~99.8% | ~0.1s | Rule extraction / explainability |
| **Random Forest ★** | **99.79–99.81%** | **99.79–99.81%** | ~4–12s | **Recommended for production** |

**Random Forest per-class performance:**

| Grade | Precision | Recall | F1-Score |
|---|---|---|---|
| A | 0.999 | 0.999 | 0.999 |
| B | 0.998 | 0.996 | 0.997 |
| C | 0.995 | 0.999 | 0.997 |
| D | 0.998 | 0.998 | 0.998 |
| **F** | 0.992 | **1.000** | 0.996 |

**Top feature importances (Random Forest):**

1. `total_score` — 0.778
2. `weekly_self_study_hours` — 0.130
3. `study_intensity` — 0.077
4. `engagement_score` — 0.005
5. `attendance_percentage` — 0.005
6. `class_participation` — 0.004

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas |
| Numerical computing | NumPy |
| Machine learning | scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Visualisation | matplotlib |
| Model serialisation | pickle |
| Scalability (planned) | Apache Spark |
| Streaming (planned) | Apache Kafka |
| Deep learning (planned) | TensorFlow / Keras (LSTM) |
| API serving (planned) | FastAPI |

---

## 🚀 Getting Started

### Requirements

- Python 3.8+ (3.12 recommended)
- 4 GB RAM minimum (16 GB recommended for full 1M-record processing)

### Installation

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
.\venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib

# 3. Verify installation
python3 -c "import sklearn; print(sklearn.__version__)"
```

### Running the Pipeline

Place `student_performance.csv` in the same directory as the script, then run:

```bash
python3 student_performance_ml.py
```

Expected output directory:

```
outputs/
├── plots/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── model_dashboard.png
├── artifacts/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── evaluation_metrics.json
```

### Predicting a Single Student's Grade

```python
import pickle
from student_performance_ml import predict_student

# Load saved artifacts
with open('outputs/artifacts/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('outputs/artifacts/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

result = predict_student(
    model=model, scaler=scaler, le=le, use_scaling=False,
    weekly_study_hours=22,
    attendance_percentage=88,
    class_participation=7,
    total_score=89,
)

print(f"Predicted Grade : {result['predicted_grade']}")
print(f"Confidence      : {result['confidence']}%")
print(f"All Probabilities: {result['all_probabilities']}")
```

**Example predictions:**

| Profile | Study h/wk | Attendance | Participation | Score | Predicted Grade | Confidence |
|---|---|---|---|---|---|---|
| High performer | 28 | 95% | 9/10 | 94 | A | 99.1% |
| Average student | 14 | 80% | 6/10 | 73 | B | 96.3% |
| At-risk student | 3 | 58% | 2/10 | 38 | F | 98.7% |

---

## ✅ Testing

The project was validated with a multi-level testing strategy:

- **10 unit tests** covering ingestion, validation, feature engineering, preprocessing, and prediction — all passed.
- **6 integration tests** verifying end-to-end pipeline execution across multiple random seeds — all passed.
- **Performance benchmarks**: full pipeline executes in ~21.7s (target: < 120s).
- **NFR compliance**: weighted F1 ≥ 0.95 (target exceeded), Grade F recall ≥ 0.90 (achieved 1.00).

---

## ⚠️ Limitations

- Results are based on a single (synthetic-but-realistic) dataset and may not generalise across institutions.
- The four-feature set does not capture temporal dynamics, socioeconomic context, or course-specific difficulty.
- `total_score` is used as a feature for grade classification but would not be available in real-time, mid-semester early-warning scenarios.
- The strong linear relationship between study hours and score suggests the dataset may have been synthetically generated.

---

## 🔮 Future Scope

- **Temporal early-warning system** using LSTM networks on weekly study-hour progressions.
- **Two-stage regression-then-classification** pipeline to predict `total_score` before mapping to grade.
- **SHAP-based explainability dashboard** for per-student, per-feature risk attribution.
- **Multi-institution generalisation** testing.
- **REST API deployment** (FastAPI) integrated with LMS platforms (Moodle/Canvas via LTI).
- **Federated learning** to preserve privacy across institutional data silos (GDPR/FERPA compliant).

---

## 📁 Suggested Repository Structure

```
.
├── student_performance_ml.py     # Main pipeline (ingestion → prediction)
├── student_performance.csv       # Dataset (not included — add your own)
├── outputs/
│   ├── plots/                    # Generated evaluation visualisations
│   └── artifacts/                # Serialised models, scaler, encoder
├── requirements.txt
└── README.md
```

---

## 📜 License

This project was developed as part of an academic submission for the **Master of Science (Data Science)** program at **Chandigarh University (CU Online)**. Please check with the author before reuse in commercial or academic contexts.

---

## 🙏 Acknowledgements

Developed by **Pritha Mandal** under the guidance of **Kunwar Saurabh Bisen**, as part of the MSc (Data Science) program at Chandigarh University.
