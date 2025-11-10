# Testing & Optimization Log — Churn Prediction

**Author:** Krystin Villeneuve  
**Date:** November 9, 2025

## Environment
- Python: 3.11.x
- OS: 🔧 (macOS/Windows/Linux)
- Commit SHA: 🔧 (git rev-parse HEAD)
- Packages: scikit-learn, pandas, numpy, streamlit, matplotlib

## Data Integrity
- Rows/Cols: 7043 × 28 (raw); 27 after dropping `Count`
- Target distribution (raw): ~0: 0.735, 1: 0.265
- Nulls: `Total Charges` = 11 → median imputed
- Splits: 70/15/15, stratified, seed=42

---

## Experiment 01 — Baseline Logistic Regression
**Preprocess:** median numeric, most-frequent categorical, one-hot (train-only fit)  
**Model:** LogisticRegression (`class_weight="balanced"`, `max_iter=2000`)  
**Threshold:** 0.50  
**Test Metrics:**
- Accuracy: **0.775**
- Precision: **0.561**
- Recall: **0.701**
- F1: **0.623**
- ROC-AUC: **0.830**
**Artifacts:** `models/baseline_logreg.joblib`, `models/metrics.json`, ROC/CM PNGs  
**Notes:** Good recall; use thresholding for business-tuned trade-offs.

---

## Experiment 02 — Threshold Sweep
**Goal:** Improve recall while controlling precision.  
**Method:** Sweep thresholds 0.30–0.70 by 0.01.  
**Artifacts:** `models/threshold_sweep.csv`, `models/threshold_summary.json`  
**Selected Rows (from `models/threshold_sweep.csv` & summary):**

| threshold | precision | recall | f1     |
|-----------|-----------|--------|--------|
| 0.50      | 0.5613    | 0.7011 | 0.6234 |
| 0.52      | 0.5757    | 0.6904 | 0.6278 |

**Decision:** For recall-focused retention, keep **t = 0.50** (meets Recall ≥ 0.70). For balanced performance, **t = 0.52** provides the highest F1 on this test set.

---

## (Optional) Experiment 03 — XGBoost Baseline
**Preprocess:** same as Exp01  
**Model:** XGBClassifier (fast hist tree method)  
**Params:** n_estimators=300, max_depth=5, lr=0.08, subsample=0.9, colsample_bytree=0.9  
**Test Metrics:** 🔧 fill  
**Notes:** Compare ROC-AUC and Recall vs. Logistic Regression.

---

## Issues & Fixes
- `Senior Citizen` treated as numeric; contained "Yes/No" → moved to categorical (config fix).
- YAML import error → installed `pyyaml`.
- XLSX ingestion → added `openpyxl`.

## Screenshots
- `reports/figures/roc_curve_baseline.png`
- `reports/figures/confusion_matrix_baseline.png`
- `reports/figures/dashboard_home.png`
- `reports/figures/dashboard_predict.png`
