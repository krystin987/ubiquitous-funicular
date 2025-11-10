# Customer Churn Prediction — Capstone (WGU C964)

A reproducible, end-to-end data product that predicts customer churn and supports retention decision-making with an interactive dashboard.

- **Author:** Krystin Villeneuve  
- **Stack:** Python 3.11, pandas, scikit-learn, matplotlib, Streamlit  
- **Status:** Baseline model + dashboard complete (Logistic Regression)  
- **Key Metrics (Test):** ROC-AUC **0.830**, Recall **0.701**, Precision **0.561**, F1 **0.623**, Accuracy **0.775**

---

## ✨ What’s inside

- **Data prep:** Telco churn XLSX → clean CSV; stratified train/val/test splits
- **Preprocessing:** Median impute (numeric) + most-frequent (categorical) + one-hot encoding
- **Modeling:** Logistic Regression (class-weighted) baseline
- **Evaluation:** Metrics JSON, ROC curve, confusion matrix, threshold sweep
- **Dashboard:** Streamlit app with 3 visuals + interactive “predict” form
- **Docs:** Quick-start, business requirements, testing log, hypotheses, accuracy

---

## 🚀 Quick Start

### 1) Environment
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate
pip install -r requirements.txt