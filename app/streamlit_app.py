import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")
DATA_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("Churn Prediction Dashboard")

# --- Load artifacts ---
model = joblib.load(MODELS_DIR / "baseline_logreg.joblib")
with open(MODELS_DIR / "metrics.json") as f:
    metrics = json.load(f)

st.subheader("Model Metrics (Test Set)")
cols = st.columns(5)
for c, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
    c.metric(k.upper(), f"{metrics[k]:.3f}")

st.divider()

# --- Visuals (3 types) ---
st.subheader("Evaluation Figures")
c1, c2 = st.columns(2)
with c1:
    st.image(str(FIG_DIR / "roc_curve_baseline.png"), caption="ROC Curve", use_column_width=True)
with c2:
    st.image(str(FIG_DIR / "confusion_matrix_baseline.png"), caption="Confusion Matrix", use_column_width=True)

# class balance bar from y_test
try:
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["churn"].value_counts(normalize=True).rename("pct")
    st.bar_chart(y_test)
except Exception:
    st.info("y_test not found for class balance chart.")

st.divider()

# --- Interactive Predict ---
st.subheader("Try a Prediction")

# Use columns from X_train to build a minimal form
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
feature_names = list(X_train.columns)

st.caption("Enter feature values (for one-hot columns, set 1 to include that category; otherwise leave 0).")
values = {}
for name in feature_names:
    # simple heuristic: binary features get a checkbox, others numeric
    if set(X_train[name].unique()).issubset({0.0, 1.0}):
        values[name] = st.checkbox(name, value=False)
    else:
        values[name] = st.number_input(name, value=float(X_train[name].mean()), step=0.1, format="%.3f")

# Assemble row in the exact column order
row = np.array([[float(values[n]) for n in feature_names]])
if st.button("Predict churn probability"):
    prob = model.predict_proba(row)[0, 1]
    st.success(f"Churn Probability: {prob:.3f}")
    st.progress(min(max(prob, 0.0), 1.0))