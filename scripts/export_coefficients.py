# scripts/export_coeffs.py
import joblib, pandas as pd
from pathlib import Path
MODELS=Path("models"); DATA=Path("data/processed")
clf = joblib.load(MODELS/"baseline_logreg.joblib")
cols = pd.read_csv(DATA/"X_train.csv").columns
coefs = pd.Series(clf.coef_.ravel(), index=cols).sort_values(key=abs, ascending=False)
coefs.to_csv(MODELS/"coefficients_sorted.csv", header=["coef"])
print("wrote models/coefficients_sorted.csv")