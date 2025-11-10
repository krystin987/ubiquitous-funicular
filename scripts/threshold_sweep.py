from pathlib import Path
import json, numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

DATA = Path("data/processed"); MODELS = Path("models")
X_test = pd.read_csv(DATA/"X_test.csv")
y_test = pd.read_csv(DATA/"y_test.csv")["churn"].to_numpy()
clf = joblib.load(MODELS/"baseline_logreg.joblib")

proba = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)

rows = []
for t in np.round(np.arange(0.30, 0.71, 0.01), 2):
  preds = (proba >= t).astype(int)
  p,r,f1,_ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
  rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)})
df = pd.DataFrame(rows)
df.to_csv(MODELS/"threshold_sweep.csv", index=False)
with open(MODELS/"threshold_summary.json","w") as f:
  json.dump({"roc_auc": float(auc),
             "best_f1": df.loc[df.f1.idxmax()].to_dict(),
             "best_recall_at_0_5": float(df.loc[df.threshold==0.50,"recall"].iloc[0])}, f, indent=2)
print("wrote models/threshold_sweep.csv and models/threshold_summary.json")