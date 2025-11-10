import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

def load_split(name):
    X = pd.read_csv(DATA_DIR / f"X_{name}.csv")
    y = pd.read_csv(DATA_DIR / f"y_{name}.csv")["churn"].to_numpy()
    return X, y

def main():
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    # fast, strong baseline; handles imbalance with class_weight
    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        n_jobs=None
    )
    clf.fit(X_train, y_train)

    # evaluate on test
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    # save artifacts
    joblib.dump(clf, MODELS_DIR / "baseline_logreg.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # figures: ROC + confusion matrix
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC Curve — Logistic Regression (baseline)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve_baseline.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix — Logistic Regression (baseline)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_baseline.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model → {MODELS_DIR / 'baseline_logreg.joblib'}")
    print(f"Saved metrics → {MODELS_DIR / 'metrics.json'}")
    print(f"Saved figures → {REPORTS_DIR}")

if __name__ == "__main__":
    main()