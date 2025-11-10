from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS = Path("models")
FIGS = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MODELS / "threshold_sweep.csv")

plt.figure()
plt.plot(df["threshold"], df["precision"], label="precision")
plt.plot(df["threshold"], df["recall"], label="recall")
plt.plot(df["threshold"], df["f1"], label="f1")
plt.xlabel("threshold")
plt.ylabel("score")
plt.title("Threshold Sweep — Precision / Recall / F1")
plt.legend()
plt.tight_layout()
out = FIGS / "threshold_sweep.png"
plt.savefig(out, dpi=150)
print(f"Saved {out.resolve()}")