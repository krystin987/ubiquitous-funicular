# Testing & Optimization Log

## Environment
- Python: 3.11.x
- OS: (fill)
- Commit: (git sha)

## Data Integrity
- Loaded rows/cols: 7043 x 28
- Nulls handled: Total Charges (11) → median impute
- Target distribution: 0: ~0.735, 1: ~0.265

## Experiments
### Exp 01 — Baseline Logistic Regression
- Preprocess: median numeric, OHE categorical (train-only fit)
- Threshold: 0.50
- Test Metrics:
  - Accuracy: 0.775
  - Precision: 0.561
  - Recall: 0.701
  - F1: 0.623
  - ROC-AUC: 0.830
- Notes: Good recall; refine threshold next.

### Exp 02 — Threshold Sweep
- Goal: improve F1 at target recall ≥ 0.70
- Results:
  - t=0.45 → P=?, R=?, F1=?
  - t=0.40 → P=?, R=?, F1=?
- Selected: t=? (why)

### Exp 03 — (Optional) XGBoost
- Params: (fill)
- Results: (fill)
- Comparison vs. baseline: (fill)

## Issues & Fixes
- Senior Citizen numeric/categorical mismatch → moved to categorical in config.
- Any others: (fill)

## Screenshots
- Attach ROC, confusion matrix, dashboard views