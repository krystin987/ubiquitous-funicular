Accuracy & Model Quality Assessment — Customer Churn Prediction

Author: Krystin Villeneuve
Date: November 9, 2025

Test-Set Metrics (Held-Out)

ROC-AUC: 0.830

Recall: 0.701

Precision: 0.561

F1: 0.623

Accuracy: 0.775
(Source: models/metrics.json from train_baseline.py.)

Interpretation:

ROC-AUC 0.83 indicates good separability between churners and non-churners.

Recall 0.70 means the model correctly identifies ~70% of actual churners — appropriate for a retention-first use case.

Precision 0.56 suggests some non-churners will be flagged; this is manageable via business thresholding and campaign capacity planning.

Confusion Matrix & ROC

Artifacts:

reports/figures/confusion_matrix_baseline.png

reports/figures/roc_curve_baseline.png

Notes: Confusion matrix shows the false-negative rate is acceptable at t=0.50; ROC curve supports operating at slightly lower thresholds to increase recall when needed.

Calibration & Thresholding
A sweep of thresholds from 0.30–0.70 shows best F1 at t = 0.52 (Precision 0.576, Recall 0.690, F1 0.628), while the default t = 0.50 yields Recall 0.701, Precision 0.561, F1 0.623. For a retention-first strategy targeting Recall ≥ 0.70, we recommend t = 0.50. If the objective shifts toward a more balanced F1, consider t = 0.52.

Threshold sweep: Results in models/threshold_sweep.csv.

Recommendation: Choose threshold by business objective:

Recall-targeted (saves): lower threshold (e.g., 0.45–0.48).

Precision-targeted (limited budget): higher threshold (e.g., 0.52–0.58).

(Optional) Add probability calibration (Platt/Isotonic) if deploying beyond demo.

Data Quality & Leakage Controls

Imputation: Median for numeric (e.g., 11 nulls in Total Charges), most-frequent for categoricals.

Encoding: Train-only one-hot encoding with handle_unknown="ignore" for robust transforms.

Leakage mitigation: Dropped explicit leakage columns (Churn Score, Churn Reason, duplicates of target).

Splits: Stratified and seeded (70/15/15).

Risks & Mitigations

Class imbalance: Mitigated with class_weight="balanced" and threshold sweep.

Overfitting risk: Controlled by simple baseline; future work may cross-validate or add regularization tuning.

Data drift: (Future) Monitor score distributions and metric deltas over time once operational.

Conclusion: The model meets the stated targets (ROC-AUC ≥ 0.80; Recall ≥ 0.65) and provides a controllable decision surface to support retention actions.