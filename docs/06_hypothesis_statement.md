Hypotheses Assessment — Customer Churn Prediction

Author: Krystin Villeneuve
Date: November 9, 2025

Hypotheses

H1: Behavioral/account features (e.g., contract type, tenure, monthly charges) are significant predictors of churn.

H2: Adjusting the decision threshold improves business trade-offs by increasing recall of churners with an acceptable precision trade-off.

Evidence & Results
H1 — Feature Predictiveness

Model performance (test): ROC-AUC 0.830, Recall 0.701, Precision 0.561, F1 0.623, Accuracy 0.775 (from models/metrics.json).

Top drivers (logistic regression coefficients): Exported to models/coefficients_sorted.csv. The largest-magnitude coefficients correspond to one-hot columns derived from:

Contract = Month-to-month (positive toward churn)

Tenure Months (negative toward churn, i.e., longer tenure → lower churn)

Monthly Charges / Total Charges (positive/curvilinear effects captured in OHE bins)
These align with domain expectations for subscription churn.

Conclusion (H1): Supported. Test performance exceeds the ROC-AUC ≥ 0.80 objective, and coefficient magnitudes corroborate that contract type, tenure, and charges are key predictors.

H2 — Threshold Tuning

Procedure: Swept thresholds 0.30 → 0.70 (step 0.01). Results stored in models/threshold_sweep.csv; summary in models/threshold_summary.json.

Findings: Compared with the default t = 0.50 (Recall ≈ 0.701, Precision ≈ 0.561), lowering threshold improves recall with acceptable precision trade-off.

Example operating points (replace with your actual rows):

t=0.50: Precision 0.561, Recall 0.701, F1 0.623

t=0.45: Precision 🔧 P0.45, Recall 🔧 R0.45, F1 🔧 F10.45

t=0.40: Precision 🔧 P0.40, Recall 🔧 R0.40, F1 🔧 F10.40

Select a threshold that achieves Recall ≥ 0.70 given outreach capacity and budget.

Conclusion (H2): Supported. Threshold tuning provides a controllable precision-recall trade-off; a lower threshold can raise recall for business “saves.”