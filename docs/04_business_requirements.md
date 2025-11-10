Purpose: Reduce churn via early identification and targeted retention actions.
Scope: Produce churn probabilities and ranked lists; provide a dashboard and CSV/API handoff; no PII or proprietary data in capstone.
Users & Needs:

Marketing/CRM: ranked high-risk customers, threshold controls.

Support Leaders: visibility into drivers (contract, tenure, charges).

Executives: KPIs (recall, ROC-AUC) and uplift framing.
Functional Requirements:

Ingest tabular customer features; output probability [0–1].

Dashboard with 3 visuals (ROC, confusion matrix, class balance).

Interactive single-record prediction.

Export: CSV of scores (future: API).
Non-Functional:

Local execution; open-source stack; run in < 5 minutes on laptop.

Reproducible: seeded splits; pinned requirements.txt.
Success Criteria: ROC-AUC ≥ 0.80; Recall ≥ 0.65 on held-out test; runnable from clean checkout with quick-start.
Risks & Mitigations: Class imbalance → class_weight, threshold sweep. Data quality → imputers, schema checks.
Ethics & Security: No PII; no credentials in repo; fairness checks by segment as time permits.