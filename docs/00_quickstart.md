# Quick-Start — Churn Prediction Data Product

## Prereqs
- Python 3.11+
- macOS/Windows/Linux
- No cloud services required

## Setup
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate
pip install -r requirements.txt

## Data prep
# if starting from Excel:
python scripts/prep_telco_xlsx.py
# otherwise use the canonical CSV:
# data/raw/churn.csv

# split, preprocess, and generate matrices
python src/churn/train.py --input_csv data/raw/churn.csv
python scripts/preprocess.py

## Train baseline & export artifacts
python src/churn/train_baseline.py

Artifacts:
- models/baseline_logreg.joblib
- models/metrics.json
- reports/figures/roc_curve_baseline.png
- reports/figures/confusion_matrix_baseline.png

## Run the dashboard
streamlit run app/streamlit_app.py

## Project structure
src/            # training/eval code
scripts/        # prep & preprocessing
app/            # streamlit dashboard
data/raw/       # telco_churn.csv
data/processed/ # train/val/test + X_/y_ CSVs
models/         # joblib + metrics.json
reports/figures # images for docs
configs/        # config.yaml