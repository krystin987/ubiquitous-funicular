import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to raw CSV (e.g., data/raw/churn.csv)")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Basic dataset report
    print("\n=== DATASET SUMMARY ===")
    print(f"Path: {csv_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nNull counts (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # If a 'churn' column exists, show class balance
    if "churn" in df.columns:
        print("\nTarget 'churn' value counts:")
        print(df["churn"].value_counts(dropna=False))
    else:
        print("\n(no 'churn' column detected yet)")

if __name__ == "__main__":
    main()