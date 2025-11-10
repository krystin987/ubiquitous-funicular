from pathlib import Path
import pandas as pd

SRC = Path("data/raw/telco_churn.xlsx")  # adjust if needed
DST = Path("data/raw/churn.csv")
SHEET = "Telco_Churn"  # change if your sheet is named differently

if not SRC.exists():
    raise FileNotFoundError(f"Missing source Excel file: {SRC}")

# Load the sheet
df = pd.read_excel(SRC, sheet_name=SHEET, engine="openpyxl")

# --- Normalize target ---
if "Churn Value" in df.columns:
    df["churn"] = df["Churn Value"].astype(int)
elif "Churn Label" in df.columns:
    df["churn"] = (df["Churn Label"].astype("string").str.strip().str.lower() == "yes").astype(int)
else:
    raise KeyError("Neither 'Churn Value' nor 'Churn Label' found to build target.")

# --- Coerce numeric quirk ---
if "Total Charges" in df.columns:
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# --- Clean strings / blanks ---
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for col in obj_cols:
    df[col] = df[col].astype("string").str.strip()
df.replace({"": pd.NA, " ": pd.NA}, inplace=True)

# --- Drop clear IDs / leakage if present ---
drop_cols = [
    "CustomerID",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Write canonical CSV
DST.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DST, index=False)
print(f"Wrote {DST.resolve()} with 'churn' 0/1 and cleaned columns.")
