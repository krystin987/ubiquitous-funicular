# scripts/preprocess.py
from pathlib import Path
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

CFG = Path("configs/config.yaml")
DATA_DIR = Path("data/processed")

def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_cfg():
    with open(CFG, "r") as f:
        return yaml.safe_load(f)

def load_split(name: str) -> pd.DataFrame:
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing split file: {p}")
    return pd.read_csv(p)

def build_pipeline(numeric_feats, categorical_feats):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_feats),
            ("cat", cat_pipe, categorical_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def to_xy(df: pd.DataFrame, target: str):
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def main():
    cfg = load_cfg()
    target = cfg["target"]
    num = cfg["numeric_features"]
    cat = cfg["categorical_features"]

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    X_train, y_train = to_xy(train, target)
    X_val, y_val = to_xy(val, target)
    X_test, y_test = to_xy(test, target)

    pre = build_pipeline(num, cat)
    # ensure numeric cols are actually numeric
    X_train = coerce_numeric(X_train, num)
    X_val = coerce_numeric(X_val, num)
    X_test = coerce_numeric(X_test, num)

    pre.fit(X_train)

    # transform
    Xt_train = pd.DataFrame(pre.transform(X_train), columns=pre.get_feature_names_out())
    Xt_val   = pd.DataFrame(pre.transform(X_val),   columns=pre.get_feature_names_out())
    Xt_test  = pd.DataFrame(pre.transform(X_test),  columns=pre.get_feature_names_out())

    out = DATA_DIR
    Xt_train.to_csv(out / "X_train.csv", index=False)
    Xt_val.to_csv(out / "X_val.csv", index=False)
    Xt_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False, header=["churn"])
    y_val.to_csv(out / "y_val.csv", index=False, header=["churn"])
    y_test.to_csv(out / "y_test.csv", index=False, header=["churn"])

    print("Wrote:")
    for f in ["X_train.csv","y_train.csv","X_val.csv","y_val.csv","X_test.csv","y_test.csv"]:
        print(" -", (out / f).resolve())



if __name__ == "__main__":
    main()