import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def stratified_split(df: pd.DataFrame, target: str, train_frac: float, val_frac: float, test_frac: float, seed: int):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, "splits must sum to 1"
    rng = np.random.default_rng(seed)
    parts = []
    for y_val, g in df.groupby(target):
        # Shuffle indices per class
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # ensure all go somewhere
        n_test = n - n_train - n_val
        parts.append((
            g.loc[idx[:n_train]],              # train
            g.loc[idx[n_train:n_train+n_val]], # val
            g.loc[idx[n_train+n_val:]]         # test
        ))
    # concat each partition across classes
    train = pd.concat([p[0] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val   = pd.concat([p[1] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test  = pd.concat([p[2] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input_csv", required=True, help="Path to raw CSV (e.g., data/raw/churn.csv)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    target = cfg["target"]
    splits = cfg["splits"]
    drop_cols = cfg.get("drop_columns", [])

    # load
    df = pd.read_csv(args.input_csv)

    # drop optional columns if present
    to_drop = [c for c in drop_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # sanity: target present?
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in columns: {df.columns.tolist()}")

    # quick report
    print("\n=== RAW DATA REPORT ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("Target distribution:")
    print(df[target].value_counts(normalize=True).rename("pct").round(3))

    # split (stratified)
    train, val, test = stratified_split(
        df, target,
        train_frac=splits["train_frac"],
        val_frac=splits["val_frac"],
        test_frac=splits["test_frac"],
        seed=cfg["random_seed"]
    )

    # write out
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    # quick summaries
    def summary(name, d):
        print(f"\n{name}: {d.shape[0]} rows")
        print(d[target].value_counts(normalize=True).rename("pct").round(3))

    summary("TRAIN", train)
    summary("VAL", val)
    summary("TEST", test)
    print(f"\nWrote processed splits to {out_dir.resolve()}")

if __name__ == "__main__":
    main()