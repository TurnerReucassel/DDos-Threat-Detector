# making a binary dataset: (DDos vs. normal); using a scikit-learn logistic regression model, 
#what metrics should i use?, what is our operational target?, what is our decision threashhold?


import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/DDoS_dataset.csv")
OUT_DIR = Path("data/processed")
SCHEMA_DIR = Path("artifacts/schemas")

LABEL_COL = "target" # the answer (0 or 1 for non-ddos and ddos, respectively)

# columns that are likely identifiers / leakage in many datasets
DROP_COLS = ["Source IP", "Dest IP"]  # start strict for v1

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    # Basic cleanup
    df = df.dropna(subset=[LABEL_COL])

    # Drop ID/leakage columns for baseline
    clean_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    feature_cols = [c for c in clean_df.columns if c != LABEL_COL]

    # Save the feature schema (order matters later for scoring)
    (SCHEMA_DIR / "features.json").write_text(json.dumps(feature_cols, indent=2))

    X = clean_df[feature_cols]
    y = clean_df[LABEL_COL]

    # Split train/val/test (stratified keeps class balance)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    X_train.assign(**{LABEL_COL: y_train}).to_csv(OUT_DIR / "train.csv", index=False)
    X_val.assign(**{LABEL_COL: y_val}).to_csv(OUT_DIR / "val.csv", index=False)
    X_test.assign(**{LABEL_COL: y_test}).to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved splits to", OUT_DIR)
    print("Saved schema to", SCHEMA_DIR / "features.json")
    print("Features:", feature_cols)
    print("Rows:", len(df), "Train/Val/Test:", len(X_train), len(X_val), len(X_test))

if __name__ == "__main__":
    main()
