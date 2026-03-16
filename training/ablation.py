
"""
"Ablation testing" = train the same model multiple times using different *subsets of features
- the results are 'too' good so we are trying to see if there is one specific data column that fits the overall model exactly
 
Docs
- ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- average_precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Paths + constants
# -----------------------------
DATA_DIR = Path("data/processed")
ART_SCHEMAS = Path("artifacts/schemas")

LABEL_COL = "target"

# These are your *known* categorical columns (strings/categories).
# We will automatically intersect these with each feature subset.
CATEGORICAL_CANON = ["Highest Layer", "Transport Layer"]


# -----------------------------
# Helper: build model pipeline for a given feature subset
# -----------------------------
def build_model(feature_subset: list[str]) -> Pipeline:
    """
    Build a scikit-learn Pipeline = (preprocess -> logistic regression)
    that only uses the columns in feature_subset.

    Why we do this:
    - Different columns need different preprocessing:
      * categorical -> OneHotEncode
      * numeric -> scale
    - ColumnTransformer routes the correct columns to the correct preprocessing.
    - Pipeline ensures preprocessing is fit on TRAIN only (reduces leakage).
    """

    # Decide which subset columns are categorical vs numeric
    categorical_cols = [c for c in feature_subset if c in CATEGORICAL_CANON]
    numeric_cols = [c for c in feature_subset if c not in categorical_cols]

    transformers = []

    # Numeric preprocessing (if we have numeric columns in this subset)
    if len(numeric_cols) > 0:
        num_pipe = Pipeline(steps=[
            # Fill missing numeric values (median is robust to outliers)
            ("imputer", SimpleImputer(strategy="median")),
            # Scale features so LR optimization converges more reliably
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    # Categorical preprocessing (if we have categorical columns in this subset)
    if len(categorical_cols) > 0:
        cat_pipe = Pipeline(steps=[
            # Fill missing categories with the mode
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encode categories; ignore unseen categories at val/test time
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    # ColumnTransformer expects a list of (name, transformer, columns) tuples.
    # It applies each transformer to its column subset, then concatenates results.
    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drop any columns not explicitly listed
    )

    # Logistic regression baseline classifier
    clf = LogisticRegression(max_iter=500)

    # Full pipeline: preprocessing then classifier
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    return model


# -----------------------------
# Helper: pick threshold by targeting FPR on validation
# -----------------------------
def threshold_for_target_fpr(y_val: np.ndarray, val_scores: np.ndarray, target_fpr: float) -> float:
    """
    Choose a threshold using ONLY benign validation scores so that the expected
    false-positive-rate (FPR) is about target_fpr.

    Logic:
    - FPR = fraction of benign samples predicted as attack.
    - If we set threshold to the (1 - target_fpr) quantile of benign scores,
      then approximately target_fpr of benign scores will lie above it.

    Example: target_fpr = 0.01
    - Choose threshold as the 99th percentile of benign scores.
    - About 1% of benign validation examples will exceed threshold -> flagged as attack.
    """
    benign_scores = val_scores[y_val == 0]
    if benign_scores.size == 0:
        # If there are no benign samples (shouldn't happen in your current splits),
        # fallback to a typical default.
        return 0.5

    thr = float(np.quantile(benign_scores, 1.0 - target_fpr))
    return thr


# -----------------------------
# Helper: compute common metrics from confusion matrix
# -----------------------------
def summarize_from_cm(tn: int, fp: int, fn: int, tp: int) -> dict:
    """
    Convert confusion matrix components into useful rates:
    - precision: among predicted attacks, how many are true attacks?
    - recall/TPR: among true attacks, how many did we catch?
    - FPR: among true benign, how many did we falsely flag?
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # aka TPR
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": float(precision),
        "recall_tpr": float(recall),
        "fpr": float(fpr),
    }


# -----------------------------
# Main ablation runner
# -----------------------------
def main():
    ART_SCHEMAS.mkdir(parents=True, exist_ok=True)

    # Load the official feature order from features.json created by make_dataset.py
    feature_cols = json.loads((ART_SCHEMAS / "features.json").read_text())

    # Load your processed splits (these already contain only chosen features + target)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Split each DF into X and y
    X_train, y_train = train_df[feature_cols], train_df[LABEL_COL].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[LABEL_COL].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[LABEL_COL].to_numpy()

    # Define feature subsets for ablation
    packets_only = [c for c in ["Packets/Time"] if c in feature_cols]
    ports_only = [c for c in ["Source Port", "Dest Port"] if c in feature_cols]
    proto_only = [c for c in ["Highest Layer", "Transport Layer"] if c in feature_cols]
    numeric_only = [c for c in ["Source Port", "Dest Port", "Packet Length", "Packets/Time"] if c in feature_cols]

    experiments = [
        ("full", feature_cols),
        ("packets_only", packets_only),
        ("ports_only", ports_only),
        ("proto_only", proto_only),
        ("numeric_only", numeric_only),
    ]

    # Choose the FPR you want to target when picking thresholds
    target_fpr = 0.01

    results = []

    print("\n=== Ablation experiments (Logistic Regression) ===")
    print(f"Target FPR (chosen on validation benign): {target_fpr}\n")

    for name, subset in experiments:
        # Skip empty subsets (just in case a column name changed)
        if len(subset) == 0:
            print(f"[{name}] SKIP (no columns found)")
            continue

        # Build model for this subset
        model = build_model(subset)

        # Train
        model.fit(X_train[subset], y_train)

        # Validate: get probability-like scores for class 1 (attack)
        val_scores = model.predict_proba(X_val[subset])[:, 1]

        # Ranking metrics on validation (threshold-free)
        val_ap = float(average_precision_score(y_val, val_scores))
        val_auc = float(roc_auc_score(y_val, val_scores))

        # Pick threshold to hit target FPR on validation
        thr = threshold_for_target_fpr(y_val, val_scores, target_fpr=target_fpr)

        # Test: apply threshold once (no tuning on test)
        test_scores = model.predict_proba(X_test[subset])[:, 1]
        test_preds = (test_scores >= thr).astype(int)

        # Confusion matrix layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()

        rates = summarize_from_cm(int(tn), int(fp), int(fn), int(tp))

        row = {
            "name": name,
            "features": subset,
            "val_ap": val_ap,
            "val_roc_auc": val_auc,
            "threshold": float(thr),
            "test_cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            **rates,
        }
        results.append(row)

        # Print a compact line for quick comparison
        print(
            f"[{name:12}] "
            f"VAL(AP={val_ap:.6f}, AUC={val_auc:.6f}) "
            f"thr={thr:.6f} "
            f"TEST(prec={rates['precision']:.4f}, rec={rates['recall_tpr']:.4f}, fpr={rates['fpr']:.4f}) "
            f"CM={row['test_cm']}"
        )

    # Save JSON results so you can reference them later / plot them in frontend
    out_path = ART_SCHEMAS / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nSaved -> {out_path}")
    print("Tip: if a tiny subset (like packets_only) performs nearly as well as full,")
    print("that suggests your dataset may be 'too easy' / shortcut-based and may not generalize.\n")


if __name__ == "__main__":
    main()
