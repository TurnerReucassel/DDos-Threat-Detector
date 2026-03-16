""" Goal:
- Load the already-prepared train/val/test CSV splits created by training/make_dataset.py
- Build a scikit-learn preprocessing + Logistic Regression model pipeline
- Train on train split
- Validate on val split (compute metrics + choose a decision threshold)
- Evaluate once on test split using the chosen threshold
- Save artifacts for later use by the backend API:
- artifacts/models/model.joblib
- artifacts/schemas/metrics.json
- artifacts/schemas/threshold.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import \
    ColumnTransformer  # docs: (name, transformer, columns)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data/processed") #where the split CSVs live (produced by make_dataset.py)

ART_MODELS = Path("artifacts/models") #where we will save the trained model artifact (binary)
ART_SCHEMAS = Path("artifacts/schemas") #Where we will save schemas/metrics/threshold as JSON
LABEL_COL = "target" #

# use one-hot encoding for the highest layer (first col) and transport layer (2nd col)

categorical_cols = ["Highest Layer","Transport Layer"]

numerical_cols = ["Source Port","Dest Port", "Packet Length", "Packets/Time"] #not including IPs right now

#Helper: choose threshold
def pick_threshold(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:  
    #inputs: y_true:  true labels (0/1) for validation
    #scores: predicted "attack probability" (DDos attack) scores in [0,1] from predict_proba
    #target_fpr: desired maximum false positives rate (fpr) on validation
    #output: threshold : a float such that if (score>threshold) then predict attack (1)
    """ What this is doing:
    - For each possible threshold value, compute the confusion matrix
    - Compute FPR = FP / (FP + TN)
    - Choose the *highest* threshold that keeps FPR <= target_fpr
    (higher threshold generally reduces false positives)
    """
    thresholds = np.unique(scores)
    best_t = 0.5 #fallback/default threshold if nothing matches fpr constraint
    best_recall = -1.0

    for t in thresholds:
        preds = (scores>=t).astype(int) #convert probability scores -> 0 or 1 predictions
        #confusion_matrix returns 2x2 matrix: 
        #[[TN, FP], true negative, false positive, ... etc
        # [FN,TP]]
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel() #ravel takes an array (usually 2D) and "flattens" it into 1D array; e.g: [[TN, FP], [FN, TP]] becomes [TN, FP, FN, TP]

        #calculate false positive rate ( how many benign (0), did we wrongly mark as 1 (attacks) )
        fpr = fp/(fp + tn) if(fp+tn) else 0.0
        recall = tp / (tp+fn) if(tp+fn) else 0.0 # recall = true positive rate: out of all the attacks, what proportion did we correctly predict?

        #if this is less than our fpr threshold, we accept it; keep updating so we end up with largest threshold that still reaches the target
        if fpr<=target_fpr and recall > best_recall:
            best_recall = recall
            best_t = float(t)

    return best_t
        


#main training routine
def main(): 
    #ensure output folders exist:
    ART_MODELS.mkdir(parents=True, exist_ok = True) # parents=True: create any missing parent directories
    ART_SCHEMAS.mkdir(parents=True, exist_ok=True)

    # Load the ordered list of feature column names created in make_dataset.py.
    # We save this so training and later inference use the exact same feature schema/order.
    feature_cols = json.loads((ART_SCHEMAS / "features.json").read_text())

    #load the split (split in make_datasets.py) datasets
    train_df = pd.read_csv(DATA_DIR/"train.csv") # the / is the path-joining operator: DATA_DIR = data/processed so this makes it data/processed/train.csv
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

     # separate inputs (X) from labels (y) for each split.
    # X_* is a dataFrame of feature columns.
    # y_* is a series of 0/1 target labels.
    X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
    X_val, y_val = val_df[feature_cols], val_df[LABEL_COL]
    X_test, y_test = test_df[feature_cols], test_df[LABEL_COL]

    #start the pipeline with encoding:
    numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), #if there are missing values, fill them in with the column median
    ('scalar', StandardScaler()), #scales numeric features, ** helps LR behave better when features have different scales
    ])

    categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "most_frequent")),
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ])

    #combine these pipelines together
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_pipeline, numerical_cols), # take numerical_col, run numerical_pipeline on it
            ('cat', categorical_pipeline, categorical_cols),
        ],
        remainder = "drop",
    )
    """ 
    transformers= and steps= are just named arguments that tell scikit-learn what list you’re passing in.
    transformers= is the list of (name, transformer, columns) rules for ColumnTransformer.
    steps= is the list of (name, transformer/estimator) stages for Pipeline. 
    """
    #final pipeline: preprocess --> logstic regression
    model = Pipeline(
        steps = [
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=500)), #max iterations allowed to do before it stops: use 500 instead of the default 100 because we are using one-hot encoding
    ])

    #train
    #this fits the preprocessing transformers on X_train, transforms X_train, then fits the classifier using the transformed data and y_train
    model.fit(X_train, y_train)

    #validate? choose thresholds?

    #predict_proba returns two columns: - [:,0] probability of class 0 (no attack), and [:1] probability of class 1 (ddos attack)
    val_scores = model.predict_proba(X_val)[:, 1]

    #one debug print:
    print("VAL score range:", float(val_scores.min()), float(val_scores.max())) #check how confident the model can get
    print("VAL mean score (benign):", float(val_scores[y_val==0].mean())) #the average predicted value for a non-attack row
    print("VAL mean score (attack):", float(val_scores[y_val==1].mean())) #average pridcited value for an attack row
    
    #check our average precision
    ap = average_precision_score(y_val, val_scores)
    # ROC-AUC too to evaluate precision of the model
    auc = roc_auc_score(y_val, val_scores)

    thr = pick_threshold(y_val.to_numpy(), val_scores, target_fpr=0.01) # targeting low false positive rate

    #test once (no tuning) 
    test_scores = model.predict_proba(X_test)[:, 1]
    test_preds = (test_scores >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()

    #save artifacts:
    #save evaluation outputs (JSON) so we can track runs and display results later
    metrics = {
        "val_average_precision": float(ap),
        "val_roc_auc": float(auc),
        "threshold": float(thr),
        "target_fpr": 0.01,
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    (ART_SCHEMAS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART_SCHEMAS / "threshold.json").write_text(json.dumps({"threshold": float(thr)}, indent=2)) #no idea what these two lines do

    #save the full pipeline (preprocess + clf) so inference uses same preprocessing
    joblib.dump(model, ART_MODELS / "model.joblib")

    #print a small summary so you can quickly see that training worked.
    print("Saved model -> artifacts/models/model.joblib")
    print("Saved metrics -> artifacts/schemas/metrics.json")
    print("Saved threshold -> artifacts/schemas/threshold.json")
    print("VAL AP:", ap, "VAL ROC-AUC:", auc, "thr:", thr)
    print("TEST CM:", {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})


if __name__ == "__main__":
    main()