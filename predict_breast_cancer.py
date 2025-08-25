#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for the breast cancer model.
- Loads calibrated model + features from joblib
- Loads the chosen threshold from threshold.json
- Accepts a CSV of samples (with or without 'diagnosis')
- Reorders/filters columns to match training features
- Outputs a CSV with probabilities and predicted labels (B/M)
Usage:
    python predict_breast_cancer.py -m outputs/breast_cancer_model.joblib -t outputs/threshold.json -i new_samples.csv -o predictions.csv
"""
import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

def load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    bundle = joblib.load(model_path)
    model = bundle.get("model", None)
    features = bundle.get("features", None)
    if model is None or features is None:
        print(f"[ERROR] Invalid model bundle (missing 'model' or 'features').", file=sys.stderr)
        sys.exit(1)
    return model, features

def load_threshold(thr_path: str) -> float:
    if not os.path.exists(thr_path):
        print(f"[ERROR] Threshold file not found: {thr_path}", file=sys.stderr)
        sys.exit(1)
    with open(thr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thr = float(data.get("threshold", 0.5))
    return thr

def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed garbage cols
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    # Standardize col names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to joblib model bundle")
    ap.add_argument("-t", "--threshold", required=True, help="Path to threshold.json")
    ap.add_argument("-i", "--input", required=True, help="CSV with samples to score")
    ap.add_argument("-o", "--output", required=True, help="Where to write predictions CSV")
    args = ap.parse_args()

    model, features = load_model(args.model)
    thr = load_threshold(args.threshold)

    df = pd.read_csv(args.input)
    df = sanitize_frame(df)

    # Keep id if present for nicer output
    id_col = None
    for cand in ["id", "ID", "Id"]:
        if cand in df.columns:
            id_col = cand
            break

    # Remove target if present
    for t in ["diagnosis", "target", "label"]:
        if t in df.columns:
            df = df.drop(columns=[t])

    # Coerce numeric, fill missing
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    # Align to training features
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in input: {missing}")
        # Create them as NaN -> fill with column medians (0 as safe fallback)
        for c in missing:
            df[c] = np.nan
        df = df.fillna(0.0)
    extra = [c for c in df.columns if c not in features and c != id_col]
    if extra:
        print(f"[INFO] Ignoring extra columns: {extra}")

    X = df[features]

    # Predict calibrated probabilities for malignant=1
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)  # 1=M, 0=B
    label = np.where(pred == 1, "M", "B")

    out = pd.DataFrame({
        "proba_malignant": proba,
        "pred_binary": pred,
        "pred_label": label
    })
    if id_col is not None:
        out.insert(0, id_col, df[id_col].values)

    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[SAVE] Predictions -> {args.output}")
    print(f"[INFO] Using threshold={thr:.4f}; label 'M' if proba >= thr, else 'B'.")

if __name__ == "__main__":
    main()
