#!/usr/bin/env python3
# Minimal training to keep CI fast: small grid, LR/HGB only.
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from breast_cancer_detection_pipeline import (
    load_data, split_X_y, calibrate_model, choose_threshold, evaluate, plot_curves
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", default="outputs_ci")
    ap.add_argument("--cv", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = split_X_y(load_data(args.input), target_col="diagnosis")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    candidates = [
        ("logreg", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": [0.3, 1.0]}),
        ("hist_gb", Pipeline([("clf", HistGradientBoostingClassifier(random_state=42))]),
            {"clf__learning_rate": [0.1], "clf__max_depth": [None, 6]}),
    ]

    best_auc, best = -1.0, None
    for name, pipe, grid in candidates:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        if gs.best_score_ > best_auc:
            best_auc, best = gs.best_score_, (name, gs.best_estimator_)
        print(f"[CV] {name}: AUC={gs.best_score_:.3f}, params={gs.best_params_}")

    name, est = best
    print(f"[SELECTION] {name} (AUC={best_auc:.3f})")

    calibrated = calibrate_model(est, Xtr, ytr, method="sigmoid", cv=3)

    from sklearn.model_selection import cross_val_predict
    proba_cv = cross_val_predict(calibrated, Xtr, ytr, cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42), method="predict_proba")[:,1]
    thr = choose_threshold(proba_cv, ytr.values, strategy="youden")

    proba_te = calibrated.predict_proba(Xte)[:,1]
    res = evaluate(yte.values, proba_te, thr)

    # Save curves & metrics
    plot_curves(yte.values, proba_te, f"CI Test ({name})", args.outdir)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    import joblib
    joblib.dump({"model": calibrated, "features": list(X.columns)}, os.path.join(args.outdir, "model_ci.joblib"))

    with open(os.path.join(args.outdir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(thr), "strategy": "youden"}, f, indent=2)

    print("[DONE] CI training done.")

if __name__ == "__main__":
    main()
