#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complet pour la détection de cancer du sein (binaire : B vs M)
- Lit un CSV (format WDBC-like : 'id', 'diagnosis', puis features numériques)
- Prétraitement : nettoyage colonnes, encodage y, standardisation des features
- (Optionnel) Rééquilibrage avec SMOTE
- Sélection de modèle via GridSearchCV :
    * LogisticRegression
    * RandomForest
    * HistGradientBoosting
    * SVM (SVC)
    * XGBoost (XGBClassifier) — si xgboost est installé
- Calibration des probabilités (isotonic/sigmoid)
- Sélection d'un seuil optimal (Youden J ou F1) à partir de prédictions CV
- Évaluation sur test : ROC-AUC, PR-AUC, matrice de confusion, rapport de classification
- Importance des features (Permutation Importance ou coefficients linéaires)
- Exporte le modèle calibré (.joblib) + le seuil (.json)
Usage:
    python breast_cancer_detection_pipeline.py -i data.csv --metric youden --no-smote
"""
import argparse
import json
import os
import sys
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target

# XGBoost (optionnel)
_XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception as _e:
    XGBClassifier = None  # type: ignore

import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier introuvable: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    # Nettoyage des colonnes parasites souvent présentes (ex: 'Unnamed: 32')
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    # Normalisation noms colonnes
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def split_X_y(df: pd.DataFrame, target_col: str = "diagnosis") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        print(f"[ERREUR] Colonne cible '{target_col}' absente du CSV.", file=sys.stderr)
        sys.exit(1)
    y_raw = df[target_col]
    # Encodage: Malignant=1, Benign=0
    if y_raw.dtype == object:
        y = y_raw.map({"M": 1, "B": 0})
    else:
        y = y_raw.astype(int)
    if y.isna().any():
        # Essai d'encodage universel (B/benign -> 0; M/malignant -> 1)
        y = y_raw.astype(str).str[0].str.upper().map({"M": 1, "B": 0})
    if y.isna().any():
        print("[ERREUR] Impossible d'encoder la cible (diagnosis). Attendu 'M'/'B' ou 0/1.", file=sys.stderr)
        sys.exit(1)
    # Drop ID et cible
    drop_cols = [target_col]
    for cand in ["id", "ID", "Id"]:
        if cand in df.columns:
            drop_cols.append(cand)
    X = df.drop(columns=drop_cols, errors="ignore")
    # Contrôle du type
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except Exception:
                pass
    # Remplacer les NaN par la médiane (robuste)
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def make_candidates(random_state: int = 42) -> List[Tuple[str, Pipeline, dict]]:
    """Retourne une liste (name, pipeline, param_grid)."""
    candidates = []

    # Logistic Regression
    pipe_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=5000, solver="saga", penalty="l2", n_jobs=None, random_state=random_state))
    ])
    grid_lr = {
        "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
    }
    candidates.append(("logreg", pipe_lr, grid_lr))

    # Random Forest
    pipe_rf = Pipeline([
        ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    grid_rf = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2]
    }
    candidates.append(("random_forest", pipe_rf, grid_rf))

    # HistGradientBoosting
    pipe_hgb = Pipeline([
        ("clf", HistGradientBoostingClassifier(random_state=random_state))
    ])
    grid_hgb = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [None, 6],
        "clf__l2_regularization": [0.0, 1.0],
        "clf__max_leaf_nodes": [31, 63]
    }
    candidates.append(("hist_gb", pipe_hgb, grid_hgb))

    # SVM (SVC) + StandardScaler
    pipe_svm = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SVC(probability=False, random_state=random_state))  # probas via calibration plus tard
    ])
    grid_svm = {
        "clf__kernel": ["rbf"],
        "clf__C": [0.5, 1.0, 3.0, 10.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.1],
        # "clf__class_weight": [None, "balanced"],  # utile si classes très déséquilibrées
    }
    candidates.append(("svm", pipe_svm, grid_svm))

    # XGBoost (optionnel)
    if _XGB_AVAILABLE:
        pipe_xgb = Pipeline([
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            ))
        ])
        grid_xgb = {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1.0, 3.0, 5.0]
        }
        candidates.append(("xgboost", pipe_xgb, grid_xgb))
    else:
        print("[INFO] xgboost non installé — XGBoost sera ignoré. `pip install xgboost` pour l'activer.", file=sys.stderr)

    return candidates


def optionally_add_smote(pipeline: Pipeline, use_smote: bool) -> Pipeline:
    if not use_smote:
        return pipeline
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        sm = SMOTE(random_state=42, k_neighbors=5)
        # On intercale SMOTE avant le classifieur; scaler avant SMOTE si présent
        steps = pipeline.steps
        new_steps = []
        scaler_idx = None
        for i, (name, step) in enumerate(steps):
            if isinstance(step, StandardScaler):
                scaler_idx = i
        if scaler_idx is not None:
            new_steps = steps[:scaler_idx+1] + [("smote", sm)] + steps[scaler_idx+1:]
        else:
            new_steps = [("smote", sm)] + steps
        return ImbPipeline(new_steps)
    except Exception as e:
        print(f"[INFO] imbalanced-learn non disponible ({e}). Poursuite sans SMOTE.")
        return pipeline


def gridsearch_best_estimator(X: pd.DataFrame, y: pd.Series, use_smote: bool, cv_splits: int, random_state: int):
    best_model = None
    best_score = -np.inf
    best_name = None
    best_cv = None

    for name, pipe, grid in make_candidates(random_state):
        pipe = optionally_add_smote(pipe, use_smote)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, verbose=0, return_train_score=False
        )
        gs.fit(X, y)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_name = name
            best_cv = cv
        print(f"[CV] {name}: ROC-AUC={gs.best_score_:.4f} (params={gs.best_params_})")
    print(f"[SELECTION] Modèle retenu: {best_name} avec ROC-AUC CV={best_score:.4f}")
    return best_name, best_model, best_cv


def calibrate_model(best_estimator, X_train, y_train, method: str = "isotonic", cv: int = 3):
    # Calibre toutes les variantes (y compris SVM qui n'a pas de proba natives)
    calib = CalibratedClassifierCV(estimator=best_estimator, method=method, cv=cv)
    calib.fit(X_train, y_train)
    return calib


def choose_threshold(proba: np.ndarray, y_true: np.ndarray, strategy: str = "youden") -> float:
    # proba: probas de la classe positive (1 = Malignant)
    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, proba)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        return float(thr[best_idx])
    elif strategy == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
        f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
        thr_full = np.concatenate([thresholds, [thresholds[-1]]])
        best_idx = int(np.nanargmax(f1s))
        return float(thr_full[best_idx])
    else:
        raise ValueError("strategy doit être 'youden' ou 'f1'")


def evaluate(y_true: np.ndarray, proba: np.ndarray, thr: float) -> dict:
    y_pred = (proba >= thr).astype(int)
    metrics = {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["Benign(0)", "Malignant(1)"])
    }
    return metrics


def plot_curves(y_true: np.ndarray, proba: np.ndarray, title_prefix: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure()
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150)
    plt.close()


def feature_importance(model, X: pd.DataFrame, y: pd.Series, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    # Permutation importance si possible
    try:
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
        importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        imp_path = os.path.join(outdir, "permutation_importance.csv")
        importances.to_csv(imp_path, header=["importance"])
        print(f"[INFO] Importances (permutation) sauvegardées: {imp_path}")
    except Exception as e:
        print(f"[WARN] Échec permutation importance: {e}")
        # Fallback: coefficients LR si dispo
        try:
            if hasattr(model, "named_steps"):
                clf = model.named_steps.get("clf", model)
            else:
                clf = model
            if hasattr(clf, "coef_"):
                coefs = pd.Series(clf.coef_.ravel(), index=X.columns).sort_values(key=np.abs, ascending=False)
                imp_path = os.path.join(outdir, "linear_coefficients.csv")
                coefs.to_csv(imp_path, header=["coef"])
                print(f"[INFO] Coefficients linéaires sauvegardés: {imp_path}")
        except Exception as e2:
            print(f"[WARN] Pas d'importance dispo: {e2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Chemin vers le CSV (ex: data.csv)")
    parser.add_argument("--smote", dest="smote", action="store_true", help="Activer SMOTE (si imblearn dispo)")
    parser.add_argument("--no-smote", dest="smote", action="store_false", help="Désactiver SMOTE")
    parser.set_defaults(smote=True)
    parser.add_argument("--metric", type=str, default="youden", choices=["youden", "f1"],
                        help="Stratégie de sélection du seuil")
    parser.add_argument("--calibration", type=str, default="isotonic", choices=["isotonic", "sigmoid"],
                        help="Méthode de calibration")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds pour la CV")
    parser.add_argument("--random_state", type=int, default=42, help="Seed")
    parser.add_argument("--outdir", type=str, default="outputs", help="Dossier de sortie")
    args = parser.parse_args()

    df = load_data(args.input)
    X, y = split_X_y(df, target_col="diagnosis")

    # Vérifications rapides
    if type_of_target(y) != "binary":
        print("[ERREUR] La cible n'est pas binaire (attendu 0/1 pour B/M).", file=sys.stderr)
        sys.exit(1)
    print(f"[DATA] Shape: X={X.shape}, y positives (M=1)={int(y.sum())}/{len(y)} ({100*y.mean():.1f}%)")
    print(f"[DATA] Colonnes: {list(X.columns)[:6]} ... (+{max(0, X.shape[1]-6)} de plus)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )

    # Sélection du meilleur estimateur via GridSearch
    name, best_estimator, best_cv = gridsearch_best_estimator(
        X_train, y_train, use_smote=args.smote, cv_splits=args.cv, random_state=args.random_state
    )

    # Calibration
    calibrated = calibrate_model(best_estimator, X_train, y_train, method=args.calibration, cv=3)

    # Seuil optimal à partir de prédictions CV sur train
    cv_proba_train = cross_val_predict(
        calibrated, X_train, y_train, cv=StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state),
        method="predict_proba", n_jobs=-1
    )[:, 1]
    thr = choose_threshold(cv_proba_train, y_train.values, strategy=args.metric)
    print(f"[SEUIL] {args.metric} -> threshold={thr:.4f}")

    # Évaluation test
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    results = evaluate(y_test.values, test_proba, thr)
    print("[METRICS TEST]")
    for k, v in results.items():
        if k == "classification_report" or k == "confusion_matrix":
            continue
        print(f"  - {k}: {v}")
    print("[CONFUSION MATRIX]")
    print(np.array(results["confusion_matrix"]))
    print("[CLASSIFICATION REPORT]")
    print(results["classification_report"])

    # Courbes
    plot_curves(y_test.values, test_proba, title_prefix=f"Test ({name})", outdir=args.outdir)

    # Importances
    feature_importance(calibrated, X_test, y_test, outdir=args.outdir)

    # Sauvegarde modèle + seuil
    model_path = os.path.join(args.outdir, "breast_cancer_model.joblib")
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump({"model": calibrated, "features": list(X.columns)}, model_path)
    thr_path = os.path.join(args.outdir, "threshold.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "strategy": args.metric}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Modèle calibré: {model_path}")
    print(f"[SAVE] Seuil: {thr_path}")


if __name__ == "__main__":
    main()
