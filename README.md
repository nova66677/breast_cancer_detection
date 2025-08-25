# Breast Cancer Detection Pipeline (Binary: Benign vs Malignant)

This folder contains a ready-to-run pipeline to train and evaluate a binary classifier for breast-cancer detection from tabular features (WDBC-like dataset). It also includes an inference script to score new samples.

## Files

- `breast_cancer_detection_pipeline.py` — end‑to‑end training pipeline with model selection (LR, RF, HistGB), probability calibration, optimal threshold selection (Youden or F1), metrics, curves, and feature importance export.
- `predict_breast_cancer.py` — loads the saved calibrated model + threshold and scores new samples (CSV), producing probabilities and predicted labels.
- `requirements.txt` — minimal Python dependencies to run the pipeline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Data Format

Your CSV should include the column `diagnosis` with values **M** (malignant) or **B** (benign), plus numeric feature columns. An `id` column is optional (kept for readability, ignored during training). Extra `Unnamed:*` columns are auto‑dropped.

Example header (WDBC-style):
```
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,...
```

## Train

```bash
python breast_cancer_detection_pipeline.py -i path/to/your_dataset.csv --metric youden --smote --cv 5 --outdir outputs
```

**Notes**
- Use `--metric f1` if you prefer optimizing F1 for the threshold.
- Use `--no-smote` to disable SMOTE (if your classes are balanced).
- Artifacts saved to `outputs/`:
  - `breast_cancer_model.joblib` — calibrated model + feature names
  - `threshold.json` — chosen threshold and strategy
  - `roc_curve.png`, `pr_curve.png` — evaluation curves (test set)
  - `permutation_importance.csv` or `linear_coefficients.csv` — feature importance

## Inference on New Samples

Prepare a CSV with the same feature columns as training (the script will align/order them automatically; extra columns will be ignored). Then:

```bash
python predict_breast_cancer.py \
  -m outputs/breast_cancer_model.joblib \
  -t outputs/threshold.json \
  -i path/to/new_samples.csv \
  -o predictions.csv
```

This writes `predictions.csv` with columns:
- `proba_malignant` — calibrated probability for malignant (1)
- `pred_binary` — 1 if malignant (probability ≥ threshold), else 0
- `pred_label` — `"M"` or `"B"` (same rule as above)
- `id` — preserved if present in input

## Tips

- **Calibration**: probabilities are calibrated (`isotonic` by default). Change with `--calibration sigmoid` if needed.
- **Class weight vs SMOTE**: If SMOTE isn’t desired, try `--no-smote`; the grid includes robust tree models.
- **Reproducibility**: set `--random_state` for deterministic splits.
- **Explainability**: check `permutation_importance.csv` to see which features matter most on the test split.
- **Safety**: Validate model on a *held‑out* test set (as done in the script). Do not deploy without proper clinical validation.
