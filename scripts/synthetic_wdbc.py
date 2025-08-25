#!/usr/bin/env python3
# Generate a small WDBC-like CSV for CI runs.
import argparse
import numpy as np
import pandas as pd

COLUMNS = [
"id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
"radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
 "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="Output CSV path")
    ap.add_argument("-n", "--n_samples", type=int, default=120, help="Number of samples")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    y = rng.integers(0, 2, size=args.n_samples)  # 0=B, 1=M
    rows = []
    for i in range(args.n_samples):
        cls = y[i]
        # Base means differ slightly by class to make learning feasible
        mu = 0.0 if cls == 0 else 0.6
        vec = rng.normal(loc=mu, scale=1.0, size=30)
        # map to positive domain + some scaling to mimic ranges
        # group the features to different scales
        scale = np.array([
            3, 2, 3, 5, 0.2, 0.4, 0.5, 0.3, 0.2, 0.05,   # means
            0.5, 0.5, 1.0, 3.0, 0.05, 0.1, 0.1 ,0.08 ,0.05, 0.02,  # se
            4, 2, 4, 6, 0.2, 0.4, 0.5, 0.3, 0.2, 0.05   # worst
        ])
        vals = np.abs(vec) * scale + scale  # simple positive-ish distribution
        row = [100000 + i, "M" if cls==1 else "B"] + vals.tolist()
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(args.output, index=False)
    print(f"[SAVE] {args.output} with shape={df.shape}")

if __name__ == "__main__":
    main()
