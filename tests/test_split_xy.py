import pandas as pd
from breast_cancer_detection_pipeline import split_X_y


def test_split_xy_basic():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "diagnosis": ["M", "B", "M"],
        "radius_mean": [10.0, 11.0, 12.0],
        "texture_mean": [5.0, 6.0, 7.0]
    })
    X, y = split_X_y(df, target_col="diagnosis")
    assert "diagnosis" not in X.columns
    assert "id" not in X.columns
    assert X.shape == (3, 2)
    assert set(y.unique()) == {0, 1}
