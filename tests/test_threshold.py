import numpy as np
from breast_cancer_detection_pipeline import choose_threshold


def test_choose_threshold_ranges():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    thr = choose_threshold(proba, y_true, strategy="youden")
    assert 0.0 <= thr <= 1.0
