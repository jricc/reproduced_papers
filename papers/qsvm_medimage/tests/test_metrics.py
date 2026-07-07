"""Tests for minority-class metrics and collapse detection."""
from __future__ import annotations

import numpy as np

from lib.svm_pipeline import _scores


def test_collapse_true_when_all_predictions_are_majority_class():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.0, -0.1, -0.2])

    metrics = _scores(y_true, y_pred, scores)

    assert metrics["f1"] == 0.0
    assert metrics["predicted_minority_count"] == 0
    assert metrics["true_minority_count"] == 2
    assert metrics["collapse"] is True
    assert metrics["confusion_matrix"] == [[3, 0], [2, 0]]
