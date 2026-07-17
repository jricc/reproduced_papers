"""Checks for the synthetic Figure 4 surrogate heatmap helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_figure4.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_figure4", MODULE_PATH)
figure4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure4
SPEC.loader.exec_module(figure4)


def test_select_samples_sorted_by_class_caps_and_sorts():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.array([1, 0, 1, 0, 1, 0])

    X_sorted, y_sorted, class_counts = figure4.select_samples_sorted_by_class(X_train, y_train, 4)

    # Caps at the requested count.
    assert X_sorted.shape == (4, 2)
    assert y_sorted.shape == (4,)
    # Sorted by class label: all 0s come before all 1s.
    assert list(y_sorted) == sorted(y_sorted)
    # Class counts sum to the selected sample count.
    assert sum(class_counts) == 4


def test_select_samples_sorted_by_class_caps_at_available_rows():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.array([1, 0, 1, 0, 1, 0])

    X_sorted, _, _ = figure4.select_samples_sorted_by_class(X_train, y_train, 99)

    assert X_sorted.shape == (6, 2)


def test_normalize_kernel_for_plot_trace_normalizes():
    kernel = np.diag([2.0, 2.0])
    normalized = figure4.normalize_kernel_for_plot(kernel, "trace")

    assert np.isclose(np.trace(normalized), 1.0)


def test_kernel_summary_reports_basic_statistics():
    summary = figure4.kernel_summary(np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert summary["min"] == 0.0
    assert summary["max"] == 1.0
    assert summary["trace"] == 2.0
