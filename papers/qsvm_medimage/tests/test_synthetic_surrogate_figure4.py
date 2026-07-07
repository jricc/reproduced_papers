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


def test_select_first_samples_caps_at_available_rows():
    X_train = np.arange(12).reshape(6, 2)

    assert figure4.select_first_samples(X_train, 3).shape == (3, 2)
    assert figure4.select_first_samples(X_train, 99).shape == (6, 2)


def test_normalize_kernel_for_plot_trace_normalizes():
    kernel = np.diag([2.0, 2.0])
    normalized = figure4.normalize_kernel_for_plot(kernel, "trace")

    assert np.isclose(np.trace(normalized), 1.0)


def test_kernel_summary_reports_basic_statistics():
    summary = figure4.kernel_summary(np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert summary["min"] == 0.0
    assert summary["max"] == 1.0
    assert summary["trace"] == 2.0
