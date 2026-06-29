"""Checks for the synthetic Table 6 surrogate kernel-stat helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table6.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table6", MODULE_PATH)
table6 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table6
SPEC.loader.exec_module(table6)


def test_select_sorted_training_subset_sorts_by_class():
    X_train = np.arange(12).reshape(6, 2)
    y_train = np.array([1, 0, 1, 0, 0, 1])

    _, y_subset = table6.select_sorted_training_subset(
        X_train,
        y_train,
        seed=0,
        subsample_size=6,
    )

    assert list(y_subset) == [0, 0, 0, 1, 1, 1]


def test_linear_kernel_stats_uses_population_variance():
    kernel = np.array([[1.0, 0.0], [0.0, 3.0]])
    stats = table6.linear_kernel_stats(kernel, np.array([0, 1]))

    assert stats["linear_kernel_mean"] == 1.0
    assert np.isclose(stats["linear_kernel_variance"], stats["linear_kernel_std"] ** 2)
    assert stats["between_class_mean"] == 0.0
