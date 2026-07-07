"""Checks for the synthetic Figure 2 surrogate eigenspectrum helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_figure2.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_figure2", MODULE_PATH)
figure2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure2
SPEC.loader.exec_module(figure2)


def test_normalized_eigenvalues_sum_to_one_and_sort_descending():
    kernel = np.diag([2.0, 1.0, 0.0])
    eigenvalues = figure2.normalized_eigenvalues(kernel)

    assert np.isclose(np.sum(eigenvalues), 1.0)
    assert list(eigenvalues) == [2.0 / 3.0, 1.0 / 3.0, 0.0]


def test_count_positive_eigenvalues_uses_tolerance():
    eigenvalues = np.array([0.5, 0.1, 1e-12, 0.0])

    assert figure2.count_positive_eigenvalues(eigenvalues) == 2
    assert figure2.count_positive_eigenvalues(eigenvalues, tol=1e-13) == 3


def test_eigenvalue_rows_include_cumulative_variance():
    rows = figure2.eigenvalue_rows(np.array([3.0, 1.0]))

    assert rows == [
        {
            "eigenvalue_index": 0,
            "raw_eigenvalue": 3.0,
            "normalized_eigenvalue": 0.75,
            "eigenvalue_count": 1,
            "cumulative_variance": 0.75,
        },
        {
            "eigenvalue_index": 1,
            "raw_eigenvalue": 1.0,
            "normalized_eigenvalue": 0.25,
            "eigenvalue_count": 2,
            "cumulative_variance": 1.0,
        },
    ]


def test_linear_kernel_rank_validation_is_bounded_by_q():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    K = X @ X.T
    raw = figure2.kernel_eigenvalues(K)

    assert figure2.count_positive_eigenvalues(raw) <= X.shape[1]
