"""Checks for the synthetic Table 10 surrogate rank-matched RBF helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table10.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table10", MODULE_PATH)
table10 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table10
SPEC.loader.exec_module(table10)


def test_find_rank_matched_gamma_moves_rank_toward_target():
    X_train = np.array(
        [
            [-1.0, -1.0],
            [-0.5, -0.5],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )
    target_rank = 2.5

    gamma = table10.find_rank_matched_gamma(X_train, target_rank, tol=0.10)
    matched_rank = table10.rbf_effective_rank(X_train, gamma)

    assert abs(matched_rank - target_rank) / target_rank < 0.25


def test_summarize_table10_computes_collapse_rates_and_mean_f1():
    rows = [
        {
            "q": 4,
            "target_effective_rank": 6.0,
            "collapsed_rbf_scale": True,
            "collapsed_rbf_star": False,
            "collapsed_qsvm": False,
            "f1_rbf_scale": 0.0,
            "f1_rbf_star": 0.2,
            "f1_qsvm": 0.4,
        },
        {
            "q": 4,
            "target_effective_rank": 6.0,
            "collapsed_rbf_scale": False,
            "collapsed_rbf_star": False,
            "collapsed_qsvm": True,
            "f1_rbf_scale": 0.2,
            "f1_rbf_star": 0.4,
            "f1_qsvm": 0.0,
        },
    ]

    summary = table10.summarize_table10(rows)

    assert len(summary) == 1
    assert summary[0]["collapse_rate_rbf_scale"] == 0.5
    assert summary[0]["collapse_rate_qsvm"] == 0.5
    assert summary[0]["f1_mean_rbf_star"] == 0.30000000000000004


def test_trace_normalization_matches_original_rectangular_behavior():
    square = np.array([[2.0, 0.5], [0.5, 2.0]])
    rectangular = np.array([[2.0, 1.0]])

    square_norm = table10.normalize_kernel_like_original(square, "trace")
    rectangular_norm = table10.normalize_kernel_like_original(rectangular, "trace")

    assert np.isclose(np.trace(square_norm), 1.0)
    assert np.allclose(rectangular_norm, rectangular)
