"""Checks for the synthetic Figure 5 surrogate qubit-sweep helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_figure5.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_figure5", MODULE_PATH)
figure5 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure5
SPEC.loader.exec_module(figure5)


def test_trace_normalization_matches_original_rectangular_behavior():
    square = np.array([[2.0, 0.5], [0.5, 2.0]])
    rectangular = np.array([[2.0, 1.0]])

    square_norm = figure5.normalize_kernel_like_original(square, "trace")
    rectangular_norm = figure5.normalize_kernel_like_original(rectangular, "trace")

    assert np.isclose(np.trace(square_norm), 1.0)
    assert np.allclose(rectangular_norm, rectangular)


def test_cosine_normalization_matches_original_rectangular_behavior():
    square = np.array([[4.0, 2.0], [2.0, 9.0]])
    rectangular = np.array([[4.0, 3.0]])

    square_norm = figure5.normalize_kernel_like_original(square, "cosine")
    rectangular_norm = figure5.normalize_kernel_like_original(rectangular, "cosine")

    assert np.allclose(np.diag(square_norm), [1.0, 1.0])
    assert np.allclose(rectangular_norm, rectangular)


def test_frobenius_normalization_applies_to_rectangular_kernel():
    rectangular = np.array([[3.0, 4.0]])
    normalized = figure5.normalize_kernel_like_original(rectangular, "frobenius")

    assert np.isclose(np.linalg.norm(normalized, ord="fro"), 1.0)


def test_metric_series_sorts_values_by_q():
    rows = [
        {"model": "medsiglip-448", "q": 6, "accuracy": 0.6},
        {"model": "medsiglip-448", "q": 2, "accuracy": 0.2},
        {"model": "rad-dino", "q": 2, "accuracy": 0.9},
    ]

    q_values, values = figure5.metric_series(
        rows,
        model="medsiglip-448",
        metric="accuracy",
    )

    assert q_values == [2, 6]
    assert values == [0.2, 0.6]
