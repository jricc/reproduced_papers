"""Checks for the synthetic Figure 3 surrogate eigenspectrum helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_figure3.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_figure3", MODULE_PATH)
figure3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure3
SPEC.loader.exec_module(figure3)


def test_cumulative_with_zero_starts_at_zero():
    counts, cumulative = figure3.cumulative_with_zero(np.array([0.25, 0.75]))

    assert list(counts) == [0, 1, 2]
    assert list(cumulative) == [0.0, 0.25, 1.0]


def test_compute_kernel_spectrum_normalizes_and_counts_rank():
    spectrum = figure3.compute_kernel_spectrum(np.diag([2.0, 1.0, 0.0]))

    assert np.isclose(np.sum(spectrum["eigenvalues"]), 1.0)
    assert spectrum["positive_rank"] == 2


def test_eigenvalue_rows_include_kernel_and_q():
    rows = figure3.eigenvalue_rows(
        [
            {
                "model": "medsiglip-448",
                "q": 4,
                "kernel": "linear",
                "eigenvalues": np.array([0.75, 0.25]),
            }
        ]
    )

    assert rows == [
        {
            "model": "medsiglip-448",
            "q": 4,
            "kernel": "linear",
            "eigenvalue_index": 0,
            "normalized_eigenvalue": 0.75,
            "eigenvalue_count": 1,
            "cumulative_variance": 0.75,
        },
        {
            "model": "medsiglip-448",
            "q": 4,
            "kernel": "linear",
            "eigenvalue_index": 1,
            "normalized_eigenvalue": 0.25,
            "eigenvalue_count": 2,
            "cumulative_variance": 1.0,
        },
    ]
