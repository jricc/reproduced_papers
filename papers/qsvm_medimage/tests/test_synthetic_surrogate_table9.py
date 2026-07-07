"""Checks for the synthetic Table 9 surrogate C-tuning helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table9.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table9", MODULE_PATH)
table9 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table9
SPEC.loader.exec_module(table9)


def test_select_best_c_by_validation_f1_tiebreaks_to_smaller_c():
    rows = [
        {"C": 1.0, "val_f1": 0.5},
        {"C": 0.1, "val_f1": 0.5},
        {"C": 10.0, "val_f1": 0.4},
    ]

    assert table9.select_best_c_by_validation_f1(rows) == 0.1


def test_trace_normalization_matches_original_rectangular_behavior():
    square = np.array([[2.0, 0.5], [0.5, 2.0]])
    rectangular = np.array([[2.0, 1.0]])

    square_norm = table9.normalize_kernel_like_original(square, "trace")
    rectangular_norm = table9.normalize_kernel_like_original(rectangular, "trace")

    assert np.isclose(np.trace(square_norm), 1.0)
    assert np.allclose(rectangular_norm, rectangular)


def test_format_signed_metric_avoids_negative_zero():
    assert table9.format_signed_metric(-0.00001) == "+0.000"
    assert table9.format_signed_metric(0.1234) == "+0.123"
    assert table9.format_signed_metric(-0.1234) == "-0.123"
