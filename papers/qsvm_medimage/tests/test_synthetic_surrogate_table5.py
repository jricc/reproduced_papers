"""Checks for the synthetic Table 5 surrogate rank helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table5.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table5", MODULE_PATH)
table5 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table5
SPEC.loader.exec_module(table5)


def test_count_positive_eigenvalues_uses_tolerance():
    kernel = np.diag([2.0, 1.0, 1e-12, 0.0])

    assert table5.count_positive_eigenvalues(kernel) == 2
    assert table5.count_positive_eigenvalues(kernel, tol=1e-13) == 3


def test_format_float_uses_three_decimals():
    assert table5.format_float(3.14159) == "3.142"
