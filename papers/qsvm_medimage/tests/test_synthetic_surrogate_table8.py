"""Checks for the synthetic Table 8 surrogate 1-DOF/3-DOF helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table8.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table8", MODULE_PATH)
table8 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table8
SPEC.loader.exec_module(table8)


def test_pca_dim_for_circuit_matches_original_switch():
    assert table8.pca_dim_for_circuit(8, "1-DOF") == 8
    assert table8.pca_dim_for_circuit(8, "3-DOF") == 24


def test_bsp_3dof_statevector_has_unit_norm():
    state = table8.bsp_3dof_statevector(np.zeros(6), q=2)

    assert state.shape == (4,)
    assert np.isclose(np.vdot(state, state).real, 1.0)


def test_bsp_3dof_statevector_validates_feature_count():
    try:
        table8.bsp_3dof_statevector(np.zeros(5), q=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid 3-DOF feature count")


def test_trace_normalization_matches_original_rectangular_behavior():
    square = np.array([[2.0, 0.5], [0.5, 2.0]])
    rectangular = np.array([[2.0, 1.0]])

    square_norm = table8.normalize_kernel_like_original(square, "trace")
    rectangular_norm = table8.normalize_kernel_like_original(rectangular, "trace")

    assert np.isclose(np.trace(square_norm), 1.0)
    assert np.allclose(rectangular_norm, rectangular)


def test_format_metric_uses_three_decimals():
    assert table8.format_metric(0.12345) == "0.123"
    assert table8.format_metric(0.98765) == "0.988"
