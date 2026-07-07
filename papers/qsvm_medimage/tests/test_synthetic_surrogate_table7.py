"""Checks for the synthetic Table 7 surrogate normalization helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table7.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table7", MODULE_PATH)
table7 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table7
SPEC.loader.exec_module(table7)


def test_trace_normalization_uses_train_trace_for_train_and_test():
    K_train = np.array([[2.0, 0.5], [0.5, 2.0]])
    K_test = np.array([[1.0, 0.25]])

    train_norm, test_norm = table7.normalize_train_test_kernels(
        K_train,
        K_test,
        "trace",
    )

    assert np.isclose(np.trace(train_norm), 1.0)
    assert np.allclose(test_norm, K_test / 4.0)


def test_frobenius_normalization_uses_train_norm_for_train_and_test():
    K_train = np.array([[2.0, 0.0], [0.0, 2.0]])
    K_test = np.array([[2.0, 1.0]])

    train_norm, test_norm = table7.normalize_train_test_kernels(
        K_train,
        K_test,
        "frobenius",
    )

    scale = np.linalg.norm(K_train, ord="fro")
    assert np.isclose(np.linalg.norm(train_norm, ord="fro"), 1.0)
    assert np.allclose(test_norm, K_test / scale)


def test_cosine_normalization_uses_train_and_test_diagonals():
    K_train = np.array([[4.0, 2.0], [2.0, 9.0]])
    K_test = np.array([[4.0, 3.0]])
    K_test_diag = np.array([16.0])

    train_norm, test_norm = table7.normalize_train_test_kernels(
        K_train,
        K_test,
        "cosine",
        K_test_diag=K_test_diag,
    )

    assert np.allclose(np.diag(train_norm), [1.0, 1.0])
    assert np.allclose(test_norm, [[0.5, 0.25]])


def test_format_metric_uses_three_decimals():
    assert table7.format_metric(0.12345) == "0.123"
    assert table7.format_metric(0.98765) == "0.988"
