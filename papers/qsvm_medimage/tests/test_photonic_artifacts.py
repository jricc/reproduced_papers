"""Checks for the photonic MerLin artifact driver (utils/photonic_artifacts.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

UTILS = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

MODULE_PATH = UTILS / "photonic_artifacts.py"
SPEC = importlib.util.spec_from_file_location("photonic_artifacts", MODULE_PATH)
pa = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pa
SPEC.loader.exec_module(pa)


def test_subsample_caps_size_and_keeps_both_classes():
    y = np.array([0] * 80 + [1] * 20)
    idx = np.arange(100)
    kept = pa.subsample(idx, y, cap=30, seed=0)
    assert len(kept) <= 32  # cap plus per-class rounding slack
    labels = y[kept]
    assert set(np.unique(labels)) == {0, 1}  # minority class preserved
    assert (labels == 1).sum() >= 1


def test_subsample_noop_when_under_cap():
    y = np.array([0, 1, 0, 1])
    idx = np.arange(4)
    kept = pa.subsample(idx, y, cap=10, seed=0)
    assert np.array_equal(kept, idx)


def test_effective_rank_identity_equals_dimension():
    # Trace-normalized identity has uniform spectrum -> eff rank == n.
    assert pa.effective_rank(np.eye(8)) == pytest.approx(8.0, rel=1e-6)


def test_effective_rank_rank_one_is_one():
    v = np.ones((6, 1))
    assert pa.effective_rank(v @ v.T) == pytest.approx(1.0, abs=1e-6)


def test_photonic_kernel_smoke_is_valid_gram():
    # Tiny two-photon SLOS kernel: diagonal 1, symmetric, bounded in [0, 1].
    rng = np.random.default_rng(0)
    q = 2
    X = rng.uniform(-1, 1, size=(6, q))
    K_tr, K_te = pa.photonic_fidelity_kernels(X, X[:2], q, n_photons=2, seed=0)
    assert K_tr.shape == (6, 6)
    assert K_te.shape == (2, 6)
    assert np.allclose(np.diag(K_tr), 1.0)
    assert np.allclose(K_tr, K_tr.T, atol=1e-6)
    assert K_tr.min() >= -1e-9 and K_tr.max() <= 1.0 + 1e-9