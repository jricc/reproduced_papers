"""Tests for linear-kernel rank and precomputed-kernel compatibility."""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from lib.quantum_kernel import trace_normalize_kernel
from lib.synthetic_data import make_synthetic_embeddings
from lib.svm_pipeline import preprocess, split_indices


def test_linear_kernel_is_symmetric_and_rank_bounded_by_q():
    X, y, _ = make_synthetic_embeddings(n_samples=120, embedding_dim=48, seed=0)
    idx_train, idx_val, idx_test = split_indices(y, seed=0)
    q = 5
    X_train, _, _, _ = preprocess(X[idx_train], X[idx_val], X[idx_test], q)

    K_linear = X_train @ X_train.T

    assert np.allclose(K_linear, K_linear.T)
    assert np.linalg.matrix_rank(K_linear, tol=1e-10) <= q


def test_trace_normalization_sets_trace_to_one():
    K = np.diag([2.0, 3.0, 5.0])
    K_norm = trace_normalize_kernel(K)

    assert np.isclose(np.trace(K_norm), 1.0)


def test_precomputed_kernel_shape_is_accepted_by_svc():
    X, y, _ = make_synthetic_embeddings(n_samples=120, embedding_dim=48, seed=1)
    idx_train, idx_val, idx_test = split_indices(y, seed=1)
    X_train, _, X_test, _ = preprocess(X[idx_train], X[idx_val], X[idx_test], q=4)
    y_train = y[idx_train]

    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    svc = SVC(kernel="precomputed", C=1.0)
    svc.fit(K_train, y_train)

    pred = svc.predict(K_test)
    assert pred.shape == (len(idx_test),)
