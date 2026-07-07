"""Tests for StandardScaler -> PCA(q) -> MinMax[-1, 1] preprocessing."""
from __future__ import annotations

import numpy as np

from lib.synthetic_data import make_synthetic_embeddings
from lib.svm_pipeline import preprocess, split_indices


def test_preprocessing_outputs_q_features_in_minmax_range():
    X, y, _ = make_synthetic_embeddings(n_samples=160, embedding_dim=64, seed=0)
    idx_train, idx_val, idx_test = split_indices(y, seed=0)
    q = 4

    X_train, X_val, X_test, _ = preprocess(X[idx_train], X[idx_val], X[idx_test], q)

    assert X_train.shape == (len(idx_train), q)
    assert X_val.shape == (len(idx_val), q)
    assert X_test.shape == (len(idx_test), q)
    for split in (X_train, X_val, X_test):
        assert np.max(split) <= 1.0 + 1e-9
        assert np.min(split) >= -1.0 - 1e-9
