"""End-to-end smoke test of the data + SVM pipeline + runner."""
from __future__ import annotations

import importlib

import numpy as np

data_mod = importlib.import_module("lib.data")
pipe = importlib.import_module("lib.svm_pipeline")
runner = importlib.import_module("lib.runner")


def test_synthetic_shapes_and_balance():
    X, y = data_mod.make_synthetic_embeddings(n_samples=300, ambient_dim=64,
                                              minority_fraction=0.3, signal=0.0, seed=0)
    assert X.shape == (300, 64)
    assert set(np.unique(y)) <= {0, 1}
    assert 0.2 < y.mean() < 0.4  # imbalance near requested minority fraction


def test_signal_knob_changes_separability():
    # With signal, a balanced linear SVM should beat chance; with signal=0 it should not.
    rows0, rows2 = {}, {}
    X0, y0 = data_mod.make_synthetic_embeddings(n_samples=400, ambient_dim=64, signal=0.0, seed=0)
    X2, y2 = data_mod.make_synthetic_embeddings(n_samples=400, ambient_dim=64, signal=3.0, seed=0)
    for r in pipe.run_one(X0, y0, q=4, seed=0, classifiers=["linear_tuned"]):
        rows0[r["method"]] = r
    for r in pipe.run_one(X2, y2, q=4, seed=0, classifiers=["linear_tuned"]):
        rows2[r["method"]] = r
    assert rows2["linear_tuned"]["auc"] > rows0["linear_tuned"]["auc"]


def test_run_one_returns_all_methods():
    X, y = data_mod.make_synthetic_embeddings(n_samples=200, ambient_dim=64, signal=0.0, seed=1)
    rows = pipe.run_one(X, y, q=4, seed=1)
    methods = {r["method"] for r in rows}
    assert {"qsvm", "linear_c1", "rbf_c1", "rbf_rank_matched",
            "linear_balanced", "linear_tuned"} <= methods
    for r in rows:
        for k in ("f1", "recall", "auc", "accuracy", "eff_rank"):
            assert k in r


def test_runner_writes_outputs(tmp_path):
    cfg = {
        "dataset": {"source": "synthetic",
                    "synthetic": {"n_samples": 150, "ambient_dim": 32, "signal": 0.0}},
        "experiment": {"q_list": [2, 4], "seeds": [0],
                       "classifiers": ["qsvm", "linear_c1", "linear_balanced"]},
    }
    runner.train_and_evaluate(cfg, tmp_path)
    for name in ("results_long.csv", "summary.csv", "collapse_rates.csv", "meta.json"):
        assert (tmp_path / name).exists()
