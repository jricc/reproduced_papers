"""Checks for the synthetic geometry audit helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "audit_synthetic_geometry.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("audit_synthetic_geometry", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_effective_rank_from_eigenvalues_handles_zero_values():
    eigenvalues = np.array([2.0, 2.0, 0.0])

    assert np.isclose(audit.effective_rank_from_eigenvalues(eigenvalues), 2.0)


def test_components_needed_returns_first_threshold_crossing():
    cumulative = np.array([0.4, 0.7, 0.95])

    assert audit.components_needed(cumulative, 0.90) == 3
    assert audit.components_needed(cumulative, 0.99) is None


def test_summarize_linear_full_rank_uses_feature_rank_bound():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    summary = audit.summarize_linear_full_rank(X)

    assert summary["linear_full_positive_rank"] <= X.shape[1]
    assert summary["linear_full_effective_rank"] <= X.shape[1]
    assert summary["linear_full_lambda_max"] > 0.0
    assert summary["linear_full_lambda_max_raw"] == summary["linear_full_lambda_max"]
    assert np.isclose(summary["linear_full_lambda_max_trace_1"], 0.75)
    assert np.isclose(summary["linear_full_lambda_max_trace_n"], 2.25)


def test_summarize_pairwise_geometry_reports_offdiagonal_stats():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    summary = audit.summarize_pairwise_geometry(X, "x")

    assert "x_distance_mean" in summary
    assert "x_cosine_mean" in summary


def test_add_paper_table5_comparison_adds_target_deltas():
    row = {
        "model": "medsiglip-448",
        "q": 4,
        "pca_explained_variance": 0.326,
        "linear_full_positive_rank": 4,
        "linear_full_effective_rank": 3.77,
        "linear_full_lambda_max": 770.6,
        "linear_full_lambda_max_trace_n": 770.6,
    }

    audit.add_paper_table5_comparison(row)

    assert row["paper_table5_has_target"] is True
    assert row["paper_table5_pca_var_percent"] == 32.6
    assert np.isclose(row["delta_pca_var_percent_vs_table5"], 0.0)
    assert row["delta_positive_rank_vs_table5"] == 0.0
    assert row["ratio_lambda_max_vs_table5"] == 1.0
    assert row["ratio_lambda_max_raw_vs_table5"] == 1.0
    assert row["ratio_lambda_max_trace_n_vs_table5"] == 1.0
    assert row["paper_table5_lambda_max_note"] == "paper normalization unknown; inspect raw and trace_n"


def test_dagger_table5_rows_do_not_add_linear_rank_deltas():
    row = {
        "model": "medsiglip-448",
        "q": 11,
        "pca_explained_variance": 0.56,
        "linear_full_positive_rank": 11,
        "linear_full_effective_rank": 11.0,
        "linear_full_lambda_max": 100.0,
        "linear_full_lambda_max_trace_n": 100.0,
    }

    audit.add_paper_table5_comparison(row)

    assert row["paper_table5_has_target"] is True
    assert row["paper_table5_linear_comparison"] is False
    assert np.isclose(row["delta_pca_var_percent_vs_table5"], 0.0)
    assert row["delta_effective_rank_vs_table5"] is None
    assert row["delta_lambda_max_vs_table5"] is None
    assert row["delta_lambda_max_raw_vs_table5"] is None
    assert row["delta_lambda_max_trace_n_vs_table5"] is None
