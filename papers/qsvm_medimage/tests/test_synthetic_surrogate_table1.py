"""Checks for the synthetic Table 1 surrogate aggregation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table1", MODULE_PATH)
table1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table1
SPEC.loader.exec_module(table1)


def test_summarize_computes_wins_totals_and_mean_f1_gains():
    tier1_delta = {
        table1.TIER1_CONFIGS[0]: 0.10,
        table1.TIER1_CONFIGS[1]: -0.20,
    }
    tier2_delta = {
        table1.TIER2_CONFIGS[0]: 0.10,
        table1.TIER2_CONFIGS[1]: -0.20,
    }

    rows = []
    for model, q in table1.table_configs():
        config = (model, q)
        qsvm_f1 = 0.50
        linear_f1 = qsvm_f1 - tier1_delta.get(config, 0.0)
        rbf_f1 = qsvm_f1 - tier2_delta.get(config, 0.0)
        rows.extend(
            [
                {"model": model, "q": q, "method": "qsvm", "f1": qsvm_f1},
                {"model": model, "q": q, "method": "linear", "f1": linear_f1},
                {"model": model, "q": q, "method": "rbf", "f1": rbf_f1},
            ]
        )

    summary_rows, aggregate = table1.summarize(rows)

    assert aggregate["tier1"]["wins"] == 1
    assert aggregate["tier1"]["total"] == 18
    assert aggregate["tier2"]["wins"] == 1
    assert aggregate["tier2"]["total"] == 7

    expected_tier1_mean = np.mean(
        [row["tier1_f1_gain"] for row in summary_rows if row["tier1_config"]]
    )
    expected_tier2_mean = np.mean(
        [row["tier2_f1_gain"] for row in summary_rows if row["tier2_config"]]
    )
    assert aggregate["tier1"]["mean_f1_gain"] == expected_tier1_mean
    assert aggregate["tier2"]["mean_f1_gain"] == expected_tier2_mean
