"""Checks for the synthetic Table 4 surrogate aggregation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table4.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table4", MODULE_PATH)
table4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table4
SPEC.loader.exec_module(table4)


def test_summarize_table4_computes_f1_and_relative_gains():
    rows = []
    for model, q in table4.TIER2_CONFIGS:
        for seed in range(2):
            rows.extend(
                [
                    {
                        "model": model,
                        "q": q,
                        "seed": seed,
                        "method": "qsvm",
                        "C": 1.0,
                        "accuracy": 0.8,
                        "f1": 0.6,
                    },
                    {
                        "model": model,
                        "q": q,
                        "seed": seed,
                        "method": "rbf",
                        "C": 10.0,
                        "accuracy": 0.7,
                        "f1": 0.4,
                    },
                ]
            )

    summary_rows = table4.summarize_table4(rows)

    assert len(summary_rows) == 7
    assert summary_rows[0]["best_svm_kernel"] == "rbf"
    assert summary_rows[0]["best_svm_selected_c_values"] == "10"
    assert summary_rows[0]["f1_gain"] == 0.19999999999999996
    assert round(summary_rows[0]["relative_gain_percent"], 6) == 50.0
    assert summary_rows[0]["verdict"] == "QSVM F1 WIN"
