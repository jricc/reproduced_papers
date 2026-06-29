"""Checks for the synthetic Table 2 surrogate aggregation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table2.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table2", MODULE_PATH)
table2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table2
SPEC.loader.exec_module(table2)


def test_summarize_table2_computes_means_stds_gains_and_verdicts():
    first_config = table2.TIER1_CONFIGS[0]
    second_config = table2.TIER1_CONFIGS[1]

    rows = []
    for model, q in table2.TIER1_CONFIGS:
        qsvm_f1_values = [0.5, 0.5]
        linear_f1_values = [0.5, 0.5]
        if (model, q) == first_config:
            qsvm_f1_values = [0.7, 0.9]
            linear_f1_values = [0.5, 0.5]
        if (model, q) == second_config:
            qsvm_f1_values = [0.2, 0.2]
            linear_f1_values = [0.5, 0.5]

        for seed, (qsvm_f1, linear_f1) in enumerate(zip(qsvm_f1_values, linear_f1_values)):
            rows.extend(
                [
                    {
                        "model": model,
                        "q": q,
                        "seed": seed,
                        "method": "qsvm",
                        "accuracy": 0.8,
                        "f1": qsvm_f1,
                    },
                    {
                        "model": model,
                        "q": q,
                        "seed": seed,
                        "method": "linear",
                        "accuracy": 0.7,
                        "f1": linear_f1,
                    },
                ]
            )

    summary_rows = table2.summarize_table2(rows)

    assert len(summary_rows) == 18
    assert summary_rows[0]["verdict"] == "QSVM F1 WIN"
    assert summary_rows[0]["f1_gain"] == 0.30000000000000004
    assert summary_rows[1]["verdict"] == "LINEAR F1 WIN"
    assert summary_rows[2]["verdict"] == "F1 TIE"
    assert summary_rows[0]["qsvm_accuracy_mean"] == 0.8
    assert summary_rows[0]["linear_accuracy_mean"] == 0.7
    assert round(summary_rows[0]["qsvm_f1_std"], 6) == 0.141421
