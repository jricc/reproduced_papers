"""Checks for the synthetic Table 3 surrogate confusion matrix output."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table3.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table3", MODULE_PATH)
table3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table3
SPEC.loader.exec_module(table3)


def test_confusion_rows_keep_label_order_and_counts():
    rows = table3.confusion_rows({"confusion_matrix": [[24, 0], [5, 1]]})

    assert rows == [
        {
            "true_label": "class_0_majority",
            "pred_class_0_majority": 24,
            "pred_class_1_minority": 0,
        },
        {
            "true_label": "class_1_minority",
            "pred_class_0_majority": 5,
            "pred_class_1_minority": 1,
        },
    ]
