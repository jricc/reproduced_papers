from __future__ import annotations

import importlib
import unittest
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

from HQPINN import run_from_project


# Minimal detailed-row payload returned by mocked DEE CSV readers.
# The test does not care about metric values themselves, only that the
# train path can complete and append a summary row for the selected case.
DEE_ROW = {
    "epoch": "1",
    "elapsed (s)": "0.01",
    "Loss": "1.0e-03",
    "IC": "2.0e-04",
    "BC": "3.0e-04",
    "F": "4.0e-04",
}

# Same idea for TAF: this mirrors the columns expected by the TAF summary
# append logic so we can exercise the control flow without real training data.
TAF_ROW = {
    "step": "1",
    "elapsed (s)": "0.01",
    "Loss": "1.0e-03",
    "BC": "2.0e-04",
    "F": "3.0e-04",
    "L_in": "4.0e-04",
    "L_out": "5.0e-04",
    "L_wall": "6.0e-04",
    "L_per": "7.0e-04",
}


# Each entry describes one config-driven training request that should trigger
# exactly one detailed training call. `expected_call` is the core assertion:
# it captures the unique `(out_dir, model_label)` pair that must be produced
# by the selected experiment and no other sibling size.
TRAIN_SELECTION_CASES: list[dict[str, Any]] = [
    {
        "experiment": "dee-hy-m",
        "module": "HQPINN.lib.DEE.dee_hy_m",
        "train_attr": "train_dee",
        "config": {
            "experiment": "dee-hy-m",
            "mode": "train",
            "backend": "local",
            "model": {"n_layers": 4, "n_nodes": 20, "n_photons": 1},
        },
        "expected_call": (
            "HQPINN/results/DEE/dee_hy_m_20-4-1",
            "hy-m_20-4-1",
        ),
        "row": DEE_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
    },
    {
        "experiment": "dee-qq-m",
        "module": "HQPINN.lib.DEE.dee_qq_m",
        "train_attr": "train_dee",
        "config": {
            "experiment": "dee-qq-m",
            "mode": "train",
            "backend": "local",
            "model": {"n_photons": 5},
        },
        "expected_call": (
            "HQPINN/results/DEE/dee_qq_m_5",
            "qq-m_5",
        ),
        "row": DEE_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
    },
    {
        "experiment": "dee-hy-pl",
        "module": "HQPINN.lib.DEE.dee_hy_pl",
        "train_attr": "train_dee",
        "config": {
            "experiment": "dee-hy-pl",
            "mode": "train",
            "backend": "local",
            "model": {"n_layers": 4, "n_nodes": 20, "q_layers": 2},
        },
        "expected_call": (
            "HQPINN/results/DEE/dee_hy_pl_20-4-2",
            "hy-pl_20-4-2",
        ),
        "row": DEE_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
    },
    {
        "experiment": "dee-qq-pl",
        "module": "HQPINN.lib.DEE.dee_qq_pl",
        "train_attr": "train_dee",
        "config": {
            "experiment": "dee-qq-pl",
            "mode": "train",
            "backend": "local",
            "model": {"q_layers": 4},
        },
        "expected_call": (
            "HQPINN/results/DEE/dee_qq_pl_4",
            "qq-pl_4",
        ),
        "row": DEE_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
    },
    {
        "experiment": "taf-hy-m",
        "module": "HQPINN.lib.TAF.taf_hy_m",
        "train_attr": "train_taf",
        "config": {
            "experiment": "taf-hy-m",
            "mode": "train",
            "backend": "local",
            "model": {"n_layers": 4, "n_nodes": 80, "n_photons": 2},
        },
        "expected_call": (
            "HQPINN/results/TAF/taf_hy_m_80-4-2",
            "hy-m_80-4-2",
        ),
        "row": TAF_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
        "patch_training_sets": True,
    },
    {
        "experiment": "taf-qq-m",
        "module": "HQPINN.lib.TAF.taf_qq_m",
        "train_attr": "train_taf",
        "config": {
            "experiment": "taf-qq-m",
            "mode": "train",
            "backend": "local",
            "model": {"n_photons": 5},
        },
        "expected_call": (
            "HQPINN/results/TAF/taf_qq_m_5",
            "qq-m_5",
        ),
        "row": TAF_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
        "patch_training_sets": True,
    },
    {
        "experiment": "taf-hy-pl",
        "module": "HQPINN.lib.TAF.taf_hy_pl",
        "train_attr": "train_taf",
        "config": {
            "experiment": "taf-hy-pl",
            "mode": "train",
            "backend": "local",
            "model": {"n_layers": 4, "n_nodes": 80, "q_layers": 2},
        },
        "expected_call": (
            "HQPINN/results/TAF/taf_hy_pl_80-4-2",
            "hy-pl_80-4-2",
        ),
        "row": TAF_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
        "patch_training_sets": True,
    },
    {
        "experiment": "taf-qq-pl",
        "module": "HQPINN.lib.TAF.taf_qq_pl",
        "train_attr": "train_taf",
        "config": {
            "experiment": "taf-qq-pl",
            "mode": "train",
            "backend": "local",
            "model": {"q_layers": 6},
        },
        "expected_call": (
            "HQPINN/results/TAF/taf_qq_pl_6",
            "qq-pl_6",
        ),
        "row": TAF_ROW,
        "train_return": (1.0e-3, 2.0e-4, 3.0e-4, 123),
        "patch_training_sets": True,
    },
]


class TrainSelectionTests(unittest.TestCase):
    def test_targeted_train_config_runs_only_the_selected_case(self) -> None:
        for case in TRAIN_SELECTION_CASES:
            module = importlib.import_module(case["module"])
            train_calls: list[tuple[str, str]] = []

            def _fake_train(**kwargs):
                # Record where the module wants to write its detailed result.
                # If a single config accidentally iterates over a whole family,
                # this list will contain multiple `(out_dir, model_label)` pairs.
                train_calls.append((kwargs["out_dir"], kwargs["model_label"]))
                return case["train_return"]

            with self.subTest(experiment=case["experiment"]):
                with ExitStack() as stack:
                    # Replace heavy training and filesystem side effects with
                    # lightweight stubs. This keeps the test focused on the
                    # execution stack: config -> runner -> experiment module ->
                    # selected case.
                    stack.enter_context(
                        patch.object(module, case["train_attr"], side_effect=_fake_train)
                    )
                    stack.enter_context(patch("builtins.print"))
                    stack.enter_context(
                        patch.object(module, "get_latest_checkpoint", return_value=None)
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_training_row_for_run_id",
                            return_value=case["row"],
                        )
                    )
                    stack.enter_context(
                        patch.object(module, "append_summary_row", return_value=False)
                    )
                    stack.enter_context(
                        patch.object(module, "make_optimizer", return_value=object())
                    )
                    stack.enter_context(patch.object(module, "seed_everything"))
                    stack.enter_context(patch.object(module.os, "makedirs"))
                    stack.enter_context(patch.object(module.torch, "save"))
                    if case.get("patch_training_sets"):
                        stack.enter_context(
                            patch.object(module, "load_training_sets", return_value={})
                        )

                    run_from_project(case["config"])

                # The regression we are guarding against is "one targeted config
                # launches several sibling models". Passing this assertion means
                # the module executed exactly one case, and it was the expected one.
                self.assertEqual(train_calls, [case["expected_call"]])
