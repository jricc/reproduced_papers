from __future__ import annotations

import importlib

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_runner_exposes_entry_point():
    runner = importlib.import_module("lib.runner")
    assert hasattr(runner, "train_and_evaluate")


def test_defaults_loads():
    cfg = load_runtime_ready_config()
    assert cfg["description"]
    assert "experiment" in cfg and "dataset" in cfg
