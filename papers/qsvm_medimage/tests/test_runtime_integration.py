"""Fast checks for the shared catalogue runtime contract."""

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_runtime_markers_are_valid():
    defaults = json.loads((PROJECT_ROOT / "configs" / "defaults.json").read_text())
    cli = json.loads((PROJECT_ROOT / "cli.json").read_text())

    assert defaults["scope"] == "merlin_photonic_adaptation"
    assert defaults["device"] == "cpu"
    assert cli["arguments"]


def test_runner_reports_missing_dataset(tmp_path):
    from lib.runner import train_and_evaluate

    cfg = {
        "scope": "merlin_photonic_adaptation",
        "seed": 0,
        "dtype": "float32",
        "device": "cpu",
        "data_root": str(tmp_path),
        "dataset": {"filename": "missing.pkl"},
        "experiment": {"pca_dim": 2, "circuit_seed": 0, "max_samples": 100},
    }

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        train_and_evaluate(cfg, tmp_path / "run")
