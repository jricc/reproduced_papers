from __future__ import annotations

import json
import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_shared_runtime_runs_the_reduced_rff_config(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--config", "configs/rff_error_reduced.json", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)
    errors = json.loads((run_dir / "metrics.json").read_text())["errors"]["rbf"]["10"]
    # Error decreases with D for a wide kernel (paper Fig. 5a).
    means = [sum(errors["2.0"][str(d)]) / 2 for d in (32, 256)]
    assert means[1] < means[0]
