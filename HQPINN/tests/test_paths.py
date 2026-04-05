from __future__ import annotations

import unittest
from pathlib import Path

from HQPINN.paths import (
    PROJECT_ROOT,
    results_case_dir_for_model_dir,
    results_dir_for_model_dir,
)


class PathsTests(unittest.TestCase):
    def test_results_dir_for_model_dir_returns_benchmark_root(self) -> None:
        model_dir = PROJECT_ROOT / "models" / "DEE"
        self.assertEqual(
            results_dir_for_model_dir(model_dir),
            str(PROJECT_ROOT / "results" / "DEE"),
        )

    def test_results_case_dir_for_model_dir_returns_case_subdir(self) -> None:
        model_dir = Path("HQPINN/models/TAF")
        self.assertEqual(
            results_case_dir_for_model_dir(model_dir, "taf_cc_40-4"),
            str(PROJECT_ROOT / "results" / "TAF" / "taf_cc_40-4"),
        )


if __name__ == "__main__":
    unittest.main()
