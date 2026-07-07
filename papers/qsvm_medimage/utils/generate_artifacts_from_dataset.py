#!/usr/bin/env python3
"""Run paper-style artifact scripts from a fixed dataset.

This is a convenience wrapper around the existing ``synthetic_surrogate_*``
scripts. It passes the same ``--source``, ``--data-root``, and ``--results-dir``
to every selected artifact script, so tables and figures are generated from one
explicit dataset source.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_BY_NAME = {
    "figure2": "synthetic_surrogate_figure2.py",
    "figure3": "synthetic_surrogate_figure3.py",
    "figure4": "synthetic_surrogate_figure4.py",
    "figure5": "synthetic_surrogate_figure5.py",
    "table1": "synthetic_surrogate_table1.py",
    "table2": "synthetic_surrogate_table2.py",
    "table3": "synthetic_surrogate_table3.py",
    "table4": "synthetic_surrogate_table4.py",
    "table5": "synthetic_surrogate_table5.py",
    "table6": "synthetic_surrogate_table6.py",
    "table7": "synthetic_surrogate_table7.py",
    "table8": "synthetic_surrogate_table8.py",
    "table9": "synthetic_surrogate_table9.py",
    "table10": "synthetic_surrogate_table10.py",
}


def parse_names(raw: str) -> list[str]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if names == ["all"]:
        return list(SCRIPT_BY_NAME)
    unknown = [name for name in names if name not in SCRIPT_BY_NAME]
    if unknown:
        raise ValueError("Unknown artifacts: " + ", ".join(unknown))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic_file", "synthetic", "real"), default="synthetic_file")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated artifact names, for example figure2,table6,table10, or all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = parse_names(args.only)
    utils_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/qsvm_medimage_matplotlib")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        script = utils_dir / SCRIPT_BY_NAME[name]
        cmd = [
            sys.executable,
            "-B",
            str(script),
            "--source",
            args.source,
            "--data-root",
            str(args.data_root),
            "--results-dir",
            str(args.results_dir),
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
