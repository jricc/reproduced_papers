#!/usr/bin/env python3
"""Run paper-shaped artifact scripts from one explicit dataset source.

This convenience wrapper invokes the dedicated scripts for Tables 1 to 10 and
Figures 2 to 5. Every selected script receives the same data source, dataset
root, and result directory.

The wrapper does not calculate scientific results itself.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_BY_NAME = {
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
    "figure2": "synthetic_surrogate_figure2.py",
    "figure3": "synthetic_surrogate_figure3.py",
    "figure4": "synthetic_surrogate_figure4.py",
    "figure5": "synthetic_surrogate_figure5.py",
}


def parse_names(raw: str) -> list:
    """Parse and validate the requested artifact names."""
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]

    if not names:
        raise argparse.ArgumentTypeError("At least one artifact name is required.")

    if names == ["all"]:
        return list(SCRIPT_BY_NAME)

    if "all" in names:
        raise argparse.ArgumentTypeError(
            "'all' cannot be combined with individual artifact names."
        )

    unknown_names = [name for name in names if name not in SCRIPT_BY_NAME]

    if unknown_names:
        raise argparse.ArgumentTypeError(
            "Unknown artifacts: " + ", ".join(unknown_names)
        )

    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("Artifact names must be unique.")

    return names


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--source",
        choices=(
            "synthetic_file",
            "synthetic",
            "real",
        ),
        default="synthetic_file",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root. Required for synthetic_file and real "
            "sources. Unused for in-memory synthetic data."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--only",
        default="all",
        help=(
            "Comma-separated artifact names, for example "
            "figure2,table6,table10, or all."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser.parse_args()


def validate_data_source(
    source: str,
    data_root: Path | None,
) -> None:
    """Validate the dataset root required by the selected source."""
    if source in {"synthetic_file", "real"}:
        if data_root is None:
            raise ValueError(f"--data-root is required for source={source!r}.")

        if not data_root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    if source == "synthetic_file":
        index_path = data_root / "synthetic_dataset_index.json"

        if not index_path.is_file():
            raise FileNotFoundError(f"Synthetic dataset index not found: {index_path}")


def build_command(
    *,
    script: Path,
    source: str,
    data_root: Path | None,
    results_dir: Path,
) -> list:
    """Build the subprocess command for one artifact script."""
    command = [
        sys.executable,
        "-B",
        str(script),
        "--source",
        source,
        "--results-dir",
        str(results_dir),
    ]

    if data_root is not None:
        command.extend(
            [
                "--data-root",
                str(data_root),
            ]
        )

    return command


def main() -> None:
    """Run every requested artifact script sequentially."""
    args = parse_args()

    names = parse_names(args.only)

    validate_data_source(
        args.source,
        args.data_root,
    )

    utils_directory = Path(__file__).resolve().parent

    environment = os.environ.copy()

    environment.setdefault(
        "MPLCONFIGDIR",
        "/tmp/qsvm_medimage_matplotlib",
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed: list[str] = []

    for name in names:
        script = utils_directory / SCRIPT_BY_NAME[name]

        if not script.is_file():
            raise FileNotFoundError(f"Artifact script not found: {script}")

        command = build_command(
            script=script,
            source=args.source,
            data_root=args.data_root,
            results_dir=args.results_dir,
        )

        print(
            " ".join(command),
            flush=True,
        )

        if args.dry_run:
            continue

        subprocess.run(
            command,
            check=True,
            env=environment,
        )

        completed.append(name)

    if args.dry_run:
        print(f"Dry run completed for {len(names)} artifacts.")
    else:
        print(f"Completed {len(completed)} artifacts: " + ", ".join(completed))


if __name__ == "__main__":
    main()
