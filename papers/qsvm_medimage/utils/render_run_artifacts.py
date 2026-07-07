#!/usr/bin/env python3
"""Render simple artifacts from an existing training run.

This script is deliberately read-only with respect to model computation: it reads
``results_long.csv``, ``summary.csv``, ``collapse_rates.csv``, and ``meta.json``
from a run directory, then writes human-readable artifacts. It does not generate
data, fit models, or recompute kernels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def format_float(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return raw
    if value != value:
        return "nan"
    return f"{value:.3f}"


def require_run_files(run_dir: Path) -> dict[str, Path]:
    paths = {
        "results_long": run_dir / "results_long.csv",
        "summary": run_dir / "summary.csv",
        "collapse_rates": run_dir / "collapse_rates.csv",
        "meta": run_dir / "meta.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing run files: " + ", ".join(missing))
    return paths


def write_markdown(
    path: Path,
    *,
    meta: dict[str, object],
    summary_rows: list[dict[str, str]],
    collapse_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Training Run Artifacts",
        "",
        "Generated from saved run outputs. No dataset generation, model training, or kernel recomputation is performed here.",
        "",
        "## Run metadata",
        "",
        "```json",
        json.dumps(meta, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| method | q | seeds | F1 | AUC | accuracy | eff. rank | collapse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {q} | {n_seeds} | {f1} | {auc} | {accuracy} | {eff_rank} | {collapse} |".format(
                method=row["method"],
                q=row["q"],
                n_seeds=row["n_seeds"],
                f1=format_float(row.get("f1_mean", "")),
                auc=format_float(row.get("auc_mean", "")),
                accuracy=format_float(row.get("accuracy_mean", "")),
                eff_rank=format_float(row.get("eff_rank_mean", "")),
                collapse=format_float(row.get("collapse_mean", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Collapse Rates",
            "",
            "| method | q | collapse rate | seeds |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in collapse_rows:
        lines.append(
            "| {method} | {q} | {collapse_rate} | {n_seeds} |".format(
                method=row["method"],
                q=row["q"],
                collapse_rate=format_float(row["collapse_rate"]),
                n_seeds=row["n_seeds"],
            )
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = require_run_files(args.run_dir)
    output_dir = args.output_dir or args.run_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv_rows(paths["summary"])
    collapse_rows = read_csv_rows(paths["collapse_rates"])
    meta = json.loads(paths["meta"].read_text())

    markdown_path = output_dir / "run_summary.md"
    manifest_path = output_dir / "run_artifacts.json"
    write_markdown(
        markdown_path,
        meta=meta,
        summary_rows=summary_rows,
        collapse_rows=collapse_rows,
    )

    manifest = {
        "source_run_dir": str(args.run_dir),
        "inputs": {key: str(value) for key, value in paths.items()},
        "outputs": {"markdown": str(markdown_path)},
        "n_summary_rows": len(summary_rows),
        "n_collapse_rows": len(collapse_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
