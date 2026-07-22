#!/usr/bin/env python3
"""Render human-readable artifacts from an existing experiment run.

This script reads saved outputs from ``lib.runner``:

- ``results_long.csv``:
  one row per method, q value, and seed;

- ``summary.csv``:
  mean and standard deviation across seeds;

- ``collapse_rates.csv``:
  fraction of seeds with majority-only predictions;

- ``meta.json``:
  run configuration and execution metadata.

The script writes:

- ``run_summary.md``:
  concise human-readable results;

- ``run_artifacts.json``:
  manifest linking the source files and rendered outputs.

This script does not:

- generate data;
- preprocess embeddings;
- construct kernels;
- fit classifiers;
- recompute metrics.

All displayed values come from the saved run outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REQUIRED_RUN_FILES = {
    "results_long": "results_long.csv",
    "summary": "summary.csv",
    "collapse_rates": "collapse_rates.csv",
    "meta": "meta.json",
}

REQUIRED_SUMMARY_COLUMNS = {
    "method",
    "q",
    "n_seeds",
}

REQUIRED_COLLAPSE_COLUMNS = {
    "method",
    "q",
    "collapse_rate",
    "n_seeds",
}


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {path}"
            )

        return list(reader)


def read_json_object(
    path: Path,
) -> dict[str, object]:
    """Read a JSON file whose top-level value must be an object."""
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return payload


def validate_columns(
    rows: list[dict[str, str]],
    required_columns: set[str],
    source: Path,
) -> None:
    """Validate the required columns of a CSV table."""
    if not rows:
        raise ValueError(
            f"CSV file contains no data rows: {source}"
        )

    available_columns = set(
        rows[0]
    )

    missing_columns = (
        required_columns
        - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"{source} is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def markdown_text(
    value: object,
) -> str:
    """Escape text that would otherwise break a Markdown table."""
    return (
        str(value)
        .replace("|", r"\|")
        .replace("\n", " ")
        .strip()
    )


def format_float(
    raw: object,
    decimals: int = 3,
) -> str:
    """Format a finite numeric value or return a clear missing-value marker."""
    if raw is None:
        return "n/a"

    text = str(raw).strip()

    if not text:
        return "n/a"

    try:
        value = float(text)
    except (TypeError, ValueError):
        return markdown_text(text)

    if math.isnan(value):
        return "n/a"

    if math.isinf(value):
        return (
            "inf"
            if value > 0
            else "-inf"
        )

    return f"{value:.{decimals}f}"


def format_rate(
    raw: object,
) -> str:
    """Format a rate in [0, 1] as a percentage."""
    if raw is None:
        return "n/a"

    text = str(raw).strip()

    if not text:
        return "n/a"

    try:
        value = float(text)
    except (TypeError, ValueError):
        return markdown_text(text)

    if not math.isfinite(value):
        return "n/a"

    return f"{100.0 * value:.1f}%"


def integer_sort_value(
    raw: object,
) -> tuple[int, str]:
    """Return a stable numeric sort key for q-like values."""
    try:
        return (
            int(float(str(raw))),
            "",
        )
    except (TypeError, ValueError):
        return (
            10**9,
            str(raw),
        )


def sorted_result_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Sort result rows by method and then q."""
    return sorted(
        rows,
        key=lambda row: (
            str(
                row.get(
                    "method",
                    "",
                )
            ),
            integer_sort_value(
                row.get(
                    "q",
                    "",
                )
            ),
        ),
    )


def require_run_files(
    run_dir: Path,
) -> dict[str, Path]:
    """Return and validate the standard files produced by one run."""
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    paths = {
        key: run_dir / filename
        for key, filename in REQUIRED_RUN_FILES.items()
    }

    missing_paths = [
        path
        for path in paths.values()
        if not path.is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing run files: "
            + ", ".join(
                str(path)
                for path in missing_paths
            )
        )

    return paths


def summary_metric_columns(
    summary_rows: list[dict[str, str]],
) -> list[tuple[str, str, str | None]]:
    """Return the metrics available in the run summary.

    Each tuple contains:

        display label;
        mean column;
        standard-deviation column.
    """
    available_columns = set(
        summary_rows[0]
    )

    candidates = [
        (
            "Minority F1",
            "f1_mean",
            "f1_std",
        ),
        (
            "ROC-AUC",
            "auc_mean",
            "auc_std",
        ),
        (
            "Accuracy",
            "accuracy_mean",
            "accuracy_std",
        ),
        (
            "Minority precision",
            "precision_mean",
            "precision_std",
        ),
        (
            "Minority recall",
            "recall_mean",
            "recall_std",
        ),
        (
            "Effective rank",
            "eff_rank_mean",
            "eff_rank_std",
        ),
        (
            "Collapse rate",
            "collapse_mean",
            "collapse_std",
        ),
        (
            "Zero-F1 rate",
            "zero_f1_mean",
            "zero_f1_std",
        ),
    ]

    return [
        (
            label,
            mean_column,
            (
                std_column
                if std_column in available_columns
                else None
            ),
        )
        for (
            label,
            mean_column,
            std_column,
        ) in candidates
        if mean_column in available_columns
    ]


def format_mean_std(
    row: dict[str, str],
    mean_column: str,
    std_column: str | None,
    *,
    rate: bool = False,
) -> str:
    """Format a mean and optional standard deviation."""
    mean_raw = row.get(
        mean_column,
        "",
    )

    if rate:
        mean_text = format_rate(
            mean_raw
        )
    else:
        mean_text = format_float(
            mean_raw
        )

    if std_column is None:
        return mean_text

    standard_deviation_raw = row.get(
        std_column,
        "",
    )

    if not str(
        standard_deviation_raw
    ).strip():
        return mean_text

    if rate:
        standard_deviation_text = format_rate(
            standard_deviation_raw
        )
    else:
        standard_deviation_text = format_float(
            standard_deviation_raw
        )

    if (
        mean_text == "n/a"
        or standard_deviation_text == "n/a"
    ):
        return mean_text

    return (
        f"{mean_text} ± "
        f"{standard_deviation_text}"
    )


def write_markdown(
    path: Path,
    *,
    meta: dict[str, object],
    summary_rows: list[dict[str, str]],
    collapse_rows: list[dict[str, str]],
    results_long_count: int,
) -> None:
    """Write the human-readable run summary."""
    metrics = summary_metric_columns(
        summary_rows
    )

    summary_headers = [
        "Method",
        "q",
        "Seeds",
    ]

    summary_headers.extend(
        label
        for (
            label,
            _,
            _,
        ) in metrics
    )

    lines = [
        "# Training Run Summary",
        "",
        (
            "Generated from saved run outputs. No dataset generation, "
            "preprocessing, kernel construction, model training, or metric "
            "recomputation was performed."
        ),
        "",
        "## Run metadata",
        "",
        "```json",
        json.dumps(
            meta,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Saved rows",
        "",
        f"- Per-seed result rows: {results_long_count}",
        f"- Aggregated summary rows: {len(summary_rows)}",
        f"- Collapse-rate rows: {len(collapse_rows)}",
        "",
        "## Aggregated metrics",
        "",
        (
            "| "
            + " | ".join(
                summary_headers
            )
            + " |"
        ),
        (
            "| "
            + " | ".join(
                [
                    "---",
                    "---:",
                    "---:",
                    *[
                        "---:"
                        for _ in metrics
                    ],
                ]
            )
            + " |"
        ),
    ]

    for row in sorted_result_rows(
        summary_rows
    ):
        values = [
            markdown_text(
                row.get(
                    "method",
                    "",
                )
            ),
            markdown_text(
                row.get(
                    "q",
                    "",
                )
            ),
            markdown_text(
                row.get(
                    "n_seeds",
                    "",
                )
            ),
        ]

        for (
            _,
            mean_column,
            std_column,
        ) in metrics:
            is_rate = mean_column in {
                "collapse_mean",
                "zero_f1_mean",
            }

            values.append(
                format_mean_std(
                    row,
                    mean_column,
                    std_column,
                    rate=is_rate,
                )
            )

        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Majority-only collapse rates",
            "",
            (
                "A run is counted as collapsed when the classifier predicts "
                "no minority-class samples. A zero minority-class F1 can also "
                "occur without majority-only collapse and is therefore a "
                "separate condition."
            ),
            "",
            "| Method | q | Collapse rate | Seeds |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for row in sorted_result_rows(
        collapse_rows
    ):
        lines.append(
            "| {method} | {q} | {collapse_rate} | {n_seeds} |".format(
                method=markdown_text(
                    row.get(
                        "method",
                        "",
                    )
                ),
                q=markdown_text(
                    row.get(
                        "q",
                        "",
                    )
                ),
                collapse_rate=format_rate(
                    row.get(
                        "collapse_rate",
                        "",
                    )
                ),
                n_seeds=markdown_text(
                    row.get(
                        "n_seeds",
                        "",
                    )
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- Results using synthetic data describe the controlled "
                "synthetic benchmark, not the inaccessible medical dataset."
            ),
            (
                "- Effective rank describes the kernel spectrum and does not "
                "by itself establish better classification."
            ),
            (
                "- Similar ROC-AUC with different minority-class F1 may "
                "indicate effects from threshold placement, margin scaling, "
                "or class imbalance, but does not prove a single cause."
            ),
            (
                "- Class-weighted and additionally tuned models are "
                "diagnostic baselines unless explicitly included in the "
                "paper protocol."
            ),
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Run directory containing results_long.csv, summary.csv, "
            "collapse_rates.csv, and meta.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Artifact output directory. Defaults to "
            "<run-dir>/artifacts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Render the Markdown summary and JSON manifest."""
    args = parse_args()

    paths = require_run_files(
        args.run_dir
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.run_dir / "artifacts"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_long_rows = read_csv_rows(
        paths["results_long"]
    )

    summary_rows = read_csv_rows(
        paths["summary"]
    )

    collapse_rows = read_csv_rows(
        paths["collapse_rates"]
    )

    meta = read_json_object(
        paths["meta"]
    )

    validate_columns(
        summary_rows,
        REQUIRED_SUMMARY_COLUMNS,
        paths["summary"],
    )

    validate_columns(
        collapse_rows,
        REQUIRED_COLLAPSE_COLUMNS,
        paths["collapse_rates"],
    )

    markdown_path = (
        output_dir
        / "run_summary.md"
    )

    manifest_path = (
        output_dir
        / "run_artifacts.json"
    )

    write_markdown(
        markdown_path,
        meta=meta,
        summary_rows=summary_rows,
        collapse_rows=collapse_rows,
        results_long_count=len(
            results_long_rows
        ),
    )

    manifest = {
        "artifact_type": (
            "rendered_saved_run"
        ),
        "recomputes_models": False,
        "recomputes_kernels": False,
        "recomputes_metrics": False,
        "source_run_dir": str(
            args.run_dir
        ),
        "inputs": {
            key: str(value)
            for key, value in paths.items()
        },
        "outputs": {
            "markdown": str(
                markdown_path
            ),
            "manifest": str(
                manifest_path
            ),
        },
        "row_counts": {
            "results_long": len(
                results_long_rows
            ),
            "summary": len(
                summary_rows
            ),
            "collapse_rates": len(
                collapse_rows
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
