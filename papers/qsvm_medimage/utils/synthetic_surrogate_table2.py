#!/usr/bin/env python3
"""Compute a Table 2-style Tier 1 detail table on surrogate or real data.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 2 protocol shape: QSVM C=1 versus linear SVM C=1,
reported as mean/std accuracy and F1 over seeds for the 18 Tier 1 configs.

The QSVM scoring implementation is imported from
``synthetic_surrogate_table1`` so Tables 1 and 2 use the same fidelity kernel,
normalization, SVM configuration, and metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]

for root in (
    PROJECT_ROOT,
    REPRO_ROOT,
):
    root_string = str(root)

    if root_string not in sys.path:
        sys.path.insert(
            0,
            root_string,
        )


from lib.svm_pipeline import (  # noqa: E402
    preprocess,
    split_indices,
)
from synthetic_surrogate_table1 import (  # noqa: E402
    TIER1_CONFIGS,
    SyntheticSpec,
    load_dataset,
    parse_ints,
    score_linear_svc_c1,
    score_qsvm_c1,
)

PAPER_TABLE2_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T2"


def metric_mean(
    rows: list[dict[str, object]],
    method: str,
    metric: str,
) -> float:
    """Return the finite mean of one metric for one method."""
    values = np.asarray(
        [row[metric] for row in rows if row["method"] == method],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def metric_std(
    rows: list[dict[str, object]],
    method: str,
    metric: str,
) -> float:
    """Return the sample standard deviation across seeds."""
    values = np.asarray(
        [row[metric] for row in rows if row["method"] == method],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if values.size < 2:
        return float("nan")

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def tier1_verdict(
    qsvm_f1: float,
    linear_f1: float,
) -> str:
    """Compare mean minority-class F1 values."""
    if np.isclose(
        qsvm_f1,
        linear_f1,
    ):
        return "F1 TIE"

    if qsvm_f1 > linear_f1:
        return "QSVM F1 WIN"

    return "LINEAR F1 WIN"


def combine_rows(
    metadata: dict[str, object],
    scores: dict[str, object],
) -> dict[str, object]:
    """Combine run metadata and classifier scores."""
    row = dict(metadata)

    row.update(scores)

    return row


def run_one_table2_config(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> list[dict[str, object]]:
    """Run the two Table 2 Tier 1 comparators for one model, q, and seed."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )

    if q > X.shape[1]:
        raise ValueError(
            f"q={q} exceeds the raw feature dimension {X.shape[1]} for model {model!r}."
        )

    (
        training_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    (
        X_train,
        _,
        X_test,
        explained_variance_ratio,
    ) = preprocess(
        X[training_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    y_train = y[training_indices]

    y_validation = y[validation_indices]

    y_test = y[test_indices]

    run_metadata = {
        "source": source,
        "synthetic_surrogate": (source != "real"),
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(training_indices)),
        "val_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "train_class_0": int(np.sum(y_train == 0)),
        "train_class_1": int(np.sum(y_train == 1)),
        "val_class_0": int(np.sum(y_validation == 0)),
        "val_class_1": int(np.sum(y_validation == 1)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "explained_variance_ratio": float(explained_variance_ratio),
    }

    qsvm_row = score_qsvm_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )

    linear_row = score_linear_svc_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )

    return [
        combine_rows(
            run_metadata,
            qsvm_row,
        ),
        combine_rows(
            run_metadata,
            linear_row,
        ),
    ]


def summarize_table2(
    long_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate seed-level rows into the Table 2 column layout."""
    summary_rows: list[dict[str, object]] = []

    for model, q in TIER1_CONFIGS:
        rows = [row for row in long_rows if (row["model"] == model and row["q"] == q)]

        if not rows:
            raise RuntimeError(f"No result rows for model={model!r}, q={q}.")

        seed_count = len({int(row["seed"]) for row in rows})

        qsvm_f1 = metric_mean(
            rows,
            "qsvm",
            "f1",
        )

        linear_f1 = metric_mean(
            rows,
            "linear",
            "f1",
        )

        summary_rows.append(
            {
                "model": model,
                "q": q,
                "n_seeds": seed_count,
                "qsvm_accuracy_mean": metric_mean(
                    rows,
                    "qsvm",
                    "accuracy",
                ),
                "qsvm_accuracy_std": metric_std(
                    rows,
                    "qsvm",
                    "accuracy",
                ),
                "qsvm_f1_mean": qsvm_f1,
                "qsvm_f1_std": metric_std(
                    rows,
                    "qsvm",
                    "f1",
                ),
                "linear_accuracy_mean": metric_mean(
                    rows,
                    "linear",
                    "accuracy",
                ),
                "linear_accuracy_std": metric_std(
                    rows,
                    "linear",
                    "accuracy",
                ),
                "linear_f1_mean": linear_f1,
                "linear_f1_std": metric_std(
                    rows,
                    "linear",
                    "f1",
                ),
                "f1_gain": (qsvm_f1 - linear_f1),
                "verdict": tier1_verdict(
                    qsvm_f1,
                    linear_f1,
                ),
            }
        )

    return summary_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write result rows to CSV."""
    if not rows:
        raise ValueError(f"No rows to write to {path}.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted({key for row in rows for key in row})

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(
    mean: float,
    std: float,
) -> str:
    """Format a mean and sample standard deviation."""
    if np.isnan(std):
        return f"{mean:.3f} +/- NA"

    return f"{mean:.3f} +/- {std:.3f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 2 artifact."""
    summary_rows = payload["summary_rows"]

    if not isinstance(
        summary_rows,
        list,
    ):
        raise TypeError("payload summary_rows must be a list.")

    lines = [
        "# Synthetic surrogate Table 2 pipeline",
        "",
        (
            "This artifact is a surrogate computation only. "
            "It does not reproduce the paper numbers because the gated "
            "MIMIC-CXR embedding dataset is not available locally."
        ),
        "",
        (f"Paper methodology pointer: {payload['paper_pointer']}"),
        "",
        (
            "| Model | q | Seeds | QSVM acc | QSVM F1 | "
            "Linear acc | Linear F1 | F1 gain | Verdict |"
        ),
        ("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"),
    ]

    for row in summary_rows:
        lines.append(
            (
                "| {model} | {q} | {seeds} | {qacc} | {qf1} | "
                "{lacc} | {lf1} | {gain:.3f} | {verdict} |"
            ).format(
                model=row["model"],
                q=row["q"],
                seeds=row["n_seeds"],
                qacc=format_mean_std(
                    row["qsvm_accuracy_mean"],
                    row["qsvm_accuracy_std"],
                ),
                qf1=format_mean_std(
                    row["qsvm_f1_mean"],
                    row["qsvm_f1_std"],
                ),
                lacc=format_mean_std(
                    row["linear_accuracy_mean"],
                    row["linear_accuracy_std"],
                ),
                lf1=format_mean_std(
                    row["linear_f1_mean"],
                    row["linear_f1_std"],
                ),
                gain=row["f1_gain"],
                verdict=row["verdict"],
            )
        )

    lines.extend(
        [
            "",
            (
                "A win requires strictly greater mean "
                "minority-class F1. Numerically equal values "
                "are reported as ties."
            ),
            "",
            "Data source metadata:",
            "",
            "```json",
            json.dumps(
                payload["data"],
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix for the selected source."""
    if source == "synthetic":
        return "synthetic_surrogate_table2"

    if source == "synthetic_file":
        return "synthetic_file_table2"

    return "real_table2"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--source",
        choices=(
            "synthetic",
            "synthetic_file",
            "real",
        ),
        default="synthetic",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--ambient-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minority-frac",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--signal",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    seeds: list[int],
) -> None:
    """Validate command-line arguments."""
    if not seeds:
        raise ValueError("At least one seed is required.")

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")

    if args.ambient_dim <= 0:
        raise ValueError("--ambient-dim must be positive.")

    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive.")

    if not 0.0 < args.minority_frac < 1.0:
        raise ValueError("--minority-frac must be in the interval (0, 1).")

    if args.signal < 0.0:
        raise ValueError("--signal must be non-negative.")

    if args.noise < 0.0:
        raise ValueError("--noise must be non-negative.")

    if args.source in {
        "synthetic_file",
        "real",
    }:
        if args.data_root is None:
            raise ValueError(f"--data-root is required for source={args.source!r}.")

        if not args.data_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")


def main() -> None:
    """Compute and write the Table 2-style artifact."""
    args = parse_args()

    seeds = parse_ints(args.seeds)

    validate_args(
        args,
        seeds,
    )

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    long_rows: list[dict[str, object]] = []

    for model, q in TIER1_CONFIGS:
        for seed in seeds:
            configuration_rows = run_one_table2_config(
                source=args.source,
                model=model,
                q=q,
                seed=seed,
                data_root=args.data_root,
                synthetic=synthetic,
            )

            long_rows.extend(configuration_rows)

            print(
                f"[table2] model={model} q={q} seed={seed}",
                flush=True,
            )

    summary_rows = summarize_table2(long_rows)

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_path = args.results_dir / f"{prefix}_long.csv"

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    write_csv(
        long_path,
        long_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 2",
        "paper_pointer": (PAPER_TABLE2_POINTER),
        "paths": {
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": (args.source != "real"),
            "synthetic_spec": (
                asdict(synthetic) if args.source == "synthetic" else None
            ),
            "data_root": (str(args.data_root) if args.data_root else None),
            "seeds": seeds,
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "kernel_normalization": ("trace"),
            "normalization_protocol": (
                "QSVM training and test kernels are divided by the "
                "training-kernel trace through score_qsvm_c1."
            ),
            "table2_configs": list(TIER1_CONFIGS),
        },
        "summary_rows": (summary_rows),
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_markdown(
        markdown_path,
        payload=payload,
    )

    wins = sum(row["verdict"] == "QSVM F1 WIN" for row in summary_rows)

    ties = sum(row["verdict"] == "F1 TIE" for row in summary_rows)

    losses = sum(row["verdict"] == "LINEAR F1 WIN" for row in summary_rows)

    console_summary = {
        "qsvm_f1_wins": wins,
        "f1_ties": ties,
        "linear_f1_wins": losses,
        "total": len(summary_rows),
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
        )
    )

    print(f"Wrote {long_path}")

    print(f"Wrote {summary_path}")

    print(f"Wrote {json_path}")

    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
