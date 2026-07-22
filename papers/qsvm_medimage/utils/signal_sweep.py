#!/usr/bin/env python3
"""Evaluate classifiers across synthetic signal-strength settings.

This script studies how classifier metrics change when label-related latent
factors become more or less visible in synthetic embeddings.

The synthetic generator first creates hidden signal factors and labels. The
``signal_strength`` parameter then controls the contribution of those
label-related factors to the final embedding vectors.

Important interpretation
------------------------
``signal_strength`` is a parameter of the synthetic generator. It is not a
quantity measured from the inaccessible medical embeddings.

A value of zero removes the signal-related contribution from the embeddings.
It does not change the label rule itself. Noise in the latent label score is
controlled separately by ``score_noise_std``.

The synthetic label rule is nonlinear and was defined by this reproduction.
It does not come from the paper.

Minority-class F1 and ROC-AUC answer different questions:

- F1 evaluates predictions at the classifier's decision threshold;
- ROC-AUC evaluates score ranking across thresholds.

Similar ROC-AUC with different F1 suggests that threshold placement, margin
scaling, or class imbalance may contribute to the difference. It does not
prove that the F1 difference has only one cause.

The script writes:

- a long CSV with one row per signal, seed, and classifier;
- an aggregated CSV with means and standard deviations;
- a JSON file describing the experiment.

Example
-------
    python utils/signal_sweep.py \
        --q 11 \
        --seeds 0,1,2,3,4 \
        --signals 0,0.25,0.5,1,2 \
        --output-prefix results/signal_sweep_q11
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
        sys.path.insert(0, root_string)


from lib.svm_pipeline import run_one  # noqa: E402
from lib.synthetic_data import (  # noqa: E402
    make_synthetic_embeddings,
)

DEFAULT_CLASSIFIERS = (
    "qsvm",
    "linear_c1",
    "rbf_tuned_c",
    "rbf_c1",
    "rbf_rank_matched",
    "linear_balanced",
    "linear_tuned",
)

AVAILABLE_CLASSIFIERS = {
    "qsvm",
    "qsvm_photonic",
    "linear_c1",
    "rbf_tuned_c",
    "rbf_c1",
    "rbf_rank_matched",
    "linear_balanced",
    "linear_tuned",
}

AGGREGATED_METRICS = (
    "f1",
    "auc",
    "precision",
    "recall",
    "accuracy",
    "eff_rank",
    "collapse",
    "zero_f1",
)


def parse_ints(
    raw: str,
) -> list:
    """Parse a non-empty comma-separated list of unique integers."""
    try:
        values = [
            int(part.strip())
            for part in raw.split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "Integer values must be unique."
        )

    return values


def parse_floats(
    raw: str,
) -> list:
    """Parse a non-empty comma-separated list of unique finite floats."""
    try:
        values = [
            float(part.strip())
            for part in raw.split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of numbers."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one number is required."
        )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise argparse.ArgumentTypeError(
            "All values must be finite."
        )

    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "Values must be unique."
        )

    return values


def parse_classifiers(
    raw: str,
) -> list:
    """Parse and validate a comma-separated classifier list."""
    classifiers = [
        part.strip()
        for part in raw.split(",")
        if part.strip()
    ]

    if not classifiers:
        raise argparse.ArgumentTypeError(
            "At least one classifier is required."
        )

    unknown = [
        classifier
        for classifier in classifiers
        if classifier not in AVAILABLE_CLASSIFIERS
    ]

    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown classifiers: "
            + ", ".join(unknown)
        )

    if len(classifiers) != len(
        set(classifiers)
    ):
        raise argparse.ArgumentTypeError(
            "Classifier names must be unique."
        )

    if (
        "rbf_rank_matched" in classifiers
        and "qsvm" not in classifiers
    ):
        raise argparse.ArgumentTypeError(
            "rbf_rank_matched requires qsvm so that the "
            "quantum effective rank can be computed."
        )

    return classifiers


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--q",
        type=int,
        default=11,
        help="PCA dimension and qubit count.",
    )

    parser.add_argument(
        "--seeds",
        type=parse_ints,
        default=parse_ints(
            "0,1,2,3,4"
        ),
    )

    parser.add_argument(
        "--signals",
        type=parse_floats,
        default=parse_floats(
            "0,0.25,0.5,1,2"
        ),
        help=(
            "Synthetic signal strengths controlling how strongly "
            "label-related latent factors appear in the embeddings."
        ),
    )

    parser.add_argument(
        "--classifiers",
        type=parse_classifiers,
        default=list(
            DEFAULT_CLASSIFIERS
        ),
        help=(
            "Comma-separated classifier names. "
            "The default includes paper comparisons and "
            "additional diagnostic baselines."
        ),
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "trace",
            "none",
            "frobenius",
        ),
        default="trace",
        help=(
            "Normalization applied consistently to training "
            "and test fidelity kernels."
        ),
    )

    parser.add_argument(
        "--model-name",
        choices=(
            "synthetic_medsiglip",
            "synthetic_raddino",
            "synthetic_vit",
        ),
        default="synthetic_medsiglip",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.304,
    )

    parser.add_argument(
        "--n-signal-latents",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n-nuisance-latents",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--score-noise-std",
        type=float,
        default=None,
        help=(
            "Noise standard deviation in the latent label score."
        ),
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help=(
            "Noise standard deviation in the final embeddings."
        ),
    )

    parser.add_argument(
        "--nuisance-scale",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-distribution",
        choices=(
            "normal",
            "uniform",
            "rademacher",
            "skewed_uniform",
        ),
        default=None,
    )

    parser.add_argument(
        "--nuisance-skew-strength",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-skew-decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "results/signal_sweep"
        ),
        help=(
            "Output prefix without extension. The script adds "
            "_long.csv, _summary.csv, and .json."
        ),
    )

    # Backward-compatible alias for the old aggregated CSV option.
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Deprecated alias for the aggregated CSV path. "
            "Prefer --output-prefix."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate argument consistency."""
    if args.q <= 0:
        raise ValueError(
            "--q must be positive."
        )

    if (
        args.n_samples is not None
        and args.n_samples <= 0
    ):
        raise ValueError(
            "--n-samples must be positive."
        )

    if (
        args.embedding_dim is not None
        and args.embedding_dim <= 0
    ):
        raise ValueError(
            "--embedding-dim must be positive."
        )

    if not 0.0 < args.positive_ratio < 1.0:
        raise ValueError(
            "--positive-ratio must be in (0, 1)."
        )

    if (
        args.embedding_dim is not None
        and args.q > args.embedding_dim
    ):
        raise ValueError(
            f"q={args.q} exceeds embedding_dim="
            f"{args.embedding_dim}."
        )

    non_negative_parameters = {
        "--score-noise-std": (
            args.score_noise_std
        ),
        "--noise-std": args.noise_std,
        "--nuisance-scale": (
            args.nuisance_scale
        ),
        "--nuisance-skew-strength": (
            args.nuisance_skew_strength
        ),
    }

    for name, value in (
        non_negative_parameters.items()
    ):
        if (
            value is not None
            and value < 0.0
        ):
            raise ValueError(
                f"{name} must be non-negative."
            )

    bounded_parameters = {
        "--nuisance-decay": (
            args.nuisance_decay
        ),
        "--nuisance-skew-decay": (
            args.nuisance_skew_decay
        ),
    }

    for name, value in (
        bounded_parameters.items()
    ):
        if (
            value is not None
            and not 0.0 < value <= 1.0
        ):
            raise ValueError(
                f"{name} must be in (0, 1]."
            )


def finite_mean(
    values: list[object],
) -> float:
    """Return the mean of finite numeric values."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = array[
        np.isfinite(array)
    ]

    if finite_values.size == 0:
        return float("nan")

    return float(
        np.mean(finite_values)
    )


def finite_std(
    values: list[object],
) -> float:
    """Return the population standard deviation of finite values."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = array[
        np.isfinite(array)
    ]

    if finite_values.size == 0:
        return float("nan")

    return float(
        np.std(finite_values)
    )


def aggregate_rows(
    long_rows: list[dict[str, object]],
    q: int,
) -> list[dict[str, object]]:
    """Aggregate per-seed rows by signal strength and method."""
    grouped: dict[
        tuple[float, str],
        list[dict[str, object]],
    ] = {}

    for row in long_rows:
        key = (
            float(row["signal"]),
            str(row["method"]),
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        signal_strength,
        method,
    ), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
        ),
    ):
        summary: dict[str, object] = {
            "signal": signal_strength,
            "method": method,
            "q": q,
            "n_seeds": len(rows),
        }

        for metric in AGGREGATED_METRICS:
            metric_values = [
                row.get(
                    metric,
                    float("nan"),
                )
                for row in rows
            ]

            summary[
                f"{metric}_mean"
            ] = finite_mean(
                metric_values
            )

            summary[
                f"{metric}_std"
            ] = finite_std(
                metric_values
            )

        # Friendly aliases used by existing plotting scripts.
        summary["collapse_rate"] = (
            summary["collapse_mean"]
        )

        summary["zero_f1_rate"] = (
            summary["zero_f1_mean"]
        )

        summary_rows.append(
            summary
        )

    return summary_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows using the union of all dictionary keys."""
    if not rows:
        raise ValueError(
            f"No rows to write to {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

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


def print_summary(
    rows: list[dict[str, object]],
    methods: list[str],
) -> None:
    """Print selected aggregated metrics without requiring pandas."""
    for method in methods:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        if not method_rows:
            continue

        print()
        print(method)
        print(
            "signal    F1      AUC     recall  "
            "collapse  zero_F1"
        )

        for row in sorted(
            method_rows,
            key=lambda item: float(
                item["signal"]
            ),
        ):
            print(
                f"{float(row['signal']):6.2f}  "
                f"{float(row['f1_mean']):6.3f}  "
                f"{float(row['auc_mean']):6.3f}  "
                f"{float(row['recall_mean']):6.3f}  "
                f"{float(row['collapse_rate']):8.3f}  "
                f"{float(row['zero_f1_rate']):7.3f}"
            )


def output_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    """Return long CSV, summary CSV, and JSON output paths."""
    if args.out is not None:
        summary_path = args.out

        stem = summary_path.stem

        if stem.endswith(
            "_summary"
        ):
            base_stem = stem[
                : -len("_summary")
            ]
        else:
            base_stem = stem

        long_path = summary_path.with_name(
            base_stem
            + "_long.csv"
        )

        json_path = summary_path.with_name(
            base_stem
            + ".json"
        )

        return (
            long_path,
            summary_path,
            json_path,
        )

    prefix = args.output_prefix

    long_path = prefix.with_name(
        prefix.name
        + "_long.csv"
    )

    summary_path = prefix.with_name(
        prefix.name
        + "_summary.csv"
    )

    json_path = prefix.with_suffix(
        ".json"
    )

    return (
        long_path,
        summary_path,
        json_path,
    )


def main() -> None:
    """Run the synthetic signal sweep."""
    args = parse_args()

    validate_args(
        args
    )

    long_rows: list[
        dict[str, object]
    ] = []

    generation_metadata_by_signal: dict[
        str,
        dict[str, object],
    ] = {}

    for signal_strength in args.signals:
        for seed in args.seeds:
            X, y, generation_metadata = (
                make_synthetic_embeddings(
                    n_samples=args.n_samples,
                    embedding_dim=(
                        args.embedding_dim
                    ),
                    positive_ratio=(
                        args.positive_ratio
                    ),
                    n_signal_latents=(
                        args.n_signal_latents
                    ),
                    n_nuisance_latents=(
                        args.n_nuisance_latents
                    ),
                    signal_strength=(
                        signal_strength
                    ),
                    score_noise_std=(
                        args.score_noise_std
                    ),
                    noise_std=args.noise_std,
                    nuisance_scale=(
                        args.nuisance_scale
                    ),
                    nuisance_decay=(
                        args.nuisance_decay
                    ),
                    nuisance_distribution=(
                        args.nuisance_distribution
                    ),
                    nuisance_skew_strength=(
                        args.nuisance_skew_strength
                    ),
                    nuisance_skew_decay=(
                        args.nuisance_skew_decay
                    ),
                    seed=seed,
                    model_name=(
                        args.model_name
                    ),
                )
            )

            generation_metadata_by_signal[
                str(signal_strength)
            ] = generation_metadata

            result_rows = run_one(
                X,
                y,
                q=args.q,
                seed=seed,
                classifiers=(
                    args.classifiers
                ),
                kernel_normalization=(
                    args.kernel_normalization
                ),
            )

            for row in result_rows:
                row["signal"] = (
                    signal_strength
                )
                row["model_name"] = (
                    args.model_name
                )
                row[
                    "kernel_normalization"
                ] = (
                    args.kernel_normalization
                )
                row["data_source"] = (
                    "synthetic_in_memory"
                )

                long_rows.append(
                    row
                )

        print(
            f"signal={signal_strength} completed",
            flush=True,
        )

    summary_rows = aggregate_rows(
        long_rows,
        args.q,
    )

    (
        long_path,
        summary_path,
        json_path,
    ) = output_paths(
        args
    )

    write_csv(
        long_path,
        long_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    payload = {
        "artifact": "signal_sweep",
        "scope": {
            "synthetic_only": True,
            "reconstructs_medical_distribution": False,
            "label_rule_from_paper": False,
            "calibrated_generator": True,
        },
        "experiment": {
            "q": args.q,
            "seeds": args.seeds,
            "signal_strengths": (
                args.signals
            ),
            "classifiers": (
                args.classifiers
            ),
            "kernel_normalization": (
                args.kernel_normalization
            ),
        },
        "generator": {
            "model_name": (
                args.model_name
            ),
            "n_samples_override": (
                args.n_samples
            ),
            "embedding_dim_override": (
                args.embedding_dim
            ),
            "positive_ratio": (
                args.positive_ratio
            ),
            "n_signal_latents_override": (
                args.n_signal_latents
            ),
            "n_nuisance_latents_override": (
                args.n_nuisance_latents
            ),
            "score_noise_std_override": (
                args.score_noise_std
            ),
            "noise_std_override": (
                args.noise_std
            ),
            "signal_strength_definition": (
                "Weight of label-related latent factors "
                "in the final embeddings."
            ),
            "generation_metadata_by_signal": (
                generation_metadata_by_signal
            ),
        },
        "metrics": {
            "primary": (
                "minority-class F1 for label 1"
            ),
            "secondary": (
                "ROC-AUC from SVM decision scores"
            ),
            "collapse_definition": (
                "No minority-class prediction."
            ),
            "zero_f1_definition": (
                "Minority-class F1 equals zero."
            ),
        },
        "paths": {
            "long_csv": str(
                long_path
            ),
            "summary_csv": str(
                summary_path
            ),
            "json": str(
                json_path
            ),
        },
        "summary_rows": (
            summary_rows
        ),
    }

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print_summary(
        summary_rows,
        methods=[
            method
            for method in (
                "qsvm",
                "linear_c1",
                "linear_balanced",
                "rbf_tuned_c",
            )
            if method
            in args.classifiers
        ],
    )

    print()
    print(
        f"Wrote {long_path}"
    )
    print(
        f"Wrote {summary_path}"
    )
    print(
        f"Wrote {json_path}"
    )


if __name__ == "__main__":
    main()
