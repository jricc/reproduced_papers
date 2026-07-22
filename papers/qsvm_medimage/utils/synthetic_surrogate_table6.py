#!/usr/bin/env python3
"""Compute a Table 6-style linear-kernel variance diagnostic.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 6 diagnostic shape: linear-kernel mean, standard
deviation, and variance on a 200-sample training subset sorted by class.

The preprocessing pipeline is fitted only on the complete training split:

    StandardScaler
    PCA(q)
    MinMaxScaler[-1, 1]

The training subset is selected after preprocessing, without replacement, and
then sorted by class label. Sorting changes only the row and column order of
the kernel. It does not change its mean, standard deviation, or variance.
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
    SyntheticSpec,
    load_dataset,
)

PAPER_TABLE6_POINTER = (
    "https://"
    "arxiv.org/html/2604.24597v1#S4.T6"
)

TABLE6_CONFIGS = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def select_sorted_training_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    subsample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic training subset and sort it by class label."""
    features = np.asarray(
        X_train,
        dtype=np.float64,
    )

    labels = np.asarray(
        y_train,
        dtype=int,
    )

    if features.ndim != 2:
        raise ValueError(
            "X_train must be two-dimensional."
        )

    if labels.ndim != 1:
        raise ValueError(
            "y_train must be one-dimensional."
        )

    if len(features) != len(labels):
        raise ValueError(
            "X_train and y_train must contain the same number of samples."
        )

    if len(labels) == 0:
        raise ValueError(
            "The training set must not be empty."
        )

    if subsample_size <= 0:
        raise ValueError(
            "subsample_size must be positive."
        )

    selected_count = min(
        subsample_size,
        len(labels),
    )

    generator = np.random.default_rng(
        seed
    )

    selected_indices = generator.choice(
        len(labels),
        size=selected_count,
        replace=False,
    )

    sort_order = np.argsort(
        labels[selected_indices],
        kind="stable",
    )

    selected_indices = selected_indices[
        sort_order
    ]

    return (
        features[selected_indices],
        labels[selected_indices],
    )


def validate_square_kernel(
    kernel: np.ndarray,
) -> np.ndarray:
    """Validate and symmetrize a square linear-kernel matrix."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "The linear kernel must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "The linear kernel must be square."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "The linear kernel must not be empty."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "The linear kernel contains non-finite values."
        )

    return 0.5 * (
        matrix + matrix.T
    )


def linear_kernel_stats(
    kernel: np.ndarray,
) -> dict[str, float]:
    """Compute the three statistics reported in Table 6."""
    matrix = validate_square_kernel(
        kernel
    )

    return {
        "k_l_mean": float(
            np.mean(matrix)
        ),
        "k_l_std": float(
            np.std(matrix)
        ),
        "k_l_var": float(
            np.var(matrix)
        ),
    }


def compute_table6_row(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    subsample_size: int,
) -> dict[str, object]:
    """Compute the Table 6 linear-kernel statistics for one configuration."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )

    if q > X.shape[1]:
        raise ValueError(
            f"q={q} exceeds the raw feature dimension "
            f"{X.shape[1]} for model {model!r}."
        )

    (
        training_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    maximum_q = min(
        len(training_indices),
        X.shape[1],
    )

    if q > maximum_q:
        raise ValueError(
            f"q={q} exceeds the maximum supported PCA dimension "
            f"{maximum_q} for model {model!r}."
        )

    (
        X_train,
        _,
        _,
        explained_variance_ratio,
    ) = preprocess(
        X[training_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    y_train = y[
        training_indices
    ]

    (
        X_subset,
        y_subset,
    ) = select_sorted_training_subset(
        X_train,
        y_train,
        seed=seed,
        subsample_size=subsample_size,
    )

    linear_kernel = (
        X_subset @ X_subset.T
    )

    statistics = linear_kernel_stats(
        linear_kernel
    )

    row: dict[str, object] = {
        "source": source,
        "synthetic_surrogate": (
            source != "real"
        ),
        "model": model,
        "model_display": (
            MODEL_DISPLAY[model]
        ),
        "q": q,
        "seed": seed,
        "raw_feature_dimension": int(
            X.shape[1]
        ),
        "train_samples": int(
            len(training_indices)
        ),
        "val_samples": int(
            len(validation_indices)
        ),
        "test_samples": int(
            len(test_indices)
        ),
        "subsample_size": int(
            len(y_subset)
        ),
        "subsample_class_0": int(
            np.sum(
                y_subset == 0
            )
        ),
        "subsample_class_1": int(
            np.sum(
                y_subset == 1
            )
        ),
        "pca_variance_percent": float(
            100.0
            * explained_variance_ratio
        ),
        "kernel": "linear",
        "kernel_normalization": "none",
        "sorted_by_class": True,
    }

    row.update(
        statistics
    )

    return row


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the Table 6 summary rows to CSV."""
    if not rows:
        raise ValueError(
            f"No rows to write to {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0]
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def format_float(
    value: float,
) -> str:
    """Format one floating-point value for Markdown."""
    return f"{value:.4f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 6 artifact."""
    summary_rows = payload[
        "summary_rows"
    ]

    if not isinstance(
        summary_rows,
        list,
    ):
        raise TypeError(
            "payload summary_rows must be a list."
        )

    lines = [
        "# Synthetic surrogate Table 6 pipeline",
        "",
        (
            "This artifact is a surrogate computation only. "
            "It does not reproduce the paper numbers because the gated "
            "MIMIC-CXR embedding dataset is not available locally."
        ),
        "",
        (
            "Paper methodology pointer: "
            f"{payload['paper_pointer']}"
        ),
        "",
        (
            "Table 6 is a linear-kernel diagnostic. It reports the mean, "
            "standard deviation, and variance of the raw PCA-q linear kernel "
            "on a subsampled training set sorted by class."
        ),
        "",
        (
            "Sorting by class changes only the display order of rows and "
            "columns. It does not change the reported kernel statistics."
        ),
        "",
        (
            "| Model | q | Samples | Class 0 | Class 1 | "
            "K_L mean | K_L std | K_L var |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]

    for row in summary_rows:
        lines.append(
            (
                "| {model} | {q} | {samples} | {class_0} | {class_1} | "
                "{mean} | {std} | {variance} |"
            ).format(
                model=row[
                    "model_display"
                ],
                q=row["q"],
                samples=row[
                    "subsample_size"
                ],
                class_0=row[
                    "subsample_class_0"
                ],
                class_1=row[
                    "subsample_class_1"
                ],
                mean=format_float(
                    row["k_l_mean"]
                ),
                std=format_float(
                    row["k_l_std"]
                ),
                variance=format_float(
                    row["k_l_var"]
                ),
            )
        )

    lines.extend(
        [
            "",
            (
                "The identity K_L = X X^T is evaluated on the selected "
                "processed training samples."
            ),
            "",
            (
                "These statistics describe the complete kernel matrix, "
                "including its diagonal."
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
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix for the selected source."""
    if source == "synthetic":
        return "synthetic_surrogate_table6"

    if source == "synthetic_file":
        return "synthetic_file_table6"

    return "real_table6"


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
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--subsample-size",
        type=int,
        default=200,
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
) -> None:
    """Validate command-line arguments."""
    if args.subsample_size <= 0:
        raise ValueError(
            "--subsample-size must be positive."
        )

    if args.n_samples <= 0:
        raise ValueError(
            "--n-samples must be positive."
        )

    if args.ambient_dim <= 0:
        raise ValueError(
            "--ambient-dim must be positive."
        )

    if args.latent_dim <= 0:
        raise ValueError(
            "--latent-dim must be positive."
        )

    if not 0.0 < args.minority_frac < 1.0:
        raise ValueError(
            "--minority-frac must be in the interval (0, 1)."
        )

    if args.signal < 0.0:
        raise ValueError(
            "--signal must be non-negative."
        )

    if args.noise < 0.0:
        raise ValueError(
            "--noise must be non-negative."
        )

    if args.source in {
        "synthetic_file",
        "real",
    }:
        if args.data_root is None:
            raise ValueError(
                f"--data-root is required for source={args.source!r}."
            )

        if not args.data_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist: {args.data_root}"
            )

    if args.source == "synthetic_file":
        index_path = (
            args.data_root
            / "synthetic_dataset_index.json"
        )

        if not index_path.is_file():
            raise FileNotFoundError(
                "Synthetic dataset index not found: "
                f"{index_path}"
            )


def main() -> None:
    """Compute and write the Table 6-style artifact."""
    args = parse_args()

    validate_args(
        args
    )

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for model, q in TABLE6_CONFIGS:
        row = compute_table6_row(
            source=args.source,
            model=model,
            q=q,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
            subsample_size=(
                args.subsample_size
            ),
        )

        summary_rows.append(
            row
        )

        print(
            f"[table6] model={model} "
            f"q={q} seed={args.seed}",
            flush=True,
        )

    prefix = (
        args.output_prefix
        or default_prefix(
            args.source
        )
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.results_dir
        / f"{prefix}_summary.csv"
    )

    json_path = (
        args.results_dir
        / f"{prefix}.json"
    )

    markdown_path = (
        args.results_dir
        / f"{prefix}.md"
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 6",
        "paper_pointer": (
            PAPER_TABLE6_POINTER
        ),
        "paths": {
            "summary_csv": str(
                summary_path
            ),
            "json": str(
                json_path
            ),
            "markdown": str(
                markdown_path
            ),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": (
                args.source != "real"
            ),
            "synthetic_spec": (
                asdict(synthetic)
                if args.source
                == "synthetic"
                else None
            ),
            "data_root": (
                str(args.data_root)
                if args.data_root
                else None
            ),
            "seed": args.seed,
            "subsample_size": (
                args.subsample_size
            ),
            "split": (
                "80/10/10 stratified via "
                "lib.svm_pipeline.split_indices"
            ),
            "preprocessing": (
                "StandardScaler fitted on train, PCA fitted on train, "
                "MinMaxScaler fitted on train with output range [-1, 1]"
            ),
            "subsample_selection": (
                "Seeded random selection without replacement from the "
                "processed training split, followed by stable class sorting."
            ),
            "kernel": (
                "Raw linear kernel X_subset @ X_subset.T"
            ),
            "kernel_normalization": "none",
            "statistics_scope": (
                "All kernel entries, including the diagonal."
            ),
            "table6_configs": list(
                TABLE6_CONFIGS
            ),
        },
        "summary_rows": (
            summary_rows
        ),
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

    console_summary = {
        "rows": len(
            summary_rows
        ),
        "seed": args.seed,
        "subsample_size": (
            args.subsample_size
        ),
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
        )
    )

    print(
        f"Wrote {summary_path}"
    )

    print(
        f"Wrote {json_path}"
    )

    print(
        f"Wrote {markdown_path}"
    )


if __name__ == "__main__":
    main()
