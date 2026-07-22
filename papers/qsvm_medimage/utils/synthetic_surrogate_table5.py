#!/usr/bin/env python3
"""Compute a Table 5-style kernel effective-rank diagnostic.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 5 diagnostic shape:

- PCA explained variance;
- linear-kernel positive rank;
- linear-kernel effective rank;
- quantum fidelity-kernel effective rank.

The linear-kernel non-zero eigenvalues are computed from ``X_train.T @
X_train``. These are equal to the non-zero eigenvalues of the complete linear
Gram matrix ``X_train @ X_train.T``.

The quantum fidelity kernel is computed on the complete training split.

Kernel effective rank is invariant under multiplication of the complete
kernel by a positive scalar. Therefore, raw and trace-normalized versions of
the same kernel have the same effective rank.
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


from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import (  # noqa: E402
    preprocess,
    split_indices,
)
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
)

PAPER_TABLE5_POINTER = (
    "https://"
    "arxiv.org/html/2604.24597v1#S4.T5"
)

TABLE5_CONFIGS = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("medsiglip-448", 11),
    ("medsiglip-448", 16),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
)

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)


def validate_square_matrix(
    matrix: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate and symmetrize a square numeric matrix."""
    array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional."
        )

    if array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} must be square."
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return 0.5 * (
        array + array.T
    )


def positive_rank_from_eigenvalues(
    eigenvalues: np.ndarray,
) -> int:
    """Count eigenvalues above a scale-dependent numerical tolerance."""
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "eigenvalues must be one-dimensional."
        )

    if values.size == 0:
        return 0

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "eigenvalues contain non-finite values."
        )

    maximum_absolute_value = float(
        np.max(
            np.abs(values)
        )
    )

    tolerance = max(
        1e-12,
        1e-10 * maximum_absolute_value,
    )

    return int(
        np.sum(
            values > tolerance
        )
    )


def effective_rank_from_eigenvalues(
    eigenvalues: np.ndarray,
) -> float:
    """Compute Shannon effective rank from kernel eigenvalues."""
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "eigenvalues must be one-dimensional."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "eigenvalues contain non-finite values."
        )

    values = np.maximum(
        values,
        0.0,
    )

    eigenvalue_sum = float(
        np.sum(values)
    )

    if eigenvalue_sum <= 0.0:
        raise ValueError(
            "Eigenvalues have no positive mass."
        )

    probabilities = (
        values / eigenvalue_sum
    )

    probabilities = probabilities[
        probabilities > 1e-15
    ]

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    return float(
        np.exp(entropy)
    )


def psd_eigenvalues(
    matrix: np.ndarray,
    name: str,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return non-negative eigenvalues of a positive-semidefinite matrix."""
    if tolerance < 0.0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    symmetric_matrix = validate_square_matrix(
        matrix,
        name,
    )

    eigenvalues = np.linalg.eigvalsh(
        symmetric_matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"{name} is not positive semidefinite within tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}. "
            f"Tolerance: {tolerance:.3e}."
        )

    return np.maximum(
        eigenvalues,
        0.0,
    )


def linear_kernel_eigenvalues(
    X_train: np.ndarray,
) -> np.ndarray:
    """Return the non-zero spectrum of the linear training kernel.

    The non-zero eigenvalues of X @ X.T are equal to the non-zero
    eigenvalues of X.T @ X. After PCA-q, the latter matrix has shape q by q.
    """
    features = np.asarray(
        X_train,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "X_train must be two-dimensional."
        )

    if features.shape[0] == 0:
        raise ValueError(
            "X_train must contain at least one sample."
        )

    if features.shape[1] == 0:
        raise ValueError(
            "X_train must contain at least one feature."
        )

    if not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            "X_train contains non-finite values."
        )

    small_gram_matrix = (
        features.T @ features
    )

    return psd_eigenvalues(
        small_gram_matrix,
        "linear_small_gram_matrix",
    )


def quantum_kernel_eigenvalues(
    X_train: np.ndarray,
) -> np.ndarray:
    """Compute the complete quantum fidelity-kernel spectrum."""
    quantum_kernel = fidelity_kernel(
        X_train
    )

    return psd_eigenvalues(
        quantum_kernel,
        "quantum_fidelity_kernel",
    )


def compute_table5_row(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> dict[str, object]:
    """Compute the Table 5 spectral diagnostics for one model and q."""
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

    linear_eigenvalues = linear_kernel_eigenvalues(
        X_train
    )

    quantum_eigenvalues = quantum_kernel_eigenvalues(
        X_train
    )

    linear_positive_rank = (
        positive_rank_from_eigenvalues(
            linear_eigenvalues
        )
    )

    if linear_positive_rank > q:
        raise RuntimeError(
            "Linear-kernel positive rank exceeds the PCA dimension. "
            f"Positive rank: {linear_positive_rank}. q: {q}."
        )

    quantum_positive_rank = (
        positive_rank_from_eigenvalues(
            quantum_eigenvalues
        )
    )

    return {
        "source": source,
        "synthetic_surrogate": (
            source != "real"
        ),
        "model": model,
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
        "pca_variance_percent": float(
            100.0
            * explained_variance_ratio
        ),
        "linear_kernel_positive_rank": (
            linear_positive_rank
        ),
        "linear_kernel_rank_upper_bound": q,
        "linear_kernel_rank_valid": bool(
            linear_positive_rank <= q
        ),
        "linear_kernel_effective_rank": (
            effective_rank_from_eigenvalues(
                linear_eigenvalues
            )
        ),
        "quantum_kernel_positive_rank": (
            quantum_positive_rank
        ),
        "quantum_kernel_effective_rank": (
            effective_rank_from_eigenvalues(
                quantum_eigenvalues
            )
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the Table 5 summary rows to CSV."""
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
    return f"{value:.3f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 5 artifact."""
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
        "# Synthetic surrogate Table 5 pipeline",
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
            "Table 5 is a spectral diagnostic. It compares the "
            "PCA-compressed linear kernel with the quantum fidelity kernel "
            "after the same preprocessing."
        ),
        "",
        (
            "A larger positive rank or effective rank does not by itself "
            "establish better classification."
        ),
        "",
        (
            "| Model | q | PCA variance % | Linear rank | "
            "Linear eff. rank | Quantum rank | Quantum eff. rank |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]

    for row in summary_rows:
        lines.append(
            (
                "| {model} | {q} | {pca_var} | {linear_rank} | "
                "{linear_eff} | {quantum_rank} | {quantum_eff} |"
            ).format(
                model=row["model"],
                q=row["q"],
                pca_var=format_float(
                    row[
                        "pca_variance_percent"
                    ]
                ),
                linear_rank=row[
                    "linear_kernel_positive_rank"
                ],
                linear_eff=format_float(
                    row[
                        "linear_kernel_effective_rank"
                    ]
                ),
                quantum_rank=row[
                    "quantum_kernel_positive_rank"
                ],
                quantum_eff=format_float(
                    row[
                        "quantum_kernel_effective_rank"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            (
                "The linear-kernel rank must not exceed q after PCA-q. "
                "The quantum fidelity-kernel rank is not restricted by q."
            ),
            "",
            (
                "The reported effective ranks are computed from raw kernels. "
                "Trace normalization would not change effective rank because "
                "it multiplies every eigenvalue by the same positive scalar."
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
        return "synthetic_surrogate_table5"

    if source == "synthetic_file":
        return "synthetic_file_table5"

    return "real_table5"


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
    """Compute and write the Table 5-style artifact."""
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

    for model, q in TABLE5_CONFIGS:
        row = compute_table5_row(
            source=args.source,
            model=model,
            q=q,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )

        summary_rows.append(
            row
        )

        print(
            f"[table5] model={model} "
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
        "paper_table": "Table 5",
        "paper_pointer": (
            PAPER_TABLE5_POINTER
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
            "split": (
                "80/10/10 stratified via "
                "lib.svm_pipeline.split_indices"
            ),
            "preprocessing": (
                "StandardScaler fitted on train, PCA fitted on train, "
                "MinMaxScaler fitted on train with output range [-1, 1]"
            ),
            "kernel_scaling": (
                "Effective rank is computed from raw kernels. "
                "Positive scalar normalization does not change it."
            ),
            "linear_spectrum": (
                "Computed from X_train.T @ X_train."
            ),
            "quantum_spectrum": (
                "Computed from the complete training fidelity kernel."
            ),
            "table5_configs": list(
                TABLE5_CONFIGS
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
