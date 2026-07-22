#!/usr/bin/env python3
"""Generate a Figure 2-style linear-kernel eigenspectrum artifact.

Figure 2 studies the eigenspectrum of the linear kernel after PCA to q
dimensions.

For a processed training matrix X with shape (n_train, q), the linear kernel is:

    K_linear = X @ X.T

Its algebraic rank satisfies:

    rank(K_linear) <= q

The non-zero eigenvalues of X @ X.T are exactly the non-zero eigenvalues of
X.T @ X. This script therefore diagonalizes the smaller q-by-q matrix and pads
the spectrum with zeros up to n_train entries.

This avoids constructing and diagonalizing the complete n_train-by-n_train
kernel only to recover at most q non-zero eigenvalues.

When synthetic data are used, the artifact reproduces the diagnostic structure
of Figure 2, not the paper's numerical result on the inaccessible
MIMIC-CXR-derived embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
UTILS_ROOT = Path(__file__).resolve().parent

for root in (
    PROJECT_ROOT,
    REPRO_ROOT,
    UTILS_ROOT,
):
    root_string = str(root)

    if root_string not in sys.path:
        sys.path.insert(0, root_string)


from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
)

PAPER_FIGURE2_POINTER = (
    "https://arxiv.org/html/2604.24597v1#S4.F2"
)

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)


def effective_rank_from_eigenvalues(
    eigenvalues: np.ndarray,
) -> float:
    """Compute Shannon effective rank from non-negative eigenvalues.

    For eigenvalues lambda_i, define:

        p_i = lambda_i / sum_j lambda_j

    The effective rank is:

        exp(-sum_i p_i * log(p_i))
    """
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "eigenvalues must be a one-dimensional array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "eigenvalues contain non-finite values."
        )

    values = np.maximum(
        values,
        0.0,
    )

    total = float(
        np.sum(values)
    )

    if total <= 0.0:
        raise ValueError(
            "Eigenvalues have no positive mass."
        )

    probabilities = values / total
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


def count_positive_eigenvalues(
    eigenvalues: np.ndarray,
    tolerance: float | None = None,
) -> int:
    """Count eigenvalues above a scale-dependent numerical tolerance."""
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "eigenvalues must be a one-dimensional array."
        )

    if values.size == 0:
        return 0

    if tolerance is None:
        maximum_absolute_value = float(
            np.max(
                np.abs(values)
            )
        )

        tolerance = max(
            1e-12,
            1e-10 * maximum_absolute_value,
        )

    if tolerance < 0.0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    return int(
        np.sum(values > tolerance)
    )


def linear_kernel_eigenvalues(
    X_train: np.ndarray,
) -> np.ndarray:
    """Return the complete linear-kernel spectrum in descending order.

    The non-zero eigenvalues are obtained from X.T @ X. Zeros are then added
    so that the returned array has one entry per training sample.
    """
    features = np.asarray(
        X_train,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "X_train must be a two-dimensional feature matrix."
        )

    if features.shape[0] == 0:
        raise ValueError(
            "X_train must contain at least one sample."
        )

    if features.shape[1] == 0:
        raise ValueError(
            "X_train must contain at least one feature."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "X_train contains non-finite values."
        )

    small_gram_matrix = (
        features.T @ features
    )

    nonzero_spectrum = np.linalg.eigvalsh(
        small_gram_matrix
    )

    minimum_eigenvalue = float(
        np.min(nonzero_spectrum)
    )

    largest_absolute_value = float(
        np.max(
            np.abs(nonzero_spectrum)
        )
    )

    numerical_tolerance = max(
        1e-10,
        1e-10 * largest_absolute_value,
    )

    if minimum_eigenvalue < -numerical_tolerance:
        raise ValueError(
            "The linear kernel is not positive semidefinite within "
            "numerical tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}."
        )

    nonzero_spectrum = np.maximum(
        nonzero_spectrum,
        0.0,
    )

    nonzero_spectrum = np.sort(
        nonzero_spectrum
    )[::-1]

    zero_count = max(
        features.shape[0]
        - len(nonzero_spectrum),
        0,
    )

    complete_spectrum = np.concatenate(
        [
            nonzero_spectrum,
            np.zeros(
                zero_count,
                dtype=np.float64,
            ),
        ]
    )

    return complete_spectrum


def normalize_eigenvalues(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Normalize eigenvalues so that their sum equals one."""
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    total = float(
        np.sum(values)
    )

    if total <= 0.0:
        raise ValueError(
            "Cannot normalize an eigenspectrum with non-positive sum."
        )

    return values / total


def eigenvalue_rows(
    raw_eigenvalues: np.ndarray,
) -> list[dict[str, object]]:
    """Convert one eigenspectrum into CSV rows."""
    values = np.asarray(
        raw_eigenvalues,
        dtype=np.float64,
    )

    normalized_values = normalize_eigenvalues(
        values
    )

    cumulative_mass = np.cumsum(
        normalized_values
    )

    return [
        {
            "eigenvalue_index": index,
            "eigenvalue_count": index + 1,
            "raw_eigenvalue": float(
                values[index]
            ),
            "normalized_eigenvalue": float(
                normalized_values[index]
            ),
            "cumulative_spectral_mass": float(
                cumulative_mass[index]
            ),
        }
        for index in range(
            len(values)
        )
    ]


def components_needed(
    cumulative_mass: np.ndarray,
    threshold: float,
) -> int | None:
    """Return the first eigenvalue count reaching a mass threshold."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must be in (0, 1]."
        )

    hits = np.flatnonzero(
        cumulative_mass >= threshold
    )

    if hits.size == 0:
        return None

    return int(
        hits[0] + 1
    )


def compute_figure2(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> dict[str, object]:
    """Compute the Figure 2-style linear-kernel eigenspectrum."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
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
        _,
        explained_variance_ratio,
    ) = preprocess(
        X[training_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    raw_eigenvalues = linear_kernel_eigenvalues(
        X_train
    )

    normalized_eigenvalues = normalize_eigenvalues(
        raw_eigenvalues
    )

    cumulative_mass = np.cumsum(
        normalized_eigenvalues
    )

    positive_rank = count_positive_eigenvalues(
        raw_eigenvalues
    )

    rank_validation = bool(
        positive_rank <= q
    )

    if not rank_validation:
        raise RuntimeError(
            "The linear-kernel rank exceeds the PCA dimension: "
            f"positive_rank={positive_rank}, q={q}."
        )

    return {
        "source": source,
        "synthetic_surrogate": (
            source != "real"
        ),
        "model": model,
        "q": q,
        "seed": seed,
        "raw_embedding_dimension": int(
            X.shape[1]
        ),
        "train_samples": int(
            len(training_indices)
        ),
        "validation_samples": int(
            len(validation_indices)
        ),
        "test_samples": int(
            len(test_indices)
        ),
        "pca_variance_percent": (
            100.0
            * float(
                explained_variance_ratio
            )
        ),
        "positive_rank": positive_rank,
        "rank_upper_bound": q,
        "rank_validation": rank_validation,
        "effective_rank": (
            effective_rank_from_eigenvalues(
                raw_eigenvalues
            )
        ),
        "largest_eigenvalue_raw": float(
            raw_eigenvalues[0]
        ),
        "largest_eigenvalue_normalized": float(
            normalized_eigenvalues[0]
        ),
        "trace_raw": float(
            np.sum(raw_eigenvalues)
        ),
        "eigenvalues_for_90_percent": (
            components_needed(
                cumulative_mass,
                0.90,
            )
        ),
        "eigenvalues_for_95_percent": (
            components_needed(
                cumulative_mass,
                0.95,
            )
        ),
        "raw_eigenvalues": (
            raw_eigenvalues
        ),
        "normalized_eigenvalues": (
            normalized_eigenvalues
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write eigenspectrum rows to CSV."""
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


def save_plot(
    path: Path,
    *,
    figure2: dict[str, object],
) -> None:
    """Save the eigenspectrum and cumulative spectral-mass panels."""
    raw_eigenvalues = np.asarray(
        figure2["raw_eigenvalues"],
        dtype=np.float64,
    )

    normalized_eigenvalues = np.asarray(
        figure2["normalized_eigenvalues"],
        dtype=np.float64,
    )

    positive_rank = int(
        figure2["positive_rank"]
    )

    plot_count = min(
        200,
        len(raw_eigenvalues),
    )

    eigenvalue_indices = np.arange(
        plot_count
    )

    plotted_eigenvalues = raw_eigenvalues[
        :plot_count
    ].copy()

    plotted_eigenvalues[
        plotted_eigenvalues <= 1e-12
    ] = np.nan

    cumulative_mass = np.concatenate(
        [
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(
                normalized_eigenvalues
            ),
        ]
    )

    cumulative_count = min(
        200,
        len(raw_eigenvalues),
    )

    eigenvalue_counts = np.arange(
        cumulative_count + 1
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
    )

    spectrum_axis = axes[0]

    spectrum_axis.semilogy(
        eigenvalue_indices,
        plotted_eigenvalues,
        marker="o",
        markersize=4,
        color="#1f77b4",
        linewidth=2.0,
        label=(
            "Linear kernel "
            f"(effective rank "
            f"{figure2['effective_rank']:.2f})"
        ),
    )

    if 0 < positive_rank < plot_count:
        spectrum_axis.axvline(
            positive_rank - 0.5,
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label=(
                f"Positive rank = "
                f"{positive_rank}"
            ),
        )

    spectrum_axis.set_xlabel(
        "Eigenvalue index, descending"
    )

    spectrum_axis.set_ylabel(
        "Raw eigenvalue, logarithmic scale"
    )

    spectrum_axis.set_xlim(
        0,
        max(
            plot_count - 1,
            1,
        ),
    )

    finite_positive_values = plotted_eigenvalues[
        np.isfinite(
            plotted_eigenvalues
        )
    ]

    if finite_positive_values.size:
        largest_value = float(
            finite_positive_values[0]
        )

        smallest_value = float(
            np.min(
                finite_positive_values
            )
        )

        spectrum_axis.set_ylim(
            max(
                smallest_value * 0.5,
                largest_value * 1e-10,
            ),
            largest_value * 1.3,
        )

    spectrum_axis.grid(
        True,
        alpha=0.3,
    )

    spectrum_axis.legend(
        loc="upper right"
    )

    cumulative_axis = axes[1]

    cumulative_axis.plot(
        eigenvalue_counts,
        cumulative_mass[
            : cumulative_count + 1
        ],
        color="#1f77b4",
        linewidth=2.0,
        label=(
            f"{figure2['model']}, "
            f"q={figure2['q']}"
        ),
    )

    cumulative_axis.axhline(
        0.90,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        label="90% spectral mass",
    )

    cumulative_axis.axhline(
        0.95,
        color="gray",
        linestyle=":",
        linewidth=0.9,
        label="95% spectral mass",
    )

    cumulative_axis.set_xlabel(
        "Number of eigenvalues"
    )

    cumulative_axis.set_ylabel(
        "Cumulative normalized eigenvalue mass"
    )

    cumulative_axis.set_xlim(
        0,
        max(
            cumulative_count,
            1,
        ),
    )

    cumulative_axis.set_ylim(
        0.0,
        1.02,
    )

    cumulative_axis.grid(
        True,
        alpha=0.3,
    )

    cumulative_axis.legend(
        loc="lower right"
    )

    figure.suptitle(
        (
            "Linear-kernel eigenspectrum after PCA "
            f"to q={figure2['q']}"
        ),
        fontsize=11,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Figure 2 artifact description."""
    summary = payload[
        "summary"
    ]

    paths = payload[
        "paths"
    ]

    if not isinstance(
        summary,
        dict,
    ):
        raise TypeError(
            "payload['summary'] must be a dictionary."
        )

    if not isinstance(
        paths,
        dict,
    ):
        raise TypeError(
            "payload['paths'] must be a dictionary."
        )

    image_name = Path(
        paths["png"]
    ).name

    lines = [
        "# Figure 2-style linear-kernel eigenspectrum",
        "",
        (
            "This artifact analyses the linear kernel after PCA. "
            "Results generated from synthetic data reproduce the diagnostic "
            "structure of Figure 2, not the paper's numerical result on the "
            "inaccessible medical embeddings."
        ),
        "",
        (
            "The non-zero eigenvalues of X X^T equal those of X^T X. "
            "After PCA to q dimensions, the linear-kernel rank cannot "
            "exceed q."
        ),
        "",
        (
            f"Paper methodology pointer: "
            f"{payload['paper_pointer']}"
        ),
        "",
        (
            f"![Figure 2-style linear-kernel eigenspectrum]"
            f"({image_name})"
        ),
        "",
        "Summary:",
        "",
        f"- source: `{summary['source']}`",
        f"- model: `{summary['model']}`",
        f"- q: `{summary['q']}`",
        f"- seed: `{summary['seed']}`",
        (
            "- raw embedding dimension: "
            f"`{summary['raw_embedding_dimension']}`"
        ),
        (
            f"- training samples: "
            f"`{summary['train_samples']}`"
        ),
        (
            "- PCA explained variance: "
            f"`{summary['pca_variance_percent']:.3f}%`"
        ),
        (
            f"- positive rank: "
            f"`{summary['positive_rank']}`"
        ),
        (
            f"- rank upper bound: "
            f"`{summary['rank_upper_bound']}`"
        ),
        (
            "- rank validation: "
            f"`{summary['rank_validation']}`"
        ),
        (
            "- effective rank: "
            f"`{summary['effective_rank']:.3f}`"
        ),
        (
            "- largest raw eigenvalue: "
            f"`{summary['largest_eigenvalue_raw']:.3f}`"
        ),
        (
            "- eigenvalues needed for 90% spectral mass: "
            f"`{summary['eigenvalues_for_90_percent']}`"
        ),
        (
            "- eigenvalues needed for 95% spectral mass: "
            f"`{summary['eigenvalues_for_95_percent']}`"
        ),
        "",
        "Interpretation:",
        "",
        (
            "- The positive rank is an algebraic property of the "
            "PCA-compressed linear kernel."
        ),
        (
            "- Effective rank describes how eigenvalue mass is distributed "
            "among the positive directions."
        ),
        (
            "- A low algebraic or effective rank does not by itself prove "
            "classifier collapse."
        ),
        (
            "- Synthetic PCA statistics may be close to the paper because "
            "the generator was calibrated against selected reported "
            "geometry statistics."
        ),
        "",
        "Data and protocol metadata:",
        "",
        "```json",
        json.dumps(
            payload["data"],
            indent=2,
            sort_keys=True,
        ),
        "```",
    ]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix associated with the data source."""
    if source == "synthetic":
        return "synthetic_surrogate_figure2"

    if source == "synthetic_file":
        return "synthetic_file_figure2"

    return "real_figure2"


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
        "--model",
        choices=MODEL_NAMES,
        default="medsiglip-448",
    )

    parser.add_argument(
        "--q",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Parameters used only for in-memory synthetic generation.
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
    """Validate command-line argument consistency."""
    if args.q <= 0:
        raise ValueError(
            "--q must be positive."
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
            "--minority-frac must be in (0, 1)."
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
    """Generate the Figure 2-style artifact."""
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

    figure2 = compute_figure2(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
    )

    prefix = (
        args.output_prefix
        or default_prefix(args.source)
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        args.results_dir
        / f"{prefix}.png"
    )

    csv_path = (
        args.results_dir
        / f"{prefix}_eigenvalues.csv"
    )

    json_path = (
        args.results_dir
        / f"{prefix}.json"
    )

    markdown_path = (
        args.results_dir
        / f"{prefix}.md"
    )

    raw_eigenvalues = np.asarray(
        figure2["raw_eigenvalues"],
        dtype=np.float64,
    )

    rows = eigenvalue_rows(
        raw_eigenvalues
    )

    write_csv(
        csv_path,
        rows,
    )

    save_plot(
        png_path,
        figure2=figure2,
    )

    summary = {
        key: value
        for key, value in figure2.items()
        if key
        not in {
            "raw_eigenvalues",
            "normalized_eigenvalues",
        }
    }

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 2",
        "paper_pointer": (
            PAPER_FIGURE2_POINTER
        ),
        "scope": {
            "kernel": "linear",
            "classifier_trained": False,
            "synthetic_result_is_paper_result": False,
            "generator_calibrated_on_selected_geometry": (
                args.source != "real"
            ),
        },
        "paths": {
            "png": str(
                png_path
            ),
            "eigenvalues_csv": str(
                csv_path
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
                if args.source == "synthetic"
                else None
            ),
            "data_root": (
                str(args.data_root)
                if args.data_root
                else None
            ),
            "split": (
                "80/10/10 stratified via "
                "lib.svm_pipeline.split_indices"
            ),
            "preprocessing": (
                "StandardScaler fitted on train, PCA fitted on train, "
                "MinMaxScaler[-1,1] fitted on train"
            ),
            "eigenspectrum_implementation": (
                "Non-zero spectrum computed from X.T @ X and padded "
                "with zeros to n_train entries."
            ),
        },
        "summary": summary,
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

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        f"Wrote {png_path}"
    )
    print(
        f"Wrote {csv_path}"
    )
    print(
        f"Wrote {json_path}"
    )
    print(
        f"Wrote {markdown_path}"
    )


if __name__ == "__main__":
    main()
