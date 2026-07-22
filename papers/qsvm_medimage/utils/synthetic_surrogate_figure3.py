#!/usr/bin/env python3
"""Generate a Figure 3-style quantum-versus-linear eigenspectrum artifact.

The artifact compares two kernels after identical preprocessing:

1. The manuscript quantum fidelity kernel.
2. The linear kernel after PCA to q dimensions.

The manuscript quantum feature map uses one Ry rotation per qubit followed by
a CNOT ring. The linear kernel is:

    K_linear = X @ X.T

The non-zero eigenvalues of X @ X.T are equal to the non-zero eigenvalues of
X.T @ X. The linear spectrum is therefore computed from the smaller q by q
matrix X.T @ X.

The complete quantum Gram matrix is diagonalized because the quantum-kernel
rank is not restricted to q.

When synthetic data are used, this artifact reproduces the diagnostic
structure of Figure 3. It does not reproduce the numerical result obtained
from the inaccessible MIMIC-CXR-derived embeddings.
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


from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
    parse_ints,
)

PAPER_FIGURE3_POINTER = (
    "https://arxiv.org/html/2604.24597v1#S4.F3"
)

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

QUANTUM_COLORS = {
    4: "#2166ac",
    6: "#4dac26",
}

LINEAR_COLORS = {
    4: "#e66101",
    6: "#d01c8b",
}


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
    """Count eigenvalues above a relative numerical tolerance."""
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


def normalize_eigenvalues(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Normalize eigenvalues so that their sum equals one."""
    values = np.asarray(
        eigenvalues,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "eigenvalues must be one-dimensional."
        )

    total = float(
        np.sum(values)
    )

    if total <= 0.0:
        raise ValueError(
            "Cannot normalize an eigenspectrum with non-positive sum."
        )

    return values / total


def components_needed(
    normalized_eigenvalues: np.ndarray,
    threshold: float,
) -> int | None:
    """Return the eigenvalue count needed to reach a spectral-mass threshold."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must be in the interval (0, 1]."
        )

    cumulative_mass = np.cumsum(
        normalized_eigenvalues
    )

    matching_indices = np.flatnonzero(
        cumulative_mass >= threshold
    )

    if matching_indices.size == 0:
        return None

    return int(
        matching_indices[0] + 1
    )


def validate_square_kernel(
    kernel: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate and symmetrize a square kernel matrix."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            f"{name} must be a square matrix."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return 0.5 * (
        matrix + matrix.T
    )


def sorted_psd_eigenvalues(
    matrix: np.ndarray,
    name: str,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return non-negative eigenvalues sorted in descending order."""
    square_matrix = validate_square_kernel(
        matrix,
        name,
    )

    eigenvalues = np.linalg.eigvalsh(
        square_matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    maximum_absolute_value = float(
        np.max(
            np.abs(eigenvalues)
        )
    )

    tolerance = max(
        1e-10,
        relative_tolerance
        * maximum_absolute_value,
    )

    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"{name} is not positive semidefinite within tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}. "
            f"Tolerance: {tolerance:.3e}."
        )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    return np.sort(
        eigenvalues
    )[::-1]


def quantum_kernel_eigenvalues(
    X_train: np.ndarray,
) -> np.ndarray:
    """Compute the complete quantum fidelity-kernel spectrum."""
    quantum_kernel = fidelity_kernel(
        X_train
    )

    return sorted_psd_eigenvalues(
        quantum_kernel,
        "quantum_kernel",
    )


def linear_kernel_eigenvalues(
    X_train: np.ndarray,
) -> np.ndarray:
    """Compute the complete linear-kernel spectrum efficiently.

    Non-zero eigenvalues are computed from X.T @ X. Zeros are appended so that
    the returned array contains one value per training sample.
    """
    features = np.asarray(
        X_train,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "X_train must be a two-dimensional matrix."
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

    nonzero_spectrum = sorted_psd_eigenvalues(
        small_gram_matrix,
        "linear_small_gram_matrix",
    )

    zero_count = max(
        features.shape[0]
        - nonzero_spectrum.size,
        0,
    )

    return np.concatenate(
        (
            nonzero_spectrum,
            np.zeros(
                zero_count,
                dtype=np.float64,
            ),
        )
    )


def spectrum_summary(
    raw_eigenvalues: np.ndarray,
) -> dict[str, object]:
    """Return normalized eigenvalues and spectral diagnostics."""
    raw_values = np.asarray(
        raw_eigenvalues,
        dtype=np.float64,
    )

    normalized_values = normalize_eigenvalues(
        raw_values
    )

    return {
        "raw_eigenvalues": raw_values,
        "normalized_eigenvalues": normalized_values,
        "positive_rank": count_positive_eigenvalues(
            raw_values
        ),
        "effective_rank": effective_rank_from_eigenvalues(
            raw_values
        ),
        "trace_raw": float(
            np.sum(raw_values)
        ),
        "lambda_max_raw": float(
            raw_values[0]
        ),
        "lambda_max_normalized": float(
            normalized_values[0]
        ),
        "eigenvalues_for_90_percent": components_needed(
            normalized_values,
            0.90,
        ),
        "eigenvalues_for_95_percent": components_needed(
            normalized_values,
            0.95,
        ),
        "eigenvalues_for_99_percent": components_needed(
            normalized_values,
            0.99,
        ),
    }


def merge_dictionaries(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, object]:
    """Return a new dictionary containing both input dictionaries."""
    merged = dict(
        first
    )

    merged.update(
        second
    )

    return merged


def compute_figure3_series(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> list[dict[str, object]]:
    """Compute quantum and linear eigenspectra for one q value."""
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

    quantum_spectrum = spectrum_summary(
        quantum_kernel_eigenvalues(
            X_train
        )
    )

    linear_spectrum = spectrum_summary(
        linear_kernel_eigenvalues(
            X_train
        )
    )

    linear_positive_rank = int(
        linear_spectrum[
            "positive_rank"
        ]
    )

    if linear_positive_rank > q:
        raise RuntimeError(
            "Linear-kernel positive rank exceeds the PCA dimension. "
            f"Positive rank: {linear_positive_rank}. q: {q}."
        )

    quantum_rank_upper_bound = min(
        len(training_indices),
        pow(4, q),
    )

    quantum_positive_rank = int(
        quantum_spectrum[
            "positive_rank"
        ]
    )

    shared = {
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
        "pca_variance_percent": float(
            100.0
            * explained_variance_ratio
        ),
    }

    quantum_row = dict(
        shared
    )

    quantum_row.update(
        {
            "kernel": "quantum",
            "kernel_definition": (
                "Manuscript Ry encoding with CNOT ring "
                "fidelity kernel"
            ),
            "rank_upper_bound": (
                quantum_rank_upper_bound
            ),
            "rank_validation": bool(
                quantum_positive_rank
                <= quantum_rank_upper_bound
            ),
        }
    )

    quantum_row.update(
        quantum_spectrum
    )

    linear_row = dict(
        shared
    )

    linear_row.update(
        {
            "kernel": "linear",
            "kernel_definition": (
                "Linear kernel X @ X.T after PCA-q"
            ),
            "rank_upper_bound": q,
            "rank_validation": bool(
                linear_positive_rank <= q
            ),
        }
    )

    linear_row.update(
        linear_spectrum
    )

    return [
        quantum_row,
        linear_row,
    ]


def eigenvalue_rows(
    series_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert all eigenspectra to CSV rows."""
    rows: list[
        dict[str, object]
    ] = []

    for series in series_rows:
        raw_values = np.asarray(
            series["raw_eigenvalues"],
            dtype=np.float64,
        )

        normalized_values = np.asarray(
            series["normalized_eigenvalues"],
            dtype=np.float64,
        )

        cumulative_mass = np.cumsum(
            normalized_values
        )

        for index in range(
            raw_values.size
        ):
            rows.append(
                {
                    "source": series["source"],
                    "model": series["model"],
                    "seed": series["seed"],
                    "q": series["q"],
                    "kernel": series["kernel"],
                    "eigenvalue_index": index,
                    "eigenvalue_count": index + 1,
                    "raw_eigenvalue": float(
                        raw_values[index]
                    ),
                    "normalized_eigenvalue": float(
                        normalized_values[index]
                    ),
                    "cumulative_spectral_mass": float(
                        cumulative_mass[index]
                    ),
                }
            )

    return rows


def cumulative_with_zero(
    normalized_eigenvalues: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return count and cumulative-mass arrays beginning at zero."""
    values = np.asarray(
        normalized_eigenvalues,
        dtype=np.float64,
    )

    eigenvalue_count = np.arange(
        values.size + 1
    )

    cumulative_mass = np.concatenate(
        (
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(
                values
            ),
        )
    )

    return (
        eigenvalue_count,
        cumulative_mass,
    )


def ordered_series(
    series_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return quantum series first, then linear series, ordered by q."""
    q_values = sorted(
        {
            int(row["q"])
            for row in series_rows
        }
    )

    ordered_rows: list[
        dict[str, object]
    ] = []

    for kernel_name in (
        "quantum",
        "linear",
    ):
        for q in q_values:
            matching_rows = [
                row
                for row in series_rows
                if (
                    row["kernel"]
                    == kernel_name
                    and int(row["q"])
                    == q
                )
            ]

            ordered_rows.extend(
                matching_rows
            )

    return ordered_rows


def plot_eigenvalue_decay(
    axis: plt.Axes,
    *,
    series_rows: list[dict[str, object]],
    maximum_index: int,
) -> None:
    """Plot normalized eigenvalue decay on a logarithmic scale."""
    actual_maximum_index = 1

    for series in ordered_series(
        series_rows
    ):
        q = int(
            series["q"]
        )

        eigenvalues = np.asarray(
            series["normalized_eigenvalues"],
            dtype=np.float64,
        )

        plot_count = min(
            maximum_index,
            eigenvalues.size,
        )

        actual_maximum_index = max(
            actual_maximum_index,
            plot_count,
        )

        x_values = np.arange(
            plot_count
        )

        y_values = eigenvalues[
            :plot_count
        ].copy()

        y_values[
            y_values <= 1e-14
        ] = np.nan

        if series["kernel"] == "quantum":
            color = QUANTUM_COLORS.get(
                q,
                "#2166ac",
            )

            label = (
                f"Quantum q={q} "
                f"(effective rank "
                f"{series['effective_rank']:.2f})"
            )

            axis.semilogy(
                x_values,
                y_values,
                color=color,
                linewidth=1.9,
                label=label,
            )

        else:
            color = LINEAR_COLORS.get(
                q,
                "#e66101",
            )

            label = (
                f"Linear q={q} "
                f"(effective rank "
                f"{series['effective_rank']:.2f})"
            )

            axis.semilogy(
                x_values,
                y_values,
                color=color,
                linewidth=2.6,
                linestyle="--",
                label=label,
            )

            positive_rank = int(
                series["positive_rank"]
            )

            if (
                positive_rank > 0
                and positive_rank <= plot_count
            ):
                rank_index = (
                    positive_rank - 1
                )

                rank_value = y_values[
                    rank_index
                ]

                if np.isfinite(
                    rank_value
                ):
                    axis.scatter(
                        [rank_index],
                        [rank_value],
                        marker="X",
                        color=color,
                        s=75,
                        zorder=4,
                    )

                axis.axvline(
                    positive_rank - 0.5,
                    color=color,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                )

    axis.set_xlabel(
        "Eigenvalue index, descending"
    )

    axis.set_ylabel(
        "Normalized eigenvalue, logarithmic scale"
    )

    axis.set_xlim(
        0,
        max(
            actual_maximum_index - 1,
            1,
        ),
    )

    axis.set_ylim(
        1e-12,
        1.05,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        fontsize=8.5,
        loc="lower left",
    )


def plot_cumulative_mass(
    axis: plt.Axes,
    *,
    series_rows: list[dict[str, object]],
    maximum_count: int,
) -> None:
    """Plot cumulative normalized eigenvalue mass."""
    actual_maximum_count = 1

    for series in ordered_series(
        series_rows
    ):
        q = int(
            series["q"]
        )

        eigenvalues = np.asarray(
            series["normalized_eigenvalues"],
            dtype=np.float64,
        )

        (
            eigenvalue_count,
            cumulative_mass,
        ) = cumulative_with_zero(
            eigenvalues
        )

        if series["kernel"] == "quantum":
            color = QUANTUM_COLORS.get(
                q,
                "#2166ac",
            )

            plot_count = min(
                maximum_count,
                eigenvalues.size,
            )

            actual_maximum_count = max(
                actual_maximum_count,
                plot_count,
            )

            axis.plot(
                eigenvalue_count[
                    : plot_count + 1
                ],
                cumulative_mass[
                    : plot_count + 1
                ],
                color=color,
                linewidth=1.9,
                label=f"Quantum q={q}",
            )

        else:
            color = LINEAR_COLORS.get(
                q,
                "#e66101",
            )

            positive_rank = int(
                series["positive_rank"]
            )

            plot_count = min(
                positive_rank,
                maximum_count,
            )

            actual_maximum_count = max(
                actual_maximum_count,
                plot_count,
            )

            axis.plot(
                eigenvalue_count[
                    : plot_count + 1
                ],
                cumulative_mass[
                    : plot_count + 1
                ],
                color=color,
                linewidth=2.6,
                linestyle="--",
                label=f"Linear q={q}",
            )

            if (
                positive_rank > 0
                and positive_rank
                <= maximum_count
            ):
                axis.scatter(
                    [positive_rank],
                    [
                        cumulative_mass[
                            positive_rank
                        ]
                    ],
                    marker="X",
                    color=color,
                    s=75,
                    zorder=4,
                )

    axis.axhline(
        0.90,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        label="90% spectral mass",
    )

    axis.axhline(
        0.99,
        color="gray",
        linestyle=":",
        linewidth=0.9,
        label="99% spectral mass",
    )

    axis.set_xlabel(
        "Number of eigenvalues"
    )

    axis.set_ylabel(
        "Cumulative normalized eigenvalue mass"
    )

    axis.set_xlim(
        0,
        max(
            actual_maximum_count,
            1,
        ),
    )

    axis.set_ylim(
        0.0,
        1.02,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        fontsize=8.5,
        loc="lower right",
    )


def save_plot(
    path: Path,
    *,
    series_rows: list[dict[str, object]],
    maximum_eigenvalue_index: int,
    maximum_cumulative_count: int,
) -> None:
    """Save the Figure 3-style two-panel plot."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    plot_eigenvalue_decay(
        axes[0],
        series_rows=series_rows,
        maximum_index=(
            maximum_eigenvalue_index
        ),
    )

    plot_cumulative_mass(
        axes[1],
        series_rows=series_rows,
        maximum_count=(
            maximum_cumulative_count
        ),
    )

    model = str(
        series_rows[0]["model"]
    )

    seed = int(
        series_rows[0]["seed"]
    )

    figure.suptitle(
        (
            "Quantum and linear kernel eigenspectra\n"
            f"{model}, seed={seed}"
        ),
        fontsize=11,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.93,
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


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Figure 3 description."""
    paths = payload[
        "paths"
    ]

    if not isinstance(
        paths,
        dict,
    ):
        raise TypeError(
            "payload paths must be a dictionary."
        )

    image_name = Path(
        paths["png"]
    ).name

    lines = [
        "# Figure 3-style quantum-versus-linear eigenspectra",
        "",
        (
            "This artifact compares the manuscript quantum fidelity kernel "
            "with the linear kernel after identical preprocessing."
        ),
        "",
        (
            "Results generated from synthetic data reproduce the diagnostic "
            "structure of Figure 3, not the paper's numerical result on the "
            "inaccessible medical embeddings."
        ),
        "",
        (
            f"Paper methodology pointer: "
            f"{payload['paper_pointer']}"
        ),
        "",
        (
            "![Figure 3-style eigenspectrum comparison]"
            f"({image_name})"
        ),
        "",
        "Summary:",
        "",
    ]

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

    for summary in summary_rows:
        lines.append(

                f"- {summary['kernel']} q={summary['q']}: "
                f"effective rank `{summary['effective_rank']:.3f}`, "
                f"positive rank `{summary['positive_rank']}`, "
                f"rank bound `{summary['rank_upper_bound']}`, "
                f"90% mass in "
                f"`{summary['eigenvalues_for_90_percent']}` "
                "eigenvalues"

        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            (
                "- Positive rank counts numerically non-zero spectral "
                "directions."
            ),
            (
                "- Effective rank measures how evenly eigenvalue mass is "
                "distributed across those directions."
            ),
            (
                "- A kernel can have a high positive rank but a low "
                "effective rank when a small number of eigenvalues dominate."
            ),
            (
                "- A higher effective rank does not by itself imply better "
                "classification."
            ),
            (
                "- Synthetic results reflect both the calibrated generator "
                "geometry and the selected feature map."
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
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix associated with the data source."""
    if source == "synthetic":
        return "synthetic_surrogate_figure3"

    if source == "synthetic_file":
        return "synthetic_file_figure3"

    return "real_figure3"


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
        "--q-values",
        default="4,6",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-eigenvalue-index",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max-cumulative-count",
        type=int,
        default=50,
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
    q_values: list[int],
) -> None:
    """Validate command-line argument consistency."""
    if not q_values:
        raise ValueError(
            "At least one q value is required."
        )

    if any(
        q <= 0
        for q in q_values
    ):
        raise ValueError(
            "All q values must be positive."
        )

    if args.max_eigenvalue_index <= 0:
        raise ValueError(
            "--max-eigenvalue-index must be positive."
        )

    if args.max_cumulative_count <= 0:
        raise ValueError(
            "--max-cumulative-count must be positive."
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
    """Generate the Figure 3-style artifact."""
    args = parse_args()

    q_values = parse_ints(
        args.q_values
    )

    validate_args(
        args,
        q_values,
    )

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    series_rows: list[
        dict[str, object]
    ] = []

    for q in q_values:
        computed_rows = compute_figure3_series(
            source=args.source,
            model=args.model,
            q=q,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )

        series_rows.extend(
            computed_rows
        )

        print(
            f"Computed model={args.model} "
            f"seed={args.seed} q={q}",
            flush=True,
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

    write_csv(
        csv_path,
        eigenvalue_rows(
            series_rows
        ),
    )

    save_plot(
        png_path,
        series_rows=series_rows,
        maximum_eigenvalue_index=(
            args.max_eigenvalue_index
        ),
        maximum_cumulative_count=(
            args.max_cumulative_count
        ),
    )

    excluded_keys = {
        "raw_eigenvalues",
        "normalized_eigenvalues",
    }

    summary_rows: list[
        dict[str, object]
    ] = []

    for row in series_rows:
        summary = {
            key: value
            for key, value in row.items()
            if key not in excluded_keys
        }

        summary_rows.append(
            summary
        )

    payload: dict[
        str,
        object,
    ] = {
        "artifact": prefix,
        "paper_figure": "Figure 3",
        "paper_pointer": (
            PAPER_FIGURE3_POINTER
        ),
        "scope": {
            "classifier_trained": False,
            "compares_quantum_and_linear_kernels": True,
            "quantum_feature_map": (
                "Manuscript Ry encoding with CNOT ring"
            ),
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
            "model": args.model,
            "q_values": q_values,
            "seed": args.seed,
            "split": (
                "80/10/10 stratified via "
                "lib.svm_pipeline.split_indices"
            ),
            "preprocessing": (
                "StandardScaler fitted on train, PCA fitted on train, "
                "MinMaxScaler[-1,1] fitted on train"
            ),
            "linear_spectrum_implementation": (
                "Non-zero spectrum computed from X.T @ X and padded "
                "with zeros to n_train entries."
            ),
            "quantum_spectrum_implementation": (
                "Complete training fidelity Gram matrix."
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

    print(
        json.dumps(
            summary_rows,
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
