#!/usr/bin/env python3
"""Generate a Figure 4-style quantum fidelity-kernel heatmap.

The default experiment uses:

- MedSigLIP embeddings;
- q=6 PCA components;
- seed 0;
- 200 stratified training samples;
- samples sorted by class label;
- trace normalization for visualization.

The preprocessing pipeline is fitted on the complete training split:

    StandardScaler
    PCA(q)
    MinMaxScaler[-1, 1]

A stratified subset is selected after preprocessing. The selected samples are
then sorted by class label for visualization.

Sorting samples by class creates visible block boundaries in the matrix layout.
The boundaries do not prove that the kernel separates the classes. The script
therefore reports within-class and between-class similarity statistics.

When synthetic data are used, this artifact reproduces the diagnostic structure
of Figure 4. It does not reproduce the numerical result obtained from the
inaccessible medical embeddings.
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
from sklearn.model_selection import train_test_split

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
        sys.path.insert(
            0,
            root_string,
        )


from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
)

PAPER_FIGURE4_POINTER = (
    "https://arxiv.org/html/2604.24597v1#S4.F4"
)

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)


def stratified_sample_indices(
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Return an exact-size stratified subset of row indices."""
    labels = np.asarray(
        labels,
        dtype=int,
    )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if labels.size == 0:
        raise ValueError(
            "labels must not be empty."
        )

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    all_indices = np.arange(
        len(labels)
    )

    if sample_count >= len(labels):
        return all_indices

    unique_labels, class_counts = np.unique(
        labels,
        return_counts=True,
    )

    if unique_labels.size < 2:
        raise ValueError(
            "Stratified sampling requires at least two classes."
        )

    if sample_count < unique_labels.size:
        raise ValueError(
            "sample_count must be at least the number of classes."
        )

    if np.any(class_counts < 2):
        raise ValueError(
            "Every class must contain at least two samples."
        )

    selected_indices, _ = train_test_split(
        all_indices,
        train_size=sample_count,
        random_state=seed,
        stratify=labels,
    )

    return np.sort(
        selected_indices
    )


def select_samples_sorted_by_class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
]:
    """Select a stratified subset and sort it by class label."""
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
            "X_train must be a two-dimensional matrix."
        )

    if labels.ndim != 1:
        raise ValueError(
            "y_train must be one-dimensional."
        )

    if len(features) != len(labels):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    selected_indices = stratified_sample_indices(
        labels,
        sample_count,
        seed,
    )

    selected_features = features[
        selected_indices
    ]

    selected_labels = labels[
        selected_indices
    ]

    sort_order = np.argsort(
        selected_labels,
        kind="stable",
    )

    sorted_features = selected_features[
        sort_order
    ]

    sorted_labels = selected_labels[
        sort_order
    ]

    class_labels, class_counts = np.unique(
        sorted_labels,
        return_counts=True,
    )

    displayed_labels = [
        int(label)
        for label in class_labels
    ]

    displayed_counts = [
        int(count)
        for count in class_counts
    ]

    return (
        sorted_features,
        sorted_labels,
        displayed_labels,
        displayed_counts,
    )


def validate_fidelity_kernel(
    kernel: np.ndarray,
) -> np.ndarray:
    """Validate and symmetrize a quantum fidelity Gram matrix."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "The fidelity kernel must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "The fidelity kernel must be square."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "The fidelity kernel must not be empty."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "The fidelity kernel contains non-finite values."
        )

    matrix = 0.5 * (
        matrix + matrix.T
    )

    diagonal = np.diag(
        matrix
    )

    diagonal_error = float(
        np.max(
            np.abs(
                diagonal - 1.0
            )
        )
    )

    if diagonal_error > 1e-10:
        raise ValueError(
            "The fidelity kernel does not have a unit diagonal. "
            f"Maximum error: {diagonal_error:.3e}."
        )

    minimum_value = float(
        np.min(matrix)
    )

    maximum_value = float(
        np.max(matrix)
    )

    if minimum_value < -1e-10:
        raise ValueError(
            "The fidelity kernel contains a negative value below "
            "numerical tolerance. "
            f"Minimum value: {minimum_value:.3e}."
        )

    if maximum_value > 1.0 + 1e-10:
        raise ValueError(
            "The fidelity kernel contains a value above one. "
            f"Maximum value: {maximum_value:.3e}."
        )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "The fidelity kernel is not positive semidefinite within "
            "tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}."
        )

    np.fill_diagonal(
        matrix,
        1.0,
    )

    return matrix


def normalize_kernel_for_plot(
    kernel: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, float]:
    """Normalize a square kernel for visualization.

    For trace normalization, the displayed matrix is:

        displayed_kernel = raw_kernel / trace(raw_kernel)

    The second returned value is the normalization divisor.
    """
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if normalization == "none":
        return (
            matrix.copy(),
            1.0,
        )

    if normalization == "trace":
        trace = float(
            np.trace(matrix)
        )

        if trace <= 0.0:
            raise ValueError(
                "Kernel trace must be positive."
            )

        return (
            matrix / trace,
            trace,
        )

    raise ValueError(
        f"Unknown kernel normalization: {normalization!r}."
    )


def finite_summary(
    values: np.ndarray,
) -> dict[str, float]:
    """Return statistics for finite numeric values."""
    finite_values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = finite_values[
        np.isfinite(finite_values)
    ]

    if finite_values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(
            np.mean(finite_values)
        ),
        "std": float(
            np.std(finite_values)
        ),
        "min": float(
            np.min(finite_values)
        ),
        "max": float(
            np.max(finite_values)
        ),
    }


def kernel_summary(
    kernel: np.ndarray,
) -> dict[str, float]:
    """Return descriptive statistics for a square kernel."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "kernel must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "kernel must be square."
        )

    off_diagonal_mask = ~np.eye(
        matrix.shape[0],
        dtype=bool,
    )

    off_diagonal_values = matrix[
        off_diagonal_mask
    ]

    off_diagonal_statistics = finite_summary(
        off_diagonal_values
    )

    return {
        "min": float(
            np.min(matrix)
        ),
        "max": float(
            np.max(matrix)
        ),
        "mean": float(
            np.mean(matrix)
        ),
        "std": float(
            np.std(matrix)
        ),
        "trace": float(
            np.trace(matrix)
        ),
        "diagonal_mean": float(
            np.mean(
                np.diag(matrix)
            )
        ),
        "off_diagonal_mean": (
            off_diagonal_statistics["mean"]
        ),
        "off_diagonal_std": (
            off_diagonal_statistics["std"]
        ),
        "off_diagonal_min": (
            off_diagonal_statistics["min"]
        ),
        "off_diagonal_max": (
            off_diagonal_statistics["max"]
        ),
    }


def class_similarity_summary(
    kernel: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    """Compare within-class and between-class kernel similarities."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "kernel must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "kernel must be square."
        )

    if len(matrix) != len(labels):
        raise ValueError(
            "kernel and labels must have the same sample count."
        )

    off_diagonal_mask = ~np.eye(
        len(labels),
        dtype=bool,
    )

    same_class_mask = np.equal.outer(
        labels,
        labels,
    )

    within_class_mask = np.logical_and(
        same_class_mask,
        off_diagonal_mask,
    )

    between_class_mask = np.logical_and(
        np.logical_not(
            same_class_mask
        ),
        off_diagonal_mask,
    )

    within_class_values = matrix[
        within_class_mask
    ]

    between_class_values = matrix[
        between_class_mask
    ]

    within_class_statistics = finite_summary(
        within_class_values
    )

    between_class_statistics = finite_summary(
        between_class_values
    )

    similarity_gap = (
        within_class_statistics["mean"]
        - between_class_statistics["mean"]
    )

    return {
        "within_class": (
            within_class_statistics
        ),
        "between_class": (
            between_class_statistics
        ),
        "within_minus_between_mean": float(
            similarity_gap
        ),
        "interpretation": (
            "A positive value means that average within-class similarity "
            "is larger than average between-class similarity on the "
            "displayed subset. This does not establish predictive performance."
        ),
    }


def effective_rank_from_kernel(
    kernel: np.ndarray,
) -> float:
    """Compute Shannon effective rank from a square kernel."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "kernel must be two-dimensional."
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "kernel must be square."
        )

    matrix = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "Kernel is not positive semidefinite within tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}."
        )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    eigenvalue_sum = float(
        np.sum(eigenvalues)
    )

    if eigenvalue_sum <= 0.0:
        raise ValueError(
            "Kernel has no positive eigenvalue mass."
        )

    probabilities = (
        eigenvalues / eigenvalue_sum
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


def class_boundaries(
    class_counts: list[int],
) -> list:
    """Return cumulative boundaries between displayed class blocks."""
    if len(class_counts) < 2:
        return []

    cumulative_counts = np.cumsum(
        class_counts
    )

    return [
        int(value)
        for value in cumulative_counts[:-1]
    ]


def compute_figure4(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    sample_count: int,
    kernel_normalization: str,
) -> dict[str, object]:
    """Compute the Figure 4-style quantum-kernel matrix."""
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

    y_train = y[
        training_indices
    ]

    (
        X_subset,
        y_subset,
        displayed_class_labels,
        displayed_class_counts,
    ) = select_samples_sorted_by_class(
        X_train,
        y_train,
        sample_count,
        seed,
    )

    raw_kernel = fidelity_kernel(
        X_subset
    )

    raw_kernel = validate_fidelity_kernel(
        raw_kernel
    )

    (
        plot_kernel,
        normalization_scale,
    ) = normalize_kernel_for_plot(
        raw_kernel,
        kernel_normalization,
    )

    displayed_boundaries = class_boundaries(
        displayed_class_counts
    )

    raw_kernel_statistics = kernel_summary(
        raw_kernel
    )

    plot_kernel_statistics = kernel_summary(
        plot_kernel
    )

    raw_similarity_statistics = class_similarity_summary(
        raw_kernel,
        y_subset,
    )

    raw_effective_rank = effective_rank_from_kernel(
        raw_kernel
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
        "sample_count": int(
            len(X_subset)
        ),
        "sample_selection": (
            "Seeded stratified subset of the processed training split"
        ),
        "sorted_by_class": True,
        "class_labels": (
            displayed_class_labels
        ),
        "class_counts": (
            displayed_class_counts
        ),
        "class_boundaries": (
            displayed_boundaries
        ),
        "pca_variance_percent": float(
            100.0
            * explained_variance_ratio
        ),
        "kernel_normalization": (
            kernel_normalization
        ),
        "normalization_scale": float(
            normalization_scale
        ),
        "raw_kernel_effective_rank": float(
            raw_effective_rank
        ),
        "raw_kernel_summary": (
            raw_kernel_statistics
        ),
        "plot_kernel_summary": (
            plot_kernel_statistics
        ),
        "raw_class_similarity": (
            raw_similarity_statistics
        ),
        "plot_kernel": plot_kernel,
    }


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
) -> None:
    """Write the displayed kernel matrix to CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerows(
            matrix.tolist()
        )


def resolve_color_scale(
    matrix: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float, bool]:
    """Return color limits and whether automatic scaling was used."""
    automatic_scale = (
        vmin is None
        or vmax is None
    )

    if vmin is None:
        vmin = float(
            np.quantile(
                matrix,
                lower_quantile,
            )
        )

    if vmax is None:
        vmax = float(
            np.quantile(
                matrix,
                upper_quantile,
            )
        )

    if not np.isfinite(vmin):
        raise ValueError(
            "Color-scale minimum is not finite."
        )

    if not np.isfinite(vmax):
        raise ValueError(
            "Color-scale maximum is not finite."
        )

    if vmin >= vmax:
        matrix_minimum = float(
            np.min(matrix)
        )

        matrix_maximum = float(
            np.max(matrix)
        )

        if matrix_minimum < matrix_maximum:
            vmin = matrix_minimum
            vmax = matrix_maximum
        else:
            padding = max(
                abs(matrix_minimum)
                * 1e-6,
                1e-12,
            )

            vmin = (
                matrix_minimum
                - padding
            )

            vmax = (
                matrix_maximum
                + padding
            )

    return (
        float(vmin),
        float(vmax),
        automatic_scale,
    )


def save_plot(
    path: Path,
    *,
    figure4: dict[str, object],
    vmin: float | None,
    vmax: float | None,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float, bool]:
    """Save the class-sorted quantum-kernel heatmap."""
    matrix = np.asarray(
        figure4["plot_kernel"],
        dtype=np.float64,
    )

    (
        used_vmin,
        used_vmax,
        automatic_scale,
    ) = resolve_color_scale(
        matrix,
        vmin,
        vmax,
        lower_quantile,
        upper_quantile,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            6.2,
        )
    )

    image = axis.imshow(
        matrix,
        cmap="viridis",
        vmin=used_vmin,
        vmax=used_vmax,
        interpolation="nearest",
        origin="upper",
    )

    displayed_boundaries = figure4[
        "class_boundaries"
    ]

    if not isinstance(
        displayed_boundaries,
        list,
    ):
        raise TypeError(
            "class_boundaries must be a list."
        )

    for boundary in displayed_boundaries:
        line_position = (
            float(boundary)
            - 0.5
        )

        axis.axhline(
            line_position,
            color="white",
            linewidth=0.9,
            linestyle="--",
        )

        axis.axvline(
            line_position,
            color="white",
            linewidth=0.9,
            linestyle="--",
        )

    axis.set_title(

            "Class-sorted quantum fidelity kernel\n"
            f"{figure4['model']}, "
            f"q={figure4['q']}, "
            f"n={figure4['sample_count']}"

    )

    axis.set_xlabel(
        "Sample index, sorted by class"
    )

    axis.set_ylabel(
        "Sample index, sorted by class"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    if (
        figure4["kernel_normalization"]
        == "trace"
    ):
        colorbar_label = (
            "Trace-normalized kernel value"
        )
    else:
        colorbar_label = (
            "Raw fidelity"
        )

    colorbar.set_label(
        colorbar_label
    )

    figure.tight_layout()

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

    return (
        used_vmin,
        used_vmax,
        automatic_scale,
    )


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Figure 4 artifact description."""
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
            "payload summary must be a dictionary."
        )

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

    class_similarity = summary[
        "raw_class_similarity"
    ]

    within_class = class_similarity[
        "within_class"
    ]

    between_class = class_similarity[
        "between_class"
    ]

    within_mean = float(
        within_class["mean"]
    )

    between_mean = float(
        between_class["mean"]
    )

    similarity_gap = float(
        class_similarity[
            "within_minus_between_mean"
        ]
    )

    lines = [
        "# Figure 4-style quantum-kernel heatmap",
        "",
        (
            "This artifact displays a class-sorted quantum fidelity kernel "
            "on a seeded, stratified subset of the processed training split."
        ),
        "",
        (
            "Results produced from synthetic data reproduce the diagnostic "
            "structure of Figure 4, not the paper's numerical result on the "
            "inaccessible medical embeddings."
        ),
        "",
        (
            "Sorting samples by class creates visible block boundaries. "
            "The boundaries alone do not establish that the kernel separates "
            "the classes."
        ),
        "",
        (
            f"Paper methodology pointer: "
            f"{payload['paper_pointer']}"
        ),
        "",
        (
            "![Figure 4-style quantum-kernel heatmap]"
            f"({image_name})"
        ),
        "",
        "Summary:",
        "",
        f"- model: `{summary['model']}`",
        f"- q: `{summary['q']}`",
        f"- seed: `{summary['seed']}`",
        (
            "- displayed samples: "
            f"`{summary['sample_count']}`"
        ),
        (
            "- sample selection: "
            f"`{summary['sample_selection']}`"
        ),
        (
            "- class labels: "
            f"`{summary['class_labels']}`"
        ),
        (
            "- class counts: "
            f"`{summary['class_counts']}`"
        ),
        (
            "- kernel normalization: "
            f"`{summary['kernel_normalization']}`"
        ),
        (
            "- PCA explained variance: "
            f"`{summary['pca_variance_percent']:.3f}%`"
        ),
        (
            "- raw-kernel effective rank: "
            f"`{summary['raw_kernel_effective_rank']:.3f}`"
        ),
        (
            "- raw off-diagonal mean: "
            f"`{summary['raw_kernel_summary']['off_diagonal_mean']:.6f}`"
        ),
        (
            "- raw within-class similarity mean: "
            f"`{within_mean:.6f}`"
        ),
        (
            "- raw between-class similarity mean: "
            f"`{between_mean:.6f}`"
        ),
        (
            "- within-minus-between similarity mean: "
            f"`{similarity_gap:.6f}`"
        ),
        "",
        "Interpretation:",
        "",
        (
            "- A positive similarity gap indicates larger average "
            "within-class similarity on this displayed subset."
        ),
        (
            "- The similarity gap is descriptive and does not establish "
            "classification performance or a quantum advantage."
        ),
        (
            "- Trace normalization changes the matrix scale but does not "
            "change its effective rank or relative structure."
        ),
        (
            "- Synthetic results reflect both the calibrated generator "
            "geometry and the selected quantum feature map."
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
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix associated with the data source."""
    if source == "synthetic":
        return "synthetic_surrogate_figure4"

    if source == "synthetic_file":
        return "synthetic_file_figure4"

    return "real_figure4"


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

    parser.add_argument(
        "--sample-count",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "none",
            "trace",
        ),
        default="trace",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--color-lower-quantile",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--color-upper-quantile",
        type=float,
        default=0.99,
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
    if args.q <= 0:
        raise ValueError(
            "--q must be positive."
        )

    if args.sample_count <= 0:
        raise ValueError(
            "--sample-count must be positive."
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

    if not 0.0 <= args.color_lower_quantile < 1.0:
        raise ValueError(
            "--color-lower-quantile must be in [0, 1)."
        )

    if not 0.0 < args.color_upper_quantile <= 1.0:
        raise ValueError(
            "--color-upper-quantile must be in (0, 1]."
        )

    if (
        args.color_lower_quantile
        >= args.color_upper_quantile
    ):
        raise ValueError(
            "The lower color quantile must be smaller than "
            "the upper color quantile."
        )

    if (
        args.vmin is not None
        and args.vmax is not None
        and args.vmin >= args.vmax
    ):
        raise ValueError(
            "--vmin must be smaller than --vmax."
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
    """Generate the Figure 4-style artifact."""
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

    figure4 = compute_figure4(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
        sample_count=args.sample_count,
        kernel_normalization=(
            args.kernel_normalization
        ),
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
        / f"{prefix}_kernel_matrix.csv"
    )

    json_path = (
        args.results_dir
        / f"{prefix}.json"
    )

    markdown_path = (
        args.results_dir
        / f"{prefix}.md"
    )

    matrix = np.asarray(
        figure4["plot_kernel"],
        dtype=np.float64,
    )

    write_matrix_csv(
        csv_path,
        matrix,
    )

    (
        used_vmin,
        used_vmax,
        automatic_scale,
    ) = save_plot(
        png_path,
        figure4=figure4,
        vmin=args.vmin,
        vmax=args.vmax,
        lower_quantile=(
            args.color_lower_quantile
        ),
        upper_quantile=(
            args.color_upper_quantile
        ),
    )

    summary = {
        key: value
        for key, value in figure4.items()
        if key != "plot_kernel"
    }

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 4",
        "paper_pointer": (
            PAPER_FIGURE4_POINTER
        ),
        "scope": {
            "kernel": (
                "Manuscript quantum fidelity kernel"
            ),
            "quantum_feature_map": (
                "Ry encoding followed by a CNOT ring"
            ),
            "classifier_trained": False,
            "synthetic_result_is_paper_result": False,
            "class_sorting_proves_separation": False,
        },
        "paths": {
            "png": str(
                png_path
            ),
            "kernel_matrix_csv": str(
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
                "StandardScaler, PCA and MinMaxScaler fitted on the "
                "complete training split before subset selection"
            ),
            "sample_selection": (
                "Seeded stratified subset of the processed training split"
            ),
            "color_scale": [
                used_vmin,
                used_vmax,
            ],
            "color_scale_automatic": (
                automatic_scale
            ),
            "color_scale_quantiles": [
                args.color_lower_quantile,
                args.color_upper_quantile,
            ],
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
