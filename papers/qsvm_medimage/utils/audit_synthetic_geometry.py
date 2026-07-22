#!/usr/bin/env python3
"""Audit embedding, PCA, and kernel geometry without training classifiers.

The script is observational. It does not:

- modify the dataset;
- tune the synthetic generator;
- train classifiers;
- estimate predictive performance;
- validate the inaccessible medical-data results.

For each model, seed, and PCA dimension q, the script measures:

- dataset size and class proportions;
- PCA explained variance;
- processed-feature geometry;
- full linear-kernel spectrum;
- sampled RBF-kernel spectrum;
- sampled quantum fidelity-kernel spectrum.

Selected values from Table V of the paper are included as documentary
references. They are not optimisation targets.

Important comparability rule
----------------------------
The linear kernel is analysed on the complete training set.

The RBF and quantum kernels may be analysed on a smaller stratified subset to
control runtime. Effective ranks computed on that subset are not directly
comparable with values computed on the paper's full training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
)

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

DEFAULT_Q_VALUES = (
    4,
    6,
    8,
    11,
    16,
)


# Values transcribed from Table V of the paper.
#
# These are documentary references, not optimisation targets.
#
# Rows marked as linear_comparison=True describe a linear PCA-q kernel.
# Dagger rows at q=11 and q=16 describe quantum-kernel quantities.
PAPER_TABLE5_REFERENCE: dict[
    tuple[str, int],
    dict[str, float | int | bool | None | str],
] = {
    ("medsiglip-448", 4): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 32.6,
        "positive_rank": 4,
        "effective_rank": 3.77,
        "lambda_max": 770.6,
        "note": "",
    },
    ("medsiglip-448", 6): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 41.1,
        "positive_rank": 6,
        "effective_rank": 5.53,
        "lambda_max": 614.8,
        "note": "",
    },
    ("medsiglip-448", 11): {
        "reference_kernel": "quantum",
        "linear_comparison": False,
        "pca_var_percent": 56.0,
        "positive_rank": None,
        "effective_rank": 43.04,
        "lambda_max": 468.6,
        "note": (
            "The dagger row reports quantum-kernel quantities, not linear "
            "PCA-q kernel quantities."
        ),
    },
    ("medsiglip-448", 16): {
        "reference_kernel": "quantum",
        "linear_comparison": False,
        "pca_var_percent": None,
        "positive_rank": None,
        "effective_rank": 92.13,
        "lambda_max": None,
        "note": (
            "The dagger row reports quantum-kernel quantities, not linear "
            "PCA-q kernel quantities."
        ),
    },
    ("rad-dino", 4): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 5.6,
        "positive_rank": 4,
        "effective_rank": 3.89,
        "lambda_max": 666.1,
        "note": "",
    },
    ("rad-dino", 6): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 7.7,
        "positive_rank": 6,
        "effective_rank": 5.85,
        "lambda_max": 475.5,
        "note": "",
    },
    ("vit-patch32-cls", 4): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 28.4,
        "positive_rank": 4,
        "effective_rank": 3.86,
        "lambda_max": 627.5,
        "note": "",
    },

("vit-patch32-cls", 6): {
        "reference_kernel": "linear",
        "linear_comparison": True,
        "pca_var_percent": 34.7,
        "positive_rank": 6,
        "effective_rank": 5.59,
        "lambda_max": 502.3,
        "note": "",
    },
}


def parse_ints(raw: str) -> list:
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


def parse_strings(raw: str) -> list:
    """Parse a non-empty comma-separated list of unique strings."""
    values = [
        part.strip()
        for part in raw.split(",")
        if part.strip()
    ]

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one value is required."
        )

    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "Values must be unique."
        )

    return values


def finite_stats(
    values: np.ndarray,
) -> dict[str, float]:
    """Return descriptive statistics after removing non-finite values."""
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
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
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

    # Small negative values can result from numerical diagonalization.
    values = np.maximum(
        values,
        0.0,
    )

    total = float(
        np.sum(values)
    )

    if total <= 0.0:
        return 0.0

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


def positive_rank_from_eigenvalues(
    eigenvalues: np.ndarray,
) -> int:
    """Count eigenvalues above a scale-dependent numerical threshold."""
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

    largest_absolute_value = float(
        np.max(np.abs(values))
    )

    tolerance = max(
        1e-12,
        1e-10 * largest_absolute_value,
    )

    return int(
        np.sum(values > tolerance)
    )


def components_needed(
    cumulative_variance: np.ndarray,
    threshold: float,
) -> int | None:
    """Return the first PCA component count reaching a variance threshold."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "threshold must be in (0, 1]."
        )

    hits = np.flatnonzero(
        cumulative_variance >= threshold
    )

    if hits.size == 0:
        return None

    return int(
        hits[0] + 1
    )


def off_diagonal_values(
    matrix: np.ndarray,
) -> np.ndarray:
    """Return all off-diagonal values of a square matrix."""
    array = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
    ):
        raise ValueError(
            "matrix must be square."
        )

    if array.shape[0] < 2:
        return np.array(
            [],
            dtype=np.float64,
        )

    mask = ~np.eye(
        array.shape[0],
        dtype=bool,
    )

    return array[mask]


def summarize_kernel(
    kernel: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    """Summarize one square kernel matrix."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            "Kernel summary requires a square matrix."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Kernel matrix must not be empty."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Kernel contains non-finite values."
        )

    # Remove small numerical asymmetries.
    matrix = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    clipped_eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    diagonal = np.diag(
        matrix
    )

    off_diagonal = off_diagonal_values(
        matrix
    )

    off_diagonal_stats = finite_stats(
        off_diagonal
    )

    return {
        f"{prefix}_sample_count": int(
            matrix.shape[0]
        ),
        f"{prefix}_trace": float(
            np.trace(matrix)
        ),
        f"{prefix}_diag_mean": float(
            np.mean(diagonal)
        ),
        f"{prefix}_diag_std": float(
            np.std(diagonal)
        ),
        f"{prefix}_offdiag_mean": (
            off_diagonal_stats["mean"]
        ),
        f"{prefix}_offdiag_std": (
            off_diagonal_stats["std"]
        ),
        f"{prefix}_offdiag_min": (
            off_diagonal_stats["min"]
        ),
        f"{prefix}_offdiag_max": (
            off_diagonal_stats["max"]
        ),
        f"{prefix}_minimum_eigenvalue": (
            minimum_eigenvalue
        ),
        f"{prefix}_positive_rank": (
            positive_rank_from_eigenvalues(
                clipped_eigenvalues
            )
        ),
        f"{prefix}_effective_rank": (
            effective_rank_from_eigenvalues(
                clipped_eigenvalues
            )
        ),
        f"{prefix}_lambda_max": float(
            np.max(clipped_eigenvalues)
        ),
    }


def summarize_linear_full_rank(
    X_train: np.ndarray,
) -> dict[str, float | int | bool]:
    """Summarize the complete training linear-kernel spectrum.

    The non-zero eigenvalues of X @ X.T equal those of X.T @ X.

    After PCA-q, X.T @ X has shape q by q, so this calculation avoids
    diagonalizing the complete training Gram matrix.
    """
    features = np.asarray(
        X_train,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "X_train must be a two-dimensional feature matrix."
        )

    small_gram_matrix = (
        features.T @ features
    )

    eigenvalues = np.linalg.eigvalsh(
        small_gram_matrix
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    positive_rank = (
        positive_rank_from_eigenvalues(
            eigenvalues
        )
    )

    effective_rank = (
        effective_rank_from_eigenvalues(
            eigenvalues
        )
    )

    lambda_max_raw = float(
        np.max(eigenvalues)
    )

    trace_raw = float(
        np.sum(eigenvalues)
    )

    if trace_raw > 0.0:
        lambda_max_trace_1 = (
            lambda_max_raw / trace_raw
        )
    else:
        lambda_max_trace_1 = 0.0

    lambda_max_trace_n = (
        lambda_max_trace_1
        * features.shape[0]
    )

    return {
        "linear_full_positive_rank": positive_rank,
        "linear_full_effective_rank": effective_rank,
        "linear_full_lambda_max_raw": lambda_max_raw,
        "linear_full_lambda_max_trace_1": (
            lambda_max_trace_1
        ),
        "linear_full_lambda_max_trace_n": (
            lambda_max_trace_n
        ),
        "linear_full_trace_raw": trace_raw,
        "linear_rank_upper_bound": int(
            features.shape[1]
        ),
        "linear_rank_validation": bool(
            positive_rank <= features.shape[1]
        ),
    }


def summarize_features(
    X: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Summarize one processed feature matrix."""
    features = np.asarray(
        X,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "Feature summary requires a two-dimensional matrix."
        )

    row_norms = np.linalg.norm(
        features,
        axis=1,
    )

    row_means = np.mean(
        features,
        axis=1,
    )

    row_stds = np.std(
        features,
        axis=1,
    )

    saturation_fraction = float(
        np.mean(
            np.abs(features) >= 0.999
        )
    )

    return {
        f"{prefix}_feature_mean": float(
            np.mean(features)
        ),
        f"{prefix}_feature_std": float(
            np.std(features)
        ),
        f"{prefix}_feature_min": float(
            np.min(features)
        ),
        f"{prefix}_feature_max": float(
            np.max(features)
        ),
        f"{prefix}_feature_saturation_fraction": (
            saturation_fraction
        ),
        f"{prefix}_row_norm_mean": float(
            np.mean(row_norms)
        ),
        f"{prefix}_row_norm_std": float(
            np.std(row_norms)
        ),
        f"{prefix}_row_mean_mean": float(
            np.mean(row_means)
        ),
        f"{prefix}_row_std_mean": float(
            np.mean(row_stds)
        ),
    }


def summarize_pairwise_geometry(
    X: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Summarize Euclidean distances and cosine similarities."""
    features = np.asarray(
        X,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "Pairwise geometry requires a two-dimensional matrix."
        )

    if len(features) < 2:
        return {
            f"{prefix}_distance_mean": float("nan"),
            f"{prefix}_distance_std": float("nan"),
            f"{prefix}_distance_min": float("nan"),
            f"{prefix}_distance_max": float("nan"),
            f"{prefix}_cosine_mean": float("nan"),
            f"{prefix}_cosine_std": float("nan"),
            f"{prefix}_cosine_min": float("nan"),
            f"{prefix}_cosine_max": float("nan"),
        }

    gram = features @ features.T
    squared_norms = np.diag(
        gram
    )

    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * gram
    )

    distances = np.sqrt(
        np.maximum(
            squared_distances,
            0.0,
        )
    )

    norms = np.sqrt(
        np.maximum(
            squared_norms,
            0.0,
        )
    )

    denominator = np.outer(
        norms,
        norms,
    )

    cosine = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )

    distance_stats = finite_stats(
        off_diagonal_values(distances)
    )

    cosine_stats = finite_stats(
        off_diagonal_values(cosine)
    )

    return {
        f"{prefix}_distance_mean": (
            distance_stats["mean"]
        ),
        f"{prefix}_distance_std": (
            distance_stats["std"]
        ),
        f"{prefix}_distance_min": (
            distance_stats["min"]
        ),
        f"{prefix}_distance_max": (
            distance_stats["max"]
        ),
        f"{prefix}_cosine_mean": (
            cosine_stats["mean"]
        ),
        f"{prefix}_cosine_std": (
            cosine_stats["std"]
        ),
        f"{prefix}_cosine_min": (
            cosine_stats["min"]
        ),
        f"{prefix}_cosine_max": (
            cosine_stats["max"]
        ),
    }


def maybe_delta(
    value: float | int | None,
    reference: float | int | None,
) -> float | None:
    """Return value minus reference when both are available."""
    if value is None or reference is None:
        return None

    return float(value) - float(reference)


def maybe_ratio(
    value: float | int | None,
    reference: float | int | None,
) -> float | None:
    """Return value divided by reference when defined."""
    if (
        value is None
        or reference is None
        or float(reference) == 0.0
    ):
        return None

    return float(value) / float(reference)


def add_paper_table5_comparison(
    row: dict[str, object],
) -> None:
    """Attach documentary Table V values to one audit row."""
    model = str(
        row["model"]
    )

    q = int(
        row["q"]
    )

    reference = PAPER_TABLE5_REFERENCE.get(
        (model, q)
    )

    row["paper_table5_has_reference"] = bool(
        reference
    )

    if reference is None:
        return

    reference_kernel = str(
        reference["reference_kernel"]
    )

    linear_comparison = bool(
        reference["linear_comparison"]
    )

    row["paper_table5_reference_kernel"] = (
        reference_kernel
    )

    row["paper_table5_linear_comparison"] = (
        linear_comparison
    )

    row["paper_table5_pca_var_percent"] = (
        reference["pca_var_percent"]
    )

    row["paper_table5_positive_rank"] = (
        reference["positive_rank"]
    )

    row["paper_table5_effective_rank"] = (
        reference["effective_rank"]
    )

    row["paper_table5_lambda_max"] = (
        reference["lambda_max"]
    )

    row["paper_table5_note"] = str(
        reference.get("note", "")
    )

    row["delta_pca_var_percent_vs_table5"] = (
        maybe_delta(
            row.get(
                "pca_explained_variance_percent"
            ),
            reference["pca_var_percent"],
        )
    )

    row["ratio_pca_var_percent_vs_table5"] = (
        maybe_ratio(
            row.get(
                "pca_explained_variance_percent"
            ),
            reference["pca_var_percent"],
        )
    )

    if not linear_comparison:
        # Dagger rows report quantum-kernel values. The sampled quantum kernel
        # in this audit is not compared with a full-training-set paper value.
        row["delta_positive_rank_vs_table5"] = None
        row["delta_effective_rank_vs_table5"] = None
        row["ratio_effective_rank_vs_table5"] = None
        row["delta_lambda_max_raw_vs_table5"] = None
        row["ratio_lambda_max_raw_vs_table5"] = None
        return

    row["delta_positive_rank_vs_table5"] = (
        maybe_delta(
            row.get(
                "linear_full_positive_rank"
            ),
            reference["positive_rank"],
        )
    )

    row["delta_effective_rank_vs_table5"] = (
        maybe_delta(
            row.get(
                "linear_full_effective_rank"
            ),
            reference["effective_rank"],
        )
    )

    row["ratio_effective_rank_vs_table5"] = (
        maybe_ratio(
            row.get(
                "linear_full_effective_rank"
            ),
            reference["effective_rank"],
        )
    )

    row["delta_lambda_max_raw_vs_table5"] = (
        maybe_delta(
            row.get(
                "linear_full_lambda_max_raw"
            ),
            reference["lambda_max"],
        )
    )

    row["ratio_lambda_max_raw_vs_table5"] = (
        maybe_ratio(
            row.get(
                "linear_full_lambda_max_raw"
            ),
            reference["lambda_max"],
        )
    )


def pca_spectrum_rows(
    *,
    X_train_raw: np.ndarray,
    model: str,
    seed: int,
    max_components: int,
) -> list[dict[str, object]]:
    """Compute the standardized training-set PCA spectrum."""
    scaler = StandardScaler().fit(
        X_train_raw
    )

    X_scaled = scaler.transform(
        X_train_raw
    )

    component_count = min(
        max_components,
        X_scaled.shape[0],
        X_scaled.shape[1],
    )

    pca = PCA(
        n_components=component_count
    ).fit(
        X_scaled
    )

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    components_50 = components_needed(
        cumulative_variance,
        0.50,
    )

    components_80 = components_needed(
        cumulative_variance,
        0.80,
    )

    components_90 = components_needed(
        cumulative_variance,
        0.90,
    )

    rows: list[dict[str, object]] = []

    for index, explained_variance in enumerate(
        pca.explained_variance_ratio_
    ):
        rows.append(
            {
                "model": model,
                "seed": seed,
                "component": index + 1,
                "explained_variance_ratio": float(
                    explained_variance
                ),
                "cumulative_explained_variance": float(
                    cumulative_variance[index]
                ),
                "components_needed_50_percent": (
                    components_50
                ),
                "components_needed_80_percent": (
                    components_80
                ),
                "components_needed_90_percent": (
                    components_90
                ),
            }
        )

    return rows


def stratified_sample_indices(
    y: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic stratified subset without replacement."""
    labels = np.asarray(
        y,
        dtype=int,
    )

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    if sample_count >= len(labels):
        return np.arange(
            len(labels)
        )

    indices = np.arange(
        len(labels)
    )

    sampled_indices, _ = train_test_split(
        indices,
        train_size=sample_count,
        random_state=seed,
        stratify=labels,
    )

    return np.sort(
        sampled_indices
    )


def audit_one_q(
    *,
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    seed: int,
    q: int,
    source: str,
    kernel_samples: int,
    max_quantum_q: int,
    skip_quantum: bool,
) -> dict[str, object]:
    """Audit one model, seed, and PCA dimension."""
    (
        train_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    (
        X_train,
        X_validation,
        X_test,
        pca_explained_variance,
    ) = preprocess(
        X[train_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    y_train = y[
        train_indices
    ]

    sample_count = min(
        kernel_samples,
        len(X_train),
    )

    sampled_indices = stratified_sample_indices(
        y_train,
        sample_count,
        seed,
    )

    X_kernel = X_train[
        sampled_indices
    ]

    y_kernel = y_train[
        sampled_indices
    ]

    row: dict[str, object] = {
        "source": source,
        "synthetic_surrogate": (
            source != "real"
        ),
        "model": model,
        "seed": seed,
        "q": q,
        "n_features_raw": int(
            X.shape[1]
        ),
        "n_train": int(
            len(X_train)
        ),
        "n_validation": int(
            len(X_validation)
        ),
        "n_test": int(
            len(X_test)
        ),
        "pca_explained_variance": float(
            pca_explained_variance
        ),
        "pca_explained_variance_percent": float(
            100.0 * pca_explained_variance
        ),
        "kernel_sample_count": int(
            sample_count
        ),
        "kernel_sample_positive_ratio": float(
            np.mean(y_kernel)
        ),
        "kernel_sample_is_full_train": bool(
            sample_count == len(X_train)
        ),
    }

    row.update(
        summarize_features(
            X_train,
            "train",
        )
    )

    row.update(
        summarize_pairwise_geometry(
            X_kernel,
            "sample",
        )
    )

    row.update(
        summarize_linear_full_rank(
            X_train
        )
    )

    linear_sample_kernel = (
        X_kernel @ X_kernel.T
    )

    row.update(
        summarize_kernel(
            linear_sample_kernel,
            "linear_sample_kernel",
        )
    )

    feature_variance = float(
        X_train.var()
    )

    if feature_variance <= 0.0:
        raise ValueError(
            "Processed training features have zero variance."
        )

    gamma_scale = 1.0 / (
        X_train.shape[1]
        * feature_variance
    )

    row["rbf_gamma_scale"] = float(
        gamma_scale
    )

    rbf_sample_kernel = rbf_kernel(
        X_kernel,
        gamma=gamma_scale,
    )

    row.update(
        summarize_kernel(
            rbf_sample_kernel,
            "rbf_sample_kernel",
        )
    )

    compute_quantum = (
        not skip_quantum
        and q <= max_quantum_q
    )

    row["quantum_kernel_computed"] = bool(
        compute_quantum
    )

    row[
        "quantum_rank_comparable_to_full_paper_kernel"
    ] = bool(
        compute_quantum
        and sample_count == len(X_train)
    )

    if compute_quantum:
        quantum_sample_kernel = fidelity_kernel(
            X_kernel
        )

        row.update(
            summarize_kernel(
                quantum_sample_kernel,
                "quantum_sample_kernel",
            )
        )

    add_paper_table5_comparison(
        row
    )

    return row


def audit_dataset(
    *,
    source: str,
    data_root: Path | None,
    models: list[str],
    seeds: list[int],
    q_values: list[int],
    synthetic: SyntheticSpec,
    max_pca_components: int,
    kernel_samples: int,
    max_quantum_q: int,
    skip_quantum: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Audit all requested models, seeds, and PCA dimensions."""
    summary_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []

    for model in models:
        if model not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model: {model!r}."
            )

        for seed in seeds:
            X, y = load_dataset(
                source=source,
                model=model,
                seed=seed,
                data_root=data_root,
                synthetic=synthetic,
            )

            (
                train_indices,
                validation_indices,
                test_indices,
            ) = split_indices(
                y,
                seed=seed,
            )

            dataset_rows.append(
                {
                    "source": source,
                    "synthetic_surrogate": (
                        source != "real"
                    ),
                    "model": model,
                    "seed": seed,
                    "n_samples": int(
                        len(y)
                    ),
                    "n_features_raw": int(
                        X.shape[1]
                    ),
                    "n_train": int(
                        len(train_indices)
                    ),
                    "n_validation": int(
                        len(validation_indices)
                    ),
                    "n_test": int(
                        len(test_indices)
                    ),
                    "positive_count": int(
                        np.sum(y == 1)
                    ),
                    "negative_count": int(
                        np.sum(y == 0)
                    ),
                    "positive_ratio_all": float(
                        np.mean(y)
                    ),
                    "positive_ratio_train": float(
                        np.mean(
                            y[train_indices]
                        )
                    ),
                    "positive_ratio_validation": float(
                        np.mean(
                            y[validation_indices]
                        )
                    ),
                    "positive_ratio_test": float(
                        np.mean(
                            y[test_indices]
                        )
                    ),
                }
            )

            pca_rows.extend(
                pca_spectrum_rows(
                    X_train_raw=X[
                        train_indices
                    ],
                    model=model,
                    seed=seed,
                    max_components=(
                        max_pca_components
                    ),
                )
            )

            for q in q_values:
                maximum_q = min(
                    X.shape[1],
                    len(train_indices),
                )

                if q > maximum_q:
                    print(
                        f"Skipped model={model} seed={seed} q={q}: "
                        f"maximum supported q is {maximum_q}."
                    )
                    continue

                summary_rows.append(
                    audit_one_q(
                        X=X,
                        y=y,
                        model=model,
                        seed=seed,
                        q=q,
                        source=source,
                        kernel_samples=(
                            kernel_samples
                        ),
                        max_quantum_q=(
                            max_quantum_q
                        ),
                        skip_quantum=(
                            skip_quantum
                        ),
                    )
                )

                print(
                    f"Audited model={model} "
                    f"seed={seed} q={q}"
                )

    return (
        summary_rows,
        pca_rows,
        dataset_rows,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries with potentially different keys to CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields = sorted(
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
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write a concise human-readable audit description."""
    data = payload["data"]
    paths = payload["paths"]

    if not isinstance(data, dict):
        raise TypeError(
            "payload['data'] must be a dictionary."
        )

    if not isinstance(paths, dict):
        raise TypeError(
            "payload['paths'] must be a dictionary."
        )

    lines = [
        "# Synthetic Geometry Audit",
        "",
        (
            "This audit is observational. It does not tune the generator, "
            "train classifiers, or validate the gated medical task."
        ),
        "",
        (
            "The synthetic benchmark matches selected reported statistics. "
            "It does not reconstruct the full distribution of the original "
            "medical embeddings."
        ),
        "",
        (
            "The nonlinear synthetic label rule was defined by this "
            "reproduction and does not come from the paper."
        ),
        "",
        "Inputs:",
        "",
        f"- source: {data['source']}",
        f"- data root: {data['data_root']}",
        f"- models: {', '.join(data['models'])}",
        f"- seeds: {data['seeds']}",
        f"- q values: {data['q_values']}",
        f"- kernel samples: {data['kernel_samples']}",
        f"- maximum quantum q: {data['max_quantum_q']}",
        f"- quantum audit skipped: {data['skip_quantum']}",
        "",
        "Outputs:",
        "",
        (
            "- dataset summary: "
            f"{Path(paths['dataset_csv']).name}"
        ),
        (
            "- PCA spectrum: "
            f"{Path(paths['pca_csv']).name}"
        ),
        (
            "- q-level geometry: "
            f"{Path(paths['summary_csv']).name}"
        ),
        (
            "- JSON payload: "
            f"{Path(paths['json']).name}"
        ),
        "",
        "Interpretation rules:",
        "",
        (
            "- Table V values are documentary references, not "
            "optimisation targets."
        ),
        (
            "- The full linear-kernel positive rank must not exceed q "
            "after PCA-q."
        ),
        (
            "- Quantum ranks computed on a subset are not directly "
            "comparable with full-training-set paper values."
        ),
        (
            "- A larger numerical rank does not necessarily imply a "
            "larger effective rank."
        ),
        (
            "- A larger effective rank does not by itself imply better "
            "classification."
        ),
    ]

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix associated with the data source."""
    if source == "synthetic_file":
        return "synthetic_file_geometry_audit"

    if source == "synthetic":
        return "synthetic_surrogate_geometry_audit"

    return "real_geometry_audit"


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
        default=Path(
            "data/synthetic_qml_mimic_cxr_embeddings"
        ),
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
        "--models",
        type=parse_strings,
        default=list(MODEL_NAMES),
    )

    parser.add_argument(
        "--seeds",
        type=parse_ints,
        default=[0],
    )

    parser.add_argument(
        "--q-values",
        type=parse_ints,
        default=list(DEFAULT_Q_VALUES),
    )

    parser.add_argument(
        "--max-pca-components",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--kernel-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max-quantum-q",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--skip-quantum",
        action="store_true",
    )

    # Parameters used only by the in-memory synthetic source.
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
    if args.max_pca_components <= 0:
        raise ValueError(
            "--max-pca-components must be positive."
        )

    if args.kernel_samples <= 0:
        raise ValueError(
            "--kernel-samples must be positive."
        )

    if args.max_quantum_q <= 0:
        raise ValueError(
            "--max-quantum-q must be positive."
        )

    if any(q <= 0 for q in args.q_values):
        raise ValueError(
            "All q values must be positive."
        )

    unknown_models = [
        model
        for model in args.models
        if model not in MODEL_NAMES
    ]

    if unknown_models:
        raise ValueError(
            "Unknown models: "
            + ", ".join(unknown_models)
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
    """Run the complete geometry audit."""
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

    (
        summary_rows,
        pca_rows,
        dataset_rows,
    ) = audit_dataset(
        source=args.source,
        data_root=args.data_root,
        models=args.models,
        seeds=args.seeds,
        q_values=args.q_values,
        synthetic=synthetic,
        max_pca_components=(
            args.max_pca_components
        ),
        kernel_samples=(
            args.kernel_samples
        ),
        max_quantum_q=(
            args.max_quantum_q
        ),
        skip_quantum=(
            args.skip_quantum
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

    dataset_path = (
        args.results_dir
        / f"{prefix}_dataset.csv"
    )

    pca_path = (
        args.results_dir
        / f"{prefix}_pca.csv"
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
        dataset_path,
        dataset_rows,
    )

    write_csv(
        pca_path,
        pca_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "scope": {
            "observational": True,
            "trains_classifiers": False,
            "tunes_generator": False,
            "reconstructs_medical_distribution": False,
            "synthetic_label_rule_from_paper": False,
            "paper_values_are_optimisation_targets": False,
        },
        "data": {
            "source": args.source,
            "data_root": (
                str(args.data_root)
                if args.data_root
                else None
            ),
            "models": args.models,
            "seeds": args.seeds,
            "q_values": args.q_values,
            "max_pca_components": (
                args.max_pca_components
            ),
            "kernel_samples": (
                args.kernel_samples
            ),
            "max_quantum_q": (
                args.max_quantum_q
            ),
            "skip_quantum": (
                args.skip_quantum
            ),
            "synthetic_spec": (
                asdict(synthetic)
                if args.source == "synthetic"
                else None
            ),
        },
        "paths": {
            "dataset_csv": str(
                dataset_path
            ),
            "pca_csv": str(
                pca_path
            ),
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
        "paper_table5_reference": {
            f"{model}:{q}": reference
            for (
                model,
                q,
            ), reference in (
                PAPER_TABLE5_REFERENCE.items()
            )
        },
        "dataset_rows": dataset_rows,
        "summary_rows": summary_rows,
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
        payload,
    )

    print(
        json.dumps(
            {
                "datasets": len(dataset_rows),
                "pca_rows": len(pca_rows),
                "summary_rows": len(summary_rows),
                "dataset_csv": str(
                    dataset_path
                ),
                "pca_csv": str(
                    pca_path
                ),
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
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
