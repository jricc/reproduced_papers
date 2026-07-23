#!/usr/bin/env python3
"""Compute a Table 10-style rank-matched RBF versus QSVM comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 10 protocol shape:

- MedSigLIP embeddings by default;
- q values 4, 6, 11, and 16;
- ten seeds;
- fixed C=1;
- default RBF gamma;
- rank-matched RBF gamma;
- quantum fidelity kernel;
- trace normalization by default.

The target effective rank for each q is computed using ``rank_seed`` and is
then held fixed across all evaluation seeds.

A majority-only collapse means that the classifier predicts no class-1
samples. A low minority-class F1 below ``collapse_threshold`` is recorded
separately and is not treated as the definition of collapse.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

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


from lib.quantum_kernel import (  # noqa: E402
    effective_rank,
    fidelity_kernel,
)
from lib.svm_pipeline import (  # noqa: E402
    preprocess,
    split_indices,
)
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
    parse_ints,
)

PAPER_TABLE10_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T10"

COLLAPSE_THRESHOLD = 0.05
DEFAULT_Q_VALUES = "4,6,11,16"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP-448",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-patch32-cls",
}

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)


def validate_kernel_pair(
    K_train: np.ndarray,
    K_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and symmetrize a training and test kernel pair."""
    training_kernel = np.asarray(
        K_train,
        dtype=np.float64,
    )

    test_kernel = np.asarray(
        K_test,
        dtype=np.float64,
    )

    if training_kernel.ndim != 2:
        raise ValueError("K_train must be two-dimensional.")

    if training_kernel.shape[0] != training_kernel.shape[1]:
        raise ValueError("K_train must be square.")

    if training_kernel.shape[0] == 0:
        raise ValueError("K_train must not be empty.")

    if test_kernel.ndim != 2:
        raise ValueError("K_test must be two-dimensional.")

    if test_kernel.shape[1] != training_kernel.shape[0]:
        raise ValueError("K_test must have one column per training sample.")

    if not np.all(np.isfinite(training_kernel)):
        raise ValueError("K_train contains non-finite values.")

    if not np.all(np.isfinite(test_kernel)):
        raise ValueError("K_test contains non-finite values.")

    training_kernel = 0.5 * (training_kernel + training_kernel.T)

    return (
        training_kernel,
        test_kernel,
    )


def normalize_train_test_kernels(
    K_train: np.ndarray,
    K_test: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize training and test kernels with one training-derived scale."""
    (
        training_kernel,
        test_kernel,
    ) = validate_kernel_pair(
        K_train,
        K_test,
    )

    if normalization == "none":
        return (
            training_kernel.copy(),
            test_kernel.copy(),
        )

    if normalization == "trace":
        scale = float(np.trace(training_kernel))

        if not np.isfinite(scale):
            raise ValueError("Training-kernel trace must be finite.")

        if scale <= 0.0:
            raise ValueError("Training-kernel trace must be positive.")

        return (
            training_kernel / scale,
            test_kernel / scale,
        )

    raise ValueError(f"Unknown normalization: {normalization!r}.")


def rbf_effective_rank(
    X_train: np.ndarray,
    gamma: float,
) -> float:
    """Compute the effective rank of an RBF training kernel."""
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite.")

    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    kernel = rbf_kernel(
        X_train,
        gamma=gamma,
    )

    return float(effective_rank(kernel))


def find_rank_matched_gamma(
    X_train: np.ndarray,
    target_rank: float,
    *,
    tol: float = 0.05,
    max_iter: int = 50,
) -> float:
    """Find an RBF gamma whose effective rank matches the target rank."""
    if not np.isfinite(target_rank):
        raise ValueError("target_rank must be finite.")

    if target_rank <= 0.0:
        raise ValueError("target_rank must be positive.")

    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    gamma_low = 1e-6
    gamma_high = 1e3

    rank_low = rbf_effective_rank(
        X_train,
        gamma_low,
    )

    rank_high = rbf_effective_rank(
        X_train,
        gamma_high,
    )

    if target_rank <= rank_low:
        return gamma_low

    if target_rank >= rank_high:
        return gamma_high

    for _ in range(max_iter):
        gamma_mid = float(np.sqrt(gamma_low * gamma_high))

        rank_mid = rbf_effective_rank(
            X_train,
            gamma_mid,
        )

        relative_error = abs(rank_mid - target_rank) / target_rank

        if relative_error < tol:
            return gamma_mid

        if rank_mid < target_rank:
            gamma_low = gamma_mid
        else:
            gamma_high = gamma_mid

    return float(np.sqrt(gamma_low * gamma_high))


def compute_rank_target(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> float:
    """Compute the fixed quantum effective-rank target for one q value."""
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
        _,
    ) = preprocess(
        X[training_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    quantum_kernel = fidelity_kernel(X_train)

    return float(effective_rank(quantum_kernel))


def prediction_metrics(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    """Compute accuracy, F1, prediction counts, and collapse diagnostics."""
    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    f1 = float(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    predicted_class_0 = int(np.sum(predictions == 0))

    predicted_class_1 = int(np.sum(predictions == 1))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "predicted_class_0": (predicted_class_0),
        "predicted_class_1": (predicted_class_1),
        "collapse": bool(predicted_class_1 == 0),
        "zero_f1": bool(
            np.isclose(
                f1,
                0.0,
            )
        ),
    }


def compute_qsvm_metrics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    seed: int,
    kernel_normalization: str,
) -> dict[str, object]:
    """Compute the fixed-C QSVM metrics for one seed."""
    training_kernel_raw = fidelity_kernel(X_train)

    test_kernel_raw = fidelity_kernel(
        X_test,
        X_train,
    )

    (
        training_kernel,
        test_kernel,
    ) = normalize_train_test_kernels(
        training_kernel_raw,
        test_kernel_raw,
        kernel_normalization,
    )

    classifier = SVC(
        kernel="precomputed",
        C=c,
        random_state=seed,
    )

    classifier.fit(
        training_kernel,
        y_train,
    )

    predictions = classifier.predict(test_kernel)

    return prediction_metrics(
        y_test,
        predictions,
    )


def compute_rbf_metrics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    gamma: str | float,
    seed: int,
) -> dict[str, object]:
    """Compute one fixed-C RBF SVC row for one seed."""
    classifier = SVC(
        kernel="rbf",
        C=c,
        gamma=gamma,
        random_state=seed,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(X_test)

    return prediction_metrics(
        y_test,
        predictions,
    )


def compute_seed_row(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    target_effective_rank: float,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    c: float,
    collapse_threshold: float,
    kernel_normalization: str,
) -> dict[str, object]:
    """Compute default RBF, rank-matched RBF, and QSVM for one seed."""
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

    training_variance = float(np.var(X_train))

    if training_variance <= 0.0:
        raise ValueError("The processed training features have zero variance.")

    gamma_scale = float(1.0 / (X_train.shape[1] * training_variance))

    rbf_scale = compute_rbf_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        c=c,
        gamma="scale",
        seed=seed,
    )

    gamma_star = find_rank_matched_gamma(
        X_train,
        target_effective_rank,
    )

    rbf_star = compute_rbf_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        c=c,
        gamma=gamma_star,
        seed=seed,
    )

    qsvm = compute_qsvm_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        c=c,
        seed=seed,
        kernel_normalization=(kernel_normalization),
    )

    effective_rank_rbf_scale = rbf_effective_rank(
        X_train,
        gamma_scale,
    )

    effective_rank_rbf_star = rbf_effective_rank(
        X_train,
        gamma_star,
    )

    return {
        "source": source,
        "synthetic_surrogate": (source != "real"),
        "model": model,
        "model_display": (
            MODEL_DISPLAY.get(
                model,
                model,
            )
        ),
        "q": q,
        "seed": seed,
        "C": c,
        "kernel_normalization": (kernel_normalization),
        "raw_feature_dimension": int(X.shape[1]),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_validation)),
        "test_samples": int(len(y_test)),
        "train_class_0": int(np.sum(y_train == 0)),
        "train_class_1": int(np.sum(y_train == 1)),
        "val_class_0": int(np.sum(y_validation == 0)),
        "val_class_1": int(np.sum(y_validation == 1)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "pca_variance_percent": float(100.0 * explained_variance_ratio),
        "target_effective_rank": float(target_effective_rank),
        "gamma_scale": gamma_scale,
        "effective_rank_rbf_scale": float(effective_rank_rbf_scale),
        "f1_rbf_scale": float(rbf_scale["f1"]),
        "accuracy_rbf_scale": float(rbf_scale["accuracy"]),
        "predicted_class_1_rbf_scale": int(rbf_scale["predicted_class_1"]),
        "collapsed_rbf_scale": bool(rbf_scale["collapse"]),
        "low_f1_rbf_scale": bool(float(rbf_scale["f1"]) < collapse_threshold),
        "zero_f1_rbf_scale": bool(rbf_scale["zero_f1"]),
        "gamma_star": float(gamma_star),
        "effective_rank_rbf_star": float(effective_rank_rbf_star),
        "f1_rbf_star": float(rbf_star["f1"]),
        "accuracy_rbf_star": float(rbf_star["accuracy"]),
        "predicted_class_1_rbf_star": int(rbf_star["predicted_class_1"]),
        "collapsed_rbf_star": bool(rbf_star["collapse"]),
        "low_f1_rbf_star": bool(float(rbf_star["f1"]) < collapse_threshold),
        "zero_f1_rbf_star": bool(rbf_star["zero_f1"]),
        "f1_qsvm": float(qsvm["f1"]),
        "accuracy_qsvm": float(qsvm["accuracy"]),
        "predicted_class_1_qsvm": int(qsvm["predicted_class_1"]),
        "collapsed_qsvm": bool(qsvm["collapse"]),
        "low_f1_qsvm": bool(float(qsvm["f1"]) < collapse_threshold),
        "zero_f1_qsvm": bool(qsvm["zero_f1"]),
    }


def mean_bool(
    rows: list[dict[str, object]],
    key: str,
) -> float:
    """Return the fraction of rows whose Boolean field is true."""
    return float(np.mean([bool(row[key]) for row in rows]))


def mean_float(
    rows: list[dict[str, object]],
    key: str,
) -> float:
    """Return the finite mean of one numeric field."""
    values = np.asarray(
        [row[key] for row in rows],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def summarize_table10(
    long_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate seed rows into the Table 10 column layout."""
    summary_rows: list[dict[str, object]] = []

    q_values = sorted({int(row["q"]) for row in long_rows})

    for q in q_values:
        rows = [row for row in long_rows if int(row["q"]) == q]

        if not rows:
            raise RuntimeError(f"No seed rows found for q={q}.")

        summary_rows.append(
            {
                "q": q,
                "target_effective_rank": float(rows[0]["target_effective_rank"]),
                "effective_rank_rbf_scale_mean": mean_float(
                    rows,
                    "effective_rank_rbf_scale",
                ),
                "effective_rank_rbf_star_mean": mean_float(
                    rows,
                    "effective_rank_rbf_star",
                ),
                "collapse_rate_rbf_scale": mean_bool(
                    rows,
                    "collapsed_rbf_scale",
                ),
                "collapse_rate_rbf_star": mean_bool(
                    rows,
                    "collapsed_rbf_star",
                ),
                "collapse_rate_qsvm": mean_bool(
                    rows,
                    "collapsed_qsvm",
                ),
                "low_f1_rate_rbf_scale": mean_bool(
                    rows,
                    "low_f1_rbf_scale",
                ),
                "low_f1_rate_rbf_star": mean_bool(
                    rows,
                    "low_f1_rbf_star",
                ),
                "low_f1_rate_qsvm": mean_bool(
                    rows,
                    "low_f1_qsvm",
                ),
                "f1_mean_rbf_scale": mean_float(
                    rows,
                    "f1_rbf_scale",
                ),
                "f1_mean_rbf_star": mean_float(
                    rows,
                    "f1_rbf_star",
                ),
                "f1_mean_qsvm": mean_float(
                    rows,
                    "f1_qsvm",
                ),
                "n_seeds": len(rows),
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


def format_metric(
    value: float,
) -> str:
    """Format one metric with three decimal places."""
    return f"{value:.3f}"


def format_rank(
    value: float,
) -> str:
    """Format one effective-rank value."""
    return f"{value:.2f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 10 artifact."""
    data = payload["data"]

    summary_rows = payload["summary_rows"]

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError("payload data must be a dictionary.")

    if not isinstance(
        summary_rows,
        list,
    ):
        raise TypeError("payload summary_rows must be a list.")

    lines = [
        "# Synthetic surrogate Table 10 pipeline",
        "",
        (
            "This artifact is a surrogate computation only. "
            "It does not reproduce the paper numbers because the gated "
            "MIMIC-CXR embedding dataset is not available locally."
        ),
        "",
        (f"Paper methodology pointer: {payload['paper_pointer']}"),
        "",
        ("Table 10 compares default RBF, rank-matched RBF, and QSVM across seeds."),
        "",
        (
            "Collapse means that the classifier predicts no class-1 "
            "samples. Low F1 means that minority-class F1 is below the "
            "configured threshold. These are reported separately."
        ),
        "",
        (
            f"Model: `{data['model']}`. "
            f"Low-F1 threshold: `{data['collapse_threshold']}`. "
            f"All methods use C=`{data['C']}`."
        ),
        "",
        (
            "| q | Quantum target rank | RBF scale rank | "
            "RBF matched rank | Collapse RBF scale | "
            "Collapse RBF matched | Collapse QSVM | "
            "Low-F1 RBF scale | Low-F1 RBF matched | Low-F1 QSVM | "
            "F1 RBF scale | F1 RBF matched | F1 QSVM |"
        ),
        (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]

    for row in summary_rows:
        if not isinstance(
            row,
            dict,
        ):
            raise TypeError("Each summary row must be a dictionary.")

        lines.append(
            (
                "| {q} | {target_rank} | {scale_rank} | {matched_rank} | "
                "{collapse_scale} | {collapse_matched} | {collapse_qsvm} | "
                "{low_scale} | {low_matched} | {low_qsvm} | "
                "{f1_scale} | {f1_matched} | {f1_qsvm} |"
            ).format(
                q=row["q"],
                target_rank=format_rank(float(row["target_effective_rank"])),
                scale_rank=format_rank(float(row["effective_rank_rbf_scale_mean"])),
                matched_rank=format_rank(float(row["effective_rank_rbf_star_mean"])),
                collapse_scale=format_metric(float(row["collapse_rate_rbf_scale"])),
                collapse_matched=format_metric(float(row["collapse_rate_rbf_star"])),
                collapse_qsvm=format_metric(float(row["collapse_rate_qsvm"])),
                low_scale=format_metric(float(row["low_f1_rate_rbf_scale"])),
                low_matched=format_metric(float(row["low_f1_rate_rbf_star"])),
                low_qsvm=format_metric(float(row["low_f1_rate_qsvm"])),
                f1_scale=format_metric(float(row["f1_mean_rbf_scale"])),
                f1_matched=format_metric(float(row["f1_mean_rbf_star"])),
                f1_qsvm=format_metric(float(row["f1_mean_qsvm"])),
            )
        )

    lines.extend(
        [
            "",
            (
                "The quantum effective-rank target for each q is computed "
                "using the configured rank seed and is held fixed across "
                "the evaluation seeds."
            ),
            "",
            (
                "The rank-matched RBF gamma is selected from training "
                "features only. Test labels are not used."
            ),
            "",
            (
                "Trace normalization applies the training-kernel trace "
                "to both the training and test quantum kernels."
            ),
            "",
            "Data source metadata:",
            "",
            "```json",
            json.dumps(
                data,
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
        return "synthetic_surrogate_table10"

    if source == "synthetic_file":
        return "synthetic_file_table10"

    return "real_table10"


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
        default=DEFAULT_Q_VALUES,
    )

    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--rank-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=COLLAPSE_THRESHOLD,
        help=(
            "Threshold used for the separate low-F1 diagnostic. "
            "Collapse itself means no predicted class-1 samples."
        ),
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "trace",
            "none",
        ),
        default="trace",
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
    seeds: list[int],
) -> None:
    """Validate command-line arguments."""
    if not q_values:
        raise ValueError("At least one q value is required.")

    if any(q <= 0 for q in q_values):
        raise ValueError("All q values must be positive.")

    if len(q_values) != len(set(q_values)):
        raise ValueError("q values must be unique.")

    if not seeds:
        raise ValueError("At least one seed is required.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")

    if args.C <= 0.0:
        raise ValueError("--C must be positive.")

    if not np.isfinite(args.collapse_threshold):
        raise ValueError("--collapse-threshold must be finite.")

    if args.collapse_threshold < 0.0:
        raise ValueError("--collapse-threshold must be non-negative.")

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

    if args.source == "synthetic_file":
        index_path = args.data_root / "synthetic_dataset_index.json"

        if not index_path.is_file():
            raise FileNotFoundError(f"Synthetic dataset index not found: {index_path}")


def main() -> None:
    """Generate the Table 10 CSV, JSON, and Markdown artifacts."""
    args = parse_args()

    q_values = parse_ints(args.q_values)

    seeds = parse_ints(args.seeds)

    validate_args(
        args,
        q_values,
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

    rank_targets: dict[
        int,
        float,
    ] = {}

    for q in q_values:
        rank_targets[q] = compute_rank_target(
            source=args.source,
            model=args.model,
            q=q,
            seed=args.rank_seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )

        print(
            f"[table10] rank target "
            f"model={args.model} "
            f"q={q} "
            f"seed={args.rank_seed} "
            f"rank={rank_targets[q]:.3f}",
            flush=True,
        )

    long_rows: list[dict[str, object]] = []

    for q in q_values:
        for seed in seeds:
            row = compute_seed_row(
                source=args.source,
                model=args.model,
                q=q,
                seed=seed,
                target_effective_rank=(rank_targets[q]),
                data_root=args.data_root,
                synthetic=synthetic,
                c=args.C,
                collapse_threshold=(args.collapse_threshold),
                kernel_normalization=(args.kernel_normalization),
            )

            long_rows.append(row)

            print(
                f"[table10] model={args.model} q={q} seed={seed}",
                flush=True,
            )

    summary_rows = summarize_table10(long_rows)

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

    rank_target_metadata = {str(q): rank_targets[q] for q in q_values}

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 10",
        "paper_pointer": (PAPER_TABLE10_POINTER),
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
            "model": args.model,
            "q_values": q_values,
            "seeds": seeds,
            "rank_seed": (args.rank_seed),
            "C": args.C,
            "collapse_definition": ("No class-1 predictions."),
            "low_f1_threshold": (args.collapse_threshold),
            "kernel_normalization": (args.kernel_normalization),
            "normalization_protocol": (
                "The training-kernel trace is applied consistently "
                "to the training and test quantum kernels."
            ),
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "rank_targets": (rank_target_metadata),
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

    console_summary = {
        "rows": len(summary_rows),
        "seed_rows": len(long_rows),
        "model": args.model,
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
