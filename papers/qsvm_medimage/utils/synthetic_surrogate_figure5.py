#!/usr/bin/env python3
"""Generate a Figure 5-style QSVM sweep over the qubit count q.

The default protocol uses:

- q values 2, 3, 4, 5, 6, and 8;
- one PCA component per qubit;
- the manuscript quantum fidelity kernel;
- Ry feature encoding followed by a CNOT ring;
- one circuit repetition;
- SVM regularization C=1;
- trace normalization;
- seed 0.

For every embedding model and q value, preprocessing is fitted only on the
training split:

    StandardScaler
    PCA(q)
    MinMaxScaler[-1, 1]

The same normalization scale is applied to the training and test kernels.

When synthetic data are used, this artifact reproduces the experimental
structure of Figure 5. It does not reproduce the numerical results obtained
from the inaccessible medical embeddings.

Increasing q changes both the PCA dimension and the quantum-state dimension.
The resulting curves are descriptive. They do not by themselves establish
that more qubits improve classification.
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
from sklearn.svm import SVC

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
    compute_metrics,
    decision_scores,
    load_dataset,
    parse_ints,
)

PAPER_FIGURE5_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F5"

FIGURE5_MODELS = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}

MODEL_STYLE = {
    "medsiglip-448": {
        "color": "#2166ac",
        "marker": "o",
    },
    "rad-dino": {
        "color": "#e66101",
        "marker": "s",
    },
    "vit-patch32-cls": {
        "color": "#4dac26",
        "marker": "^",
    },
}


def validate_kernel_pair(
    K_train: np.ndarray,
    K_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one training and test kernel pair."""
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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize training and test kernels consistently.

    For trace normalization:

        K_train_normalized = K_train / trace(K_train)

        K_test_normalized = K_test / trace(K_train)

    For Frobenius normalization, both matrices are divided by the Frobenius
    norm of the training kernel.

    For cosine normalization, the training kernel is normalized with its
    diagonal. For a fidelity kernel, all self-similarities equal one, so this
    normalization normally leaves both kernels unchanged.
    """
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
            1.0,
        )

    if normalization == "trace":
        scale = float(np.trace(training_kernel))

        if scale <= 0.0:
            raise ValueError("Training-kernel trace must be positive.")

        return (
            training_kernel / scale,
            test_kernel / scale,
            scale,
        )

    if normalization == "frobenius":
        scale = float(
            np.linalg.norm(
                training_kernel,
                ord="fro",
            )
        )

        if scale <= 0.0:
            raise ValueError("Training-kernel Frobenius norm must be positive.")

        return (
            training_kernel / scale,
            test_kernel / scale,
            scale,
        )

    if normalization == "cosine":
        training_diagonal = np.diag(training_kernel)

        if np.any(training_diagonal <= 0.0):
            raise ValueError(
                "Training-kernel diagonal must be positive for cosine normalization."
            )

        training_scale = np.sqrt(
            np.outer(
                training_diagonal,
                training_diagonal,
            )
        )

        normalized_training = training_kernel / training_scale

        # Every test self-fidelity is one for an exact fidelity kernel.
        test_scale = np.sqrt(training_diagonal)[None, :]

        normalized_test = test_kernel / test_scale

        return (
            normalized_training,
            normalized_test,
            1.0,
        )

    raise ValueError(f"Unknown normalization: {normalization!r}.")


def effective_rank_from_kernel(
    kernel: np.ndarray,
) -> float:
    """Compute Shannon effective rank from a square kernel."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError("kernel must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be square.")

    if matrix.shape[0] == 0:
        raise ValueError("kernel must not be empty.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("kernel contains non-finite values.")

    matrix = 0.5 * (matrix + matrix.T)

    eigenvalues = np.linalg.eigvalsh(matrix)

    minimum_eigenvalue = float(np.min(eigenvalues))

    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "Kernel is not positive semidefinite within tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}."
        )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    eigenvalue_sum = float(np.sum(eigenvalues))

    if eigenvalue_sum <= 0.0:
        raise ValueError("Kernel has no positive eigenvalue mass.")

    probabilities = eigenvalues / eigenvalue_sum

    probabilities = probabilities[probabilities > 1e-15]

    entropy = -np.sum(probabilities * np.log(probabilities))

    return float(np.exp(entropy))


def positive_rank_from_kernel(
    kernel: np.ndarray,
) -> int:
    """Count numerically positive kernel eigenvalues."""
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError("kernel must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be square.")

    if matrix.shape[0] == 0:
        raise ValueError("kernel must not be empty.")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("kernel contains non-finite values.")

    matrix = 0.5 * (matrix + matrix.T)

    eigenvalues = np.linalg.eigvalsh(matrix)

    maximum_absolute_value = float(np.max(np.abs(eigenvalues)))

    tolerance = max(
        1e-12,
        1e-10 * maximum_absolute_value,
    )

    return int(np.sum(eigenvalues > tolerance))


def score_qsvm_for_q(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    q: int,
    reps: int,
    c: float,
    seed: int,
    kernel_normalization: str,
) -> dict[str, object]:
    """Train and evaluate one fixed-C QSVM configuration."""
    if X_train.ndim != 2:
        raise ValueError("X_train must be two-dimensional.")

    if X_test.ndim != 2:
        raise ValueError("X_test must be two-dimensional.")

    if X_train.shape[1] != q:
        raise ValueError(f"X_train has {X_train.shape[1]} features, but q={q}.")

    if X_test.shape[1] != q:
        raise ValueError(f"X_test has {X_test.shape[1]} features, but q={q}.")

    raw_training_kernel = fidelity_kernel(
        X_train,
        reps=reps,
    )

    raw_test_kernel = fidelity_kernel(
        X_test,
        X_train,
        reps=reps,
    )

    raw_effective_rank = effective_rank_from_kernel(raw_training_kernel)

    raw_positive_rank = positive_rank_from_kernel(raw_training_kernel)

    (
        training_kernel,
        test_kernel,
        normalization_scale,
    ) = normalize_train_test_kernels(
        raw_training_kernel,
        raw_test_kernel,
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

    scores = decision_scores(
        classifier,
        test_kernel,
    )

    metrics = compute_metrics(
        y_test,
        predictions,
        scores,
    )

    result = dict(metrics)

    result["q"] = q

    result["effective_rank"] = float(raw_effective_rank)

    result["positive_rank"] = int(raw_positive_rank)

    result["normalization_scale"] = float(normalization_scale)

    result["predicted_class_0"] = int(np.sum(predictions == 0))

    result["predicted_class_1"] = int(np.sum(predictions == 1))

    result["collapse"] = bool(np.sum(predictions == 1) == 0)

    result["zero_f1"] = bool(
        np.isclose(
            float(metrics["f1"]),
            0.0,
        )
    )

    return result


def compute_model_sweep_rows(
    *,
    source: str,
    model: str,
    q_values: list[int],
    reps: int,
    c: float,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    kernel_normalization: str,
) -> list[dict[str, object]]:
    """Compute Figure 5 QSVM rows for one embedding model."""
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

    y_train = y[training_indices]

    y_validation = y[validation_indices]

    y_test = y[test_indices]

    rows: list[dict[str, object]] = []

    for q in q_values:
        if q > X.shape[1]:
            raise ValueError(
                f"q={q} exceeds the raw embedding dimension "
                f"{X.shape[1]} for model {model}."
            )

        (
            X_train,
            X_validation,
            X_test,
            explained_variance_ratio,
        ) = preprocess(
            X[training_indices],
            X[validation_indices],
            X[test_indices],
            q,
        )

        if X_validation.shape[1] != q:
            raise RuntimeError("The processed validation dimension is inconsistent.")

        metrics = score_qsvm_for_q(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            q=q,
            reps=reps,
            c=c,
            seed=seed,
            kernel_normalization=(kernel_normalization),
        )

        row = {
            "source": source,
            "synthetic_surrogate": (source != "real"),
            "model": model,
            "model_display": (MODEL_DISPLAY[model]),
            "q": q,
            "pca_dimension": q,
            "qubit_count": q,
            "reps": reps,
            "C": c,
            "seed": seed,
            "kernel_normalization": (kernel_normalization),
            "train_samples": int(len(y_train)),
            "validation_samples": int(len(y_validation)),
            "test_samples": int(len(y_test)),
            "train_class_0": int(np.sum(y_train == 0)),
            "train_class_1": int(np.sum(y_train == 1)),
            "validation_class_0": int(np.sum(y_validation == 0)),
            "validation_class_1": int(np.sum(y_validation == 1)),
            "test_class_0": int(np.sum(y_test == 0)),
            "test_class_1": int(np.sum(y_test == 1)),
            "pca_variance_percent": float(100.0 * explained_variance_ratio),
            "accuracy": float(metrics["accuracy"]),
            "auc": float(metrics["auc"]),
            "f1": float(metrics["f1"]),
            "precision": float(
                metrics.get(
                    "precision",
                    float("nan"),
                )
            ),
            "recall": float(
                metrics.get(
                    "recall",
                    float("nan"),
                )
            ),
            "effective_rank": float(metrics["effective_rank"]),
            "positive_rank": int(metrics["positive_rank"]),
            "normalization_scale": float(metrics["normalization_scale"]),
            "predicted_class_0": int(metrics["predicted_class_0"]),
            "predicted_class_1": int(metrics["predicted_class_1"]),
            "collapse": bool(metrics["collapse"]),
            "zero_f1": bool(metrics["zero_f1"]),
        }

        rows.append(row)

        print(
            f"[figure5] model={model} "
            f"q={q} "
            f"accuracy={row['accuracy']:.3f} "
            f"auc={row['auc']:.3f} "
            f"f1={row['f1']:.3f} "
            f"effective_rank="
            f"{row['effective_rank']:.2f}",
            flush=True,
        )

    return rows


def metric_series(
    rows: list[dict[str, object]],
    *,
    model: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    """Return one model's metric values sorted by q."""
    selected_rows = [row for row in rows if row["model"] == model]

    selected_rows.sort(key=lambda row: int(row["q"]))

    q_axis = [int(row["q"]) for row in selected_rows]

    metric_values = [float(row[metric]) for row in selected_rows]

    return (
        q_axis,
        metric_values,
    )


def padded_ylim(
    values: list[float],
    *,
    lower_floor: float = 0.0,
    upper_ceiling: float = 1.0,
) -> tuple[float, float]:
    """Return y-axis limits with a small visible margin."""
    finite_values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        return (
            lower_floor,
            upper_ceiling,
        )

    minimum_value = float(np.min(finite_values))

    maximum_value = float(np.max(finite_values))

    if np.isclose(
        minimum_value,
        maximum_value,
    ):
        padding = 0.05
    else:
        padding = 0.08 * (maximum_value - minimum_value)

    lower_limit = max(
        lower_floor,
        minimum_value - padding,
    )

    upper_limit = min(
        upper_ceiling,
        maximum_value + padding,
    )

    if np.isclose(
        lower_limit,
        upper_limit,
    ):
        upper_limit = min(
            upper_ceiling,
            lower_limit + 0.05,
        )

    return (
        lower_limit,
        upper_limit,
    )


def save_plot(
    path: Path,
    *,
    rows: list[dict[str, object]],
    q_values: list[int],
) -> None:
    """Save accuracy, ROC-AUC, and minority-F1 sweep plots."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            17,
            4.8,
        ),
    )

    metric_configuration = (
        (
            "accuracy",
            "Test accuracy",
            axes[0],
        ),
        (
            "auc",
            "Test ROC-AUC",
            axes[1],
        ),
        (
            "f1",
            "Minority-class F1",
            axes[2],
        ),
    )

    for (
        metric,
        ylabel,
        axis,
    ) in metric_configuration:
        for model in FIGURE5_MODELS:
            (
                model_q_values,
                model_metric_values,
            ) = metric_series(
                rows,
                model=model,
                metric=metric,
            )

            style = MODEL_STYLE[model]

            axis.plot(
                model_q_values,
                model_metric_values,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.8,
                markersize=5.5,
                label=MODEL_DISPLAY[model],
            )

        all_metric_values = [float(row[metric]) for row in rows]

        axis.set_xlabel("PCA dimension and qubit count q")

        axis.set_ylabel(ylabel)

        axis.set_xticks(q_values)

        (
            lower_limit,
            upper_limit,
        ) = padded_ylim(
            all_metric_values,
            lower_floor=0.0,
            upper_ceiling=1.0,
        )

        axis.set_ylim(
            lower_limit,
            upper_limit,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.legend(
            fontsize=8,
        )

    axes[1].axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Random-ranking reference",
    )

    axes[1].legend(
        fontsize=8,
    )

    figure.suptitle(
        ("QSVM performance versus PCA dimension and qubit count"),
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

    plt.close(figure)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write summary rows to CSV."""
    if not rows:
        raise ValueError(f"No rows to write to {path}.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0])

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
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Figure 5 artifact description."""
    paths = payload["paths"]

    summary_rows = payload["summary_rows"]

    if not isinstance(
        paths,
        dict,
    ):
        raise TypeError("payload paths must be a dictionary.")

    if not isinstance(
        summary_rows,
        list,
    ):
        raise TypeError("payload summary_rows must be a list.")

    image_name = Path(paths["png"]).name

    lines = [
        "# Figure 5-style QSVM qubit sweep",
        "",
        (
            "This artifact evaluates the manuscript QSVM as the PCA "
            "dimension and qubit count q change."
        ),
        "",
        (
            "Results generated from synthetic data reproduce the "
            "experimental structure of Figure 5, not the paper's "
            "numerical result on the inaccessible medical embeddings."
        ),
        "",
        (
            "Increasing q simultaneously changes the PCA representation, "
            "the number of qubits, and the Hilbert-space dimension. "
            "The curves do not isolate a single causal effect."
        ),
        "",
        (f"Paper methodology pointer: {payload['paper_pointer']}"),
        "",
        (f"![Figure 5-style QSVM qubit sweep]({image_name})"),
        "",
        ("| Model | q | PCA var. | Accuracy | AUC | F1 | Eff. rank | Collapse |"),
        ("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"),
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['model_display']} "
            f"| {row['q']} "
            f"| {row['pca_variance_percent']:.2f}% "
            f"| {row['accuracy']:.3f} "
            f"| {row['auc']:.3f} "
            f"| {row['f1']:.3f} "
            f"| {row['effective_rank']:.2f} "
            f"| {row['collapse']} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            (
                "- Accuracy can remain high when minority-class F1 is "
                "zero because the task is imbalanced."
            ),
            (
                "- ROC-AUC evaluates ranking across thresholds and must "
                "be interpreted separately from thresholded F1."
            ),
            (
                "- Effective rank describes the kernel eigenspectrum and "
                "does not by itself establish classification quality."
            ),
            (
                "- Majority-only collapse means that the classifier "
                "predicts no samples of label 1."
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
        return "synthetic_surrogate_figure5"

    if source == "synthetic_file":
        return "synthetic_file_figure5"

    return "real_figure5"


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
        "--q-values",
        default="2,3,4,5,6,8",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "trace",
            "none",
            "cosine",
            "frobenius",
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
) -> None:
    """Validate command-line argument consistency."""
    if not q_values:
        raise ValueError("At least one q value is required.")

    if any(q <= 0 for q in q_values):
        raise ValueError("All q values must be positive.")

    if len(q_values) != len(set(q_values)):
        raise ValueError("q values must be unique.")

    if args.reps <= 0:
        raise ValueError("--reps must be positive.")

    if args.C <= 0.0:
        raise ValueError("--C must be positive.")

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
    """Generate the Figure 5-style artifact."""
    args = parse_args()

    q_values = parse_ints(args.q_values)

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

    summary_rows: list[dict[str, object]] = []

    for model in FIGURE5_MODELS:
        model_rows = compute_model_sweep_rows(
            source=args.source,
            model=model,
            q_values=q_values,
            reps=args.reps,
            c=args.C,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
            kernel_normalization=(args.kernel_normalization),
        )

        summary_rows.extend(model_rows)

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = args.results_dir / f"{prefix}.png"

    csv_path = args.results_dir / f"{prefix}_summary.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    write_csv(
        csv_path,
        summary_rows,
    )

    save_plot(
        png_path,
        rows=summary_rows,
        q_values=q_values,
    )

    scope = {
        "kernel": ("Manuscript quantum fidelity kernel"),
        "quantum_feature_map": ("Ry encoding followed by a CNOT ring"),
        "pca_dimension_equals_qubit_count": True,
        "synthetic_result_is_paper_result": False,
        "increasing_q_isolated_causal_effect": False,
    }

    data_metadata = {
        "source": args.source,
        "synthetic_surrogate": (args.source != "real"),
        "synthetic_spec": (asdict(synthetic) if args.source == "synthetic" else None),
        "data_root": (str(args.data_root) if args.data_root else None),
        "seed": args.seed,
        "q_values": q_values,
        "reps": args.reps,
        "C": args.C,
        "kernel_normalization": (args.kernel_normalization),
        "normalization_protocol": (
            "The training-kernel normalization scale is applied "
            "consistently to the test kernel."
        ),
        "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
        "preprocessing": (
            "StandardScaler fitted on train, PCA fitted on train, "
            "MinMaxScaler fitted on train with output range [-1, 1]"
        ),
        "models": list(FIGURE5_MODELS),
    }

    paths = {
        "png": str(png_path),
        "summary_csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(markdown_path),
    }

    payload: dict[
        str,
        object,
    ] = {
        "artifact": prefix,
        "paper_figure": "Figure 5",
        "paper_pointer": (PAPER_FIGURE5_POINTER),
        "scope": scope,
        "paths": paths,
        "data": data_metadata,
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
        "seed": args.seed,
        "q_values": q_values,
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
        )
    )

    print(f"Wrote {png_path}")

    print(f"Wrote {csv_path}")

    print(f"Wrote {json_path}")

    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
