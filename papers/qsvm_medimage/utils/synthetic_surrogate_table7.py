#!/usr/bin/env python3
"""Compute a Table 7-style QSVM kernel-normalization comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 7 protocol shape:

- q=8;
- one circuit repetition;
- SVM regularization C=1;
- seed 0;
- three embedding models;
- four kernel normalizations.

The four normalization methods are:

- trace;
- none;
- cosine;
- Frobenius.

For scalar normalizations, the scale derived from the training kernel is also
applied to the test kernel. For cosine normalization, training self-kernel
values and test self-kernel values are used consistently.

When synthetic data are used, the results describe a controlled surrogate
benchmark. They do not reproduce the paper's numerical results on the
inaccessible medical embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
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


from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    compute_metrics,
    load_dataset,
)

PAPER_TABLE7_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T7"

TABLE7_MODELS = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

TABLE7_NORMALIZATIONS = (
    "trace",
    "none",
    "cosine",
    "frobenius",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def safe_scale(
    value: float,
) -> float:
    """Return a positive scale suitable for division."""
    if not np.isfinite(value):
        raise ValueError("Kernel normalization scale must be finite.")

    if value <= 0.0:
        raise ValueError("Kernel normalization scale must be positive.")

    return float(value)


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
    *,
    K_test_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one normalization consistently to train and test kernels."""
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
        scale = safe_scale(float(np.trace(training_kernel)))

        return (
            training_kernel / scale,
            test_kernel / scale,
        )

    if normalization == "frobenius":
        scale = safe_scale(
            float(
                np.linalg.norm(
                    training_kernel,
                    ord="fro",
                )
            )
        )

        return (
            training_kernel / scale,
            test_kernel / scale,
        )

    if normalization == "cosine":
        training_diagonal = np.diag(training_kernel)

        if np.any(training_diagonal <= 0.0):
            raise ValueError(
                "K_train diagonal values must be positive for cosine normalization."
            )

        if K_test_diag is None:
            test_diagonal = np.ones(
                test_kernel.shape[0],
                dtype=np.float64,
            )
        else:
            test_diagonal = np.asarray(
                K_test_diag,
                dtype=np.float64,
            )

        if test_diagonal.ndim != 1:
            raise ValueError("K_test_diag must be one-dimensional.")

        if len(test_diagonal) != test_kernel.shape[0]:
            raise ValueError("K_test_diag must contain one value per test sample.")

        if not np.all(np.isfinite(test_diagonal)):
            raise ValueError("K_test_diag contains non-finite values.")

        if np.any(test_diagonal <= 0.0):
            raise ValueError(
                "K_test_diag values must be positive for cosine normalization."
            )

        training_norms = np.sqrt(training_diagonal)

        test_norms = np.sqrt(test_diagonal)

        training_scale = np.outer(
            training_norms,
            training_norms,
        )

        test_scale = np.outer(
            test_norms,
            training_norms,
        )

        return (
            training_kernel / training_scale,
            test_kernel / test_scale,
        )

    raise ValueError(f"Unknown normalization: {normalization!r}.")


def score_qsvm_normalization(
    *,
    K_train_raw: np.ndarray,
    K_test_raw: np.ndarray,
    K_test_diag: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    normalization: str,
    c: float,
    seed: int,
) -> dict[str, object]:
    """Train the fixed-C QSVM for one kernel normalization."""
    (
        training_kernel,
        test_kernel,
    ) = normalize_train_test_kernels(
        K_train_raw,
        K_test_raw,
        normalization,
        K_test_diag=K_test_diag,
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

    scores = classifier.decision_function(test_kernel)

    metrics = compute_metrics(
        y_test,
        predictions,
        scores,
    )

    result: dict[str, object] = dict(metrics)

    predicted_class_0 = int(np.sum(predictions == 0))

    predicted_class_1 = int(np.sum(predictions == 1))

    result["predicted_class_0"] = predicted_class_0

    result["predicted_class_1"] = predicted_class_1

    result["collapse"] = bool(predicted_class_1 == 0)

    result["zero_f1"] = bool(
        np.isclose(
            float(metrics["f1"]),
            0.0,
        )
    )

    return result


def compute_table7_rows(
    *,
    source: str,
    model: str,
    q: int,
    reps: int,
    c: float,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> list[dict[str, object]]:
    """Compute all Table 7 normalization rows for one embedding model."""
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

    training_kernel_raw = fidelity_kernel(
        X_train,
        reps=reps,
    )

    test_kernel_raw = fidelity_kernel(
        X_test,
        X_train,
        reps=reps,
    )

    # A fidelity kernel has unit self-similarity for every exact state.
    test_kernel_diagonal = np.ones(
        len(y_test),
        dtype=np.float64,
    )

    rows: list[dict[str, object]] = []

    for normalization in TABLE7_NORMALIZATIONS:
        metrics = score_qsvm_normalization(
            K_train_raw=(training_kernel_raw),
            K_test_raw=(test_kernel_raw),
            K_test_diag=(test_kernel_diagonal),
            y_train=y_train,
            y_test=y_test,
            normalization=normalization,
            c=c,
            seed=seed,
        )

        rows.append(
            {
                "source": source,
                "synthetic_surrogate": (source != "real"),
                "model": model,
                "model_display": (MODEL_DISPLAY[model]),
                "normalization": (normalization),
                "q": q,
                "reps": reps,
                "C": c,
                "seed": seed,
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
                "accuracy": float(metrics["accuracy"]),
                "auc": float(metrics["auc"]),
                "f1": float(metrics["f1"]),
                "predicted_class_0": int(metrics["predicted_class_0"]),
                "predicted_class_1": int(metrics["predicted_class_1"]),
                "collapse": bool(metrics["collapse"]),
                "zero_f1": bool(metrics["zero_f1"]),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the Table 7 summary rows to CSV."""
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


def format_metric(
    value: float,
) -> str:
    """Format one classification metric."""
    return f"{value:.3f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 7 artifact."""
    summary_rows = payload["summary_rows"]

    if not isinstance(
        summary_rows,
        list,
    ):
        raise TypeError("payload summary_rows must be a list.")

    lines = [
        "# Synthetic surrogate Table 7 pipeline",
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
            "Table 7 compares QSVM test metrics for q=8, one circuit "
            "repetition, C=1, and seed 0 under four kernel normalizations."
        ),
        "",
        (
            "Each normalization is applied consistently to the training "
            "and test kernels."
        ),
        "",
        ("| Model | Normalization | Accuracy | AUC | F1 | Pred. class 1 | Collapse |"),
        ("| --- | --- | ---: | ---: | ---: | ---: | --- |"),
    ]

    for row in summary_rows:
        lines.append(
            (
                "| {model} | {normalization} | {accuracy} | "
                "{auc} | {f1} | {predicted_class_1} | {collapse} |"
            ).format(
                model=row["model_display"],
                normalization=row["normalization"],
                accuracy=format_metric(row["accuracy"]),
                auc=format_metric(row["auc"]),
                f1=format_metric(row["f1"]),
                predicted_class_1=row["predicted_class_1"],
                collapse=row["collapse"],
            )
        )

    lines.extend(
        [
            "",
            (
                "Trace and Frobenius normalization multiply the complete "
                "train and test kernels by one training-derived scalar."
            ),
            "",
            (
                "Cosine normalization uses training self-similarities for "
                "the training columns and test self-similarities for the "
                "test rows."
            ),
            "",
            (
                "For an exact fidelity kernel, every self-similarity is one. "
                "Cosine normalization should therefore be numerically close "
                "to the unnormalized kernel."
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
        return "synthetic_surrogate_table7"

    if source == "synthetic_file":
        return "synthetic_file_table7"

    return "real_table7"


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
        "--q",
        type=int,
        default=8,
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
        raise ValueError("--q must be positive.")

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
    """Compute and write the Table 7-style artifact."""
    args = parse_args()

    validate_args(args)

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    summary_rows: list[dict[str, object]] = []

    for model in TABLE7_MODELS:
        model_rows = compute_table7_rows(
            source=args.source,
            model=model,
            q=args.q,
            reps=args.reps,
            c=args.C,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )

        summary_rows.extend(model_rows)

        print(
            f"[table7] model={model} q={args.q} seed={args.seed}",
            flush=True,
        )

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    write_csv(
        summary_path,
        summary_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 7",
        "paper_pointer": (PAPER_TABLE7_POINTER),
        "paths": {
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
            "seed": args.seed,
            "q": args.q,
            "reps": args.reps,
            "C": args.C,
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "preprocessing": (
                "StandardScaler fitted on train, PCA fitted on train, "
                "MinMaxScaler fitted on train with output range [-1, 1]"
            ),
            "models": list(TABLE7_MODELS),
            "normalizations": list(TABLE7_NORMALIZATIONS),
            "normalization_protocol": (
                "Trace and Frobenius scales are computed from the training "
                "kernel and applied to both training and test kernels. "
                "Cosine normalization uses the corresponding train and test "
                "self-kernel values."
            ),
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
        "seed": args.seed,
        "q": args.q,
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
        )
    )

    print(f"Wrote {summary_path}")

    print(f"Wrote {json_path}")

    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
