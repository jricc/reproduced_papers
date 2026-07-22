#!/usr/bin/env python3
"""Compute a Table 3-style QSVM confusion matrix on surrogate or real data.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 3 protocol shape: one representative MedSigLIP
QSVM confusion matrix at q=11, seed 0, and C=1.

The QSVM training and test kernels use the same trace-normalization factor,
computed from the training kernel.
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
    normalize_train_test_kernels,
    preprocess,
    split_indices,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    load_dataset,
)

PAPER_TABLE3_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T3"

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)


def predict_qsvm_c1(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Predict with the Table 3 QSVM using fixed C=1."""
    training_kernel = fidelity_kernel(X_train)

    test_kernel = fidelity_kernel(
        X_test,
        X_train,
    )

    (
        training_kernel,
        test_kernel,
    ) = normalize_train_test_kernels(
        training_kernel,
        test_kernel,
        method="trace",
    )

    classifier = SVC(
        kernel="precomputed",
        C=1.0,
        random_state=seed,
    )

    classifier.fit(
        training_kernel,
        y_train,
    )

    predictions = classifier.predict(test_kernel)

    return np.asarray(
        predictions,
        dtype=int,
    )


def compute_table3(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> dict[str, object]:
    """Compute the representative Table 3 confusion matrix."""
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

    y_pred = predict_qsvm_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        seed=seed,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    predicted_class_0 = int(np.sum(y_pred == 0))

    predicted_class_1 = int(np.sum(y_pred == 1))

    return {
        "source": source,
        "synthetic_surrogate": (source != "real"),
        "model": model,
        "q": q,
        "seed": seed,
        "C": 1.0,
        "kernel_normalization": "trace",
        "train_samples": int(len(training_indices)),
        "val_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "train_class_0": int(np.sum(y_train == 0)),
        "train_class_1": int(np.sum(y_train == 1)),
        "val_class_0": int(np.sum(y_validation == 0)),
        "val_class_1": int(np.sum(y_validation == 1)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "predicted_class_0": (predicted_class_0),
        "predicted_class_1": (predicted_class_1),
        "collapse": bool(predicted_class_1 == 0),
        "explained_variance_ratio": float(explained_variance_ratio),
        "confusion_matrix": (matrix.astype(int).tolist()),
        "accuracy": float(
            accuracy_score(
                y_test,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
    }


def confusion_rows(
    table3: dict[str, object],
) -> list[dict[str, object]]:
    """Convert the confusion matrix to labelled CSV rows."""
    matrix = table3["confusion_matrix"]

    if not isinstance(
        matrix,
        list,
    ):
        raise TypeError("confusion_matrix must be a list.")

    if len(matrix) != 2 or not all(
        isinstance(row, list) and len(row) == 2 for row in matrix
    ):
        raise ValueError("confusion_matrix must have shape 2 by 2.")

    labels = (
        "class_0_majority",
        "class_1_minority",
    )

    return [
        {
            "true_label": labels[0],
            "pred_class_0_majority": (matrix[0][0]),
            "pred_class_1_minority": (matrix[0][1]),
        },
        {
            "true_label": labels[1],
            "pred_class_0_majority": (matrix[1][0]),
            "pred_class_1_minority": (matrix[1][1]),
        },
    ]


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the confusion matrix rows to CSV."""
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
    """Write the human-readable Table 3 artifact."""
    table3 = payload["table3"]

    if not isinstance(
        table3,
        dict,
    ):
        raise TypeError("payload table3 must be a dictionary.")

    rows = confusion_rows(table3)

    lines = [
        "# Synthetic surrogate Table 3 pipeline",
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
            "Table 3 reports a representative MedSigLIP QSVM "
            "confusion matrix at q=11 and seed 0. Results obtained "
            "from synthetic data are not expected to match the "
            "paper values."
        ),
        "",
        ("| True label | Pred class 0 majority | Pred class 1 minority |"),
        "| --- | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['true_label']} "
            f"| {row['pred_class_0_majority']} "
            f"| {row['pred_class_1_minority']} |"
        )

    lines.extend(
        [
            "",
            (f"Accuracy: {table3['accuracy']:.6f}"),
            (f"Precision: {table3['precision']:.6f}"),
            (f"Recall: {table3['recall']:.6f}"),
            (f"F1: {table3['f1']:.6f}"),
            (f"Predicted class 1 samples: {table3['predicted_class_1']}"),
            (f"Majority-only collapse: {table3['collapse']}"),
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
        return "synthetic_surrogate_table3"

    if source == "synthetic_file":
        return "synthetic_file_table3"

    return "real_table3"


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
        default=11,
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
    if args.q <= 0:
        raise ValueError("--q must be positive.")

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
    """Compute and write the Table 3-style artifact."""
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

    table3 = compute_table3(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
    )

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = args.results_dir / f"{prefix}_confusion_matrix.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 3",
        "paper_pointer": (PAPER_TABLE3_POINTER),
        "paths": {
            "confusion_matrix_csv": str(csv_path),
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
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "kernel_normalization": ("trace"),
            "normalization_protocol": (
                "The training and test kernels are divided by the "
                "training-kernel trace."
            ),
        },
        "table3": table3,
    }

    write_csv(
        csv_path,
        confusion_rows(table3),
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

    write_markdown(
        markdown_path,
        payload=payload,
    )

    print(
        json.dumps(
            table3,
            indent=2,
            sort_keys=True,
        )
    )

    print(f"Wrote {csv_path}")

    print(f"Wrote {json_path}")

    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
