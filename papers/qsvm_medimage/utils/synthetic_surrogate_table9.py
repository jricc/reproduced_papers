#!/usr/bin/env python3
"""Compute a Table 9-style q=16 QSVM C-tuning comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 9 protocol shape:

- target q=16;
- reference q=8;
- one circuit repetition;
- seed 0;
- trace-normalized QSVM;
- C selected using validation minority-class F1.

The training, validation, and test kernels use the same normalization factor,
derived from the training kernel.

The reference result is recomputed on the same data source with the same
validation-selection procedure. It is not copied from the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
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


from lib.quantum_kernel import feature_states  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    compute_metrics,
    load_dataset,
    parse_floats,
)

PAPER_TABLE9_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T9"

TABLE9_MODELS = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP-448",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-patch32-cls",
}


def validate_kernel_triplet(
    K_train: np.ndarray,
    K_val: np.ndarray,
    K_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate training, validation, and test kernel matrices."""
    training_kernel = np.asarray(
        K_train,
        dtype=np.float64,
    )

    validation_kernel = np.asarray(
        K_val,
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

    if validation_kernel.ndim != 2:
        raise ValueError("K_val must be two-dimensional.")

    if test_kernel.ndim != 2:
        raise ValueError("K_test must be two-dimensional.")

    if validation_kernel.shape[1] != training_kernel.shape[0]:
        raise ValueError("K_val must have one column per training sample.")

    if test_kernel.shape[1] != training_kernel.shape[0]:
        raise ValueError("K_test must have one column per training sample.")

    if not np.all(np.isfinite(training_kernel)):
        raise ValueError("K_train contains non-finite values.")

    if not np.all(np.isfinite(validation_kernel)):
        raise ValueError("K_val contains non-finite values.")

    if not np.all(np.isfinite(test_kernel)):
        raise ValueError("K_test contains non-finite values.")

    training_kernel = 0.5 * (training_kernel + training_kernel.T)

    return (
        training_kernel,
        validation_kernel,
        test_kernel,
    )


def normalize_kernel_triplet(
    K_train: np.ndarray,
    K_val: np.ndarray,
    K_test: np.ndarray,
    normalization: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Normalize train, validation, and test kernels consistently."""
    (
        training_kernel,
        validation_kernel,
        test_kernel,
    ) = validate_kernel_triplet(
        K_train,
        K_val,
        K_test,
    )

    if normalization == "none":
        return (
            training_kernel.copy(),
            validation_kernel.copy(),
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
            validation_kernel / scale,
            test_kernel / scale,
        )

    raise ValueError(f"Unknown normalization: {normalization!r}.")


def qsvm_kernels(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    reps: int,
    kernel_normalization: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute train, validation, and test kernels once for C tuning."""
    training_states = feature_states(
        X_train,
        reps=reps,
    )

    validation_states = feature_states(
        X_val,
        reps=reps,
    )

    test_states = feature_states(
        X_test,
        reps=reps,
    )

    training_overlaps = training_states.conj() @ training_states.T

    validation_overlaps = validation_states.conj() @ training_states.T

    test_overlaps = test_states.conj() @ training_states.T

    training_kernel_raw = np.square(np.abs(training_overlaps))

    validation_kernel_raw = np.square(np.abs(validation_overlaps))

    test_kernel_raw = np.square(np.abs(test_overlaps))

    training_kernel_raw = 0.5 * (training_kernel_raw + training_kernel_raw.T)

    diagonal_error = float(np.max(np.abs(np.diag(training_kernel_raw) - 1.0)))

    if diagonal_error > 1e-10:
        raise RuntimeError(
            "The feature states are not normalized. "
            "Maximum training-kernel diagonal error: "
            f"{diagonal_error:.3e}."
        )

    np.fill_diagonal(
        training_kernel_raw,
        1.0,
    )

    return normalize_kernel_triplet(
        training_kernel_raw,
        validation_kernel_raw,
        test_kernel_raw,
        kernel_normalization,
    )


def select_best_c_by_validation_f1(
    tuning_rows: list[dict[str, object]],
) -> float:
    """Select highest validation F1, using smaller C as the tie-break."""
    if not tuning_rows:
        raise ValueError("C tuning rows must not be empty.")

    best_row = tuning_rows[0]

    for row in tuning_rows[1:]:
        row_f1 = float(row["val_f1"])

        best_f1 = float(best_row["val_f1"])

        row_c = float(row["C"])

        best_c = float(best_row["C"])

        better_f1 = row_f1 > best_f1 and not np.isclose(
            row_f1,
            best_f1,
        )

        tied_f1_with_smaller_c = (
            np.isclose(
                row_f1,
                best_f1,
            )
            and row_c < best_c
        )

        if better_f1 or tied_f1_with_smaller_c:
            best_row = row

    return float(best_row["C"])


def score_qsvm_best_c(
    *,
    K_train: np.ndarray,
    K_val: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """Tune C on validation F1, then score the selected QSVM on test."""
    if not c_grid:
        raise ValueError("c_grid must not be empty.")

    tuning_rows: list[dict[str, object]] = []

    for c_value in c_grid:
        classifier = SVC(
            kernel="precomputed",
            C=c_value,
            random_state=seed,
        )

        classifier.fit(
            K_train,
            y_train,
        )

        validation_predictions = classifier.predict(K_val)

        validation_f1 = float(
            f1_score(
                y_val,
                validation_predictions,
                zero_division=0,
            )
        )

        tuning_rows.append(
            {
                "C": float(c_value),
                "val_f1": (validation_f1),
                "val_predicted_class_0": int(np.sum(validation_predictions == 0)),
                "val_predicted_class_1": int(np.sum(validation_predictions == 1)),
            }
        )

    best_c = select_best_c_by_validation_f1(tuning_rows)

    classifier = SVC(
        kernel="precomputed",
        C=best_c,
        random_state=seed,
    )

    classifier.fit(
        K_train,
        y_train,
    )

    predictions = classifier.predict(K_test)

    scores = classifier.decision_function(K_test)

    metrics = compute_metrics(
        y_test,
        predictions,
        scores,
    )

    result: dict[str, object] = dict(metrics)

    predicted_class_0 = int(np.sum(predictions == 0))

    predicted_class_1 = int(np.sum(predictions == 1))

    result["best_c"] = best_c

    result["predicted_class_0"] = predicted_class_0

    result["predicted_class_1"] = predicted_class_1

    result["collapse"] = bool(predicted_class_1 == 0)

    result["zero_f1"] = bool(
        np.isclose(
            float(metrics["f1"]),
            0.0,
        )
    )

    return (
        result,
        tuning_rows,
    )


def combine_tuning_metadata(
    metadata: dict[str, object],
    tuning_row: dict[str, object],
) -> dict[str, object]:
    """Combine model metadata and one validation-tuning row."""
    combined = dict(metadata)

    combined.update(tuning_row)

    return combined


def compute_model_q_result(
    *,
    source: str,
    model: str,
    q: int,
    reps: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    c_grid: list[float],
    kernel_normalization: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """Compute one model and q result plus per-C validation rows."""
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
        X_validation,
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

    (
        training_kernel,
        validation_kernel,
        test_kernel,
    ) = qsvm_kernels(
        X_train,
        X_validation,
        X_test,
        reps=reps,
        kernel_normalization=(kernel_normalization),
    )

    (
        metrics,
        tuning_rows,
    ) = score_qsvm_best_c(
        K_train=training_kernel,
        K_val=validation_kernel,
        K_test=test_kernel,
        y_train=y_train,
        y_val=y_validation,
        y_test=y_test,
        c_grid=c_grid,
        seed=seed,
    )

    result: dict[str, object] = {
        "source": source,
        "synthetic_surrogate": (source != "real"),
        "model": model,
        "model_display": (MODEL_DISPLAY[model]),
        "q": q,
        "reps": reps,
        "seed": seed,
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
        "best_c": float(metrics["best_c"]),
        "accuracy": float(metrics["accuracy"]),
        "auc": float(metrics["auc"]),
        "f1": float(metrics["f1"]),
        "predicted_class_0": int(metrics["predicted_class_0"]),
        "predicted_class_1": int(metrics["predicted_class_1"]),
        "collapse": bool(metrics["collapse"]),
        "zero_f1": bool(metrics["zero_f1"]),
    }

    tuning_metadata = {
        "source": source,
        "synthetic_surrogate": (source != "real"),
        "model": model,
        "model_display": (MODEL_DISPLAY[model]),
        "q": q,
        "reps": reps,
        "seed": seed,
        "kernel_normalization": (kernel_normalization),
    }

    full_tuning_rows: list[dict[str, object]] = []

    for tuning_row in tuning_rows:
        full_tuning_rows.append(
            combine_tuning_metadata(
                tuning_metadata,
                tuning_row,
            )
        )

    return (
        result,
        full_tuning_rows,
    )


def combine_target_and_reference(
    target: dict[str, object],
    reference: dict[str, object],
    q_reference: int,
) -> dict[str, object]:
    """Combine one target-q result with its reference-q result."""
    combined = dict(target)

    combined["reference_q"] = q_reference

    combined["reference_best_c"] = float(reference["best_c"])

    combined["reference_f1"] = float(reference["f1"])

    combined["delta_f1_vs_reference"] = float(
        float(target["f1"]) - float(reference["f1"])
    )

    return combined


def compute_table9_rows(
    *,
    source: str,
    q_target: int,
    q_reference: int,
    reps: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    c_grid: list[float],
    kernel_normalization: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Compute target-q and reference-q results for all models."""
    summary_rows: list[dict[str, object]] = []

    tuning_rows: list[dict[str, object]] = []

    for model in TABLE9_MODELS:
        (
            target_result,
            target_tuning_rows,
        ) = compute_model_q_result(
            source=source,
            model=model,
            q=q_target,
            reps=reps,
            seed=seed,
            data_root=data_root,
            synthetic=synthetic,
            c_grid=c_grid,
            kernel_normalization=(kernel_normalization),
        )

        (
            reference_result,
            reference_tuning_rows,
        ) = compute_model_q_result(
            source=source,
            model=model,
            q=q_reference,
            reps=reps,
            seed=seed,
            data_root=data_root,
            synthetic=synthetic,
            c_grid=c_grid,
            kernel_normalization=(kernel_normalization),
        )

        tuning_rows.extend(target_tuning_rows)

        tuning_rows.extend(reference_tuning_rows)

        summary_rows.append(
            combine_target_and_reference(
                target_result,
                reference_result,
                q_reference,
            )
        )

        print(
            f"[table9] model={model} "
            f"target_q={q_target} "
            f"reference_q={q_reference} "
            f"seed={seed}",
            flush=True,
        )

    return (
        summary_rows,
        tuning_rows,
    )


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


def format_c(
    value: float,
) -> str:
    """Format one SVM regularization value."""
    return f"{value:g}"


def format_signed_metric(
    value: float,
) -> str:
    """Format a signed metric difference."""
    if abs(value) < 0.0005:
        value = 0.0

    return f"{value:+.3f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 9 artifact."""
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
        "# Synthetic surrogate Table 9 pipeline",
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
            "Table 9 selects QSVM C on validation minority-class F1 "
            "at the target q value and compares test F1 with the same "
            "selection procedure at the reference q value."
        ),
        "",
        (
            "The reference result is recomputed on the same source. "
            "It is not copied from the paper."
        ),
        "",
        (f"Target q: `{data['q_target']}`. Reference q: `{data['q_reference']}`."),
        "",
        (
            "| Model | Target q | Best C | Accuracy | AUC | F1 | "
            "Reference q | Reference C | Reference F1 | Delta F1 | "
            "Pred. class 1 | Collapse |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | --- |"
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
                "| {model} | {target_q} | {best_c} | {accuracy} | "
                "{auc} | {f1} | {reference_q} | {reference_c} | "
                "{reference_f1} | {delta} | "
                "{predicted_class_1} | {collapse} |"
            ).format(
                model=row["model_display"],
                target_q=row["q"],
                best_c=format_c(float(row["best_c"])),
                accuracy=format_metric(float(row["accuracy"])),
                auc=format_metric(float(row["auc"])),
                f1=format_metric(float(row["f1"])),
                reference_q=row["reference_q"],
                reference_c=format_c(float(row["reference_best_c"])),
                reference_f1=format_metric(float(row["reference_f1"])),
                delta=format_signed_metric(float(row["delta_f1_vs_reference"])),
                predicted_class_1=row["predicted_class_1"],
                collapse=row["collapse"],
            )
        )

    lines.extend(
        [
            "",
            (
                "C is selected exclusively from validation "
                "minority-class F1. Test labels are not used during "
                "hyperparameter selection."
            ),
            "",
            (
                "When validation F1 values are numerically equal, "
                "the smaller C value is selected."
            ),
            "",
            (
                "Trace normalization uses the training-kernel trace "
                "for the training, validation, and test kernels."
            ),
            "",
            (
                "Majority-only collapse means that no test sample is "
                "predicted as class 1."
            ),
            "",
            "Data and protocol metadata:",
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
        return "synthetic_surrogate_table9"

    if source == "synthetic_file":
        return "synthetic_file_table9"

    return "real_table9"


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
        "--q-target",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--q-reference",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--c-grid",
        default="0.01,0.1,1,10,100",
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
    c_grid: list[float],
) -> None:
    """Validate command-line arguments."""
    if args.q_target <= 0:
        raise ValueError("--q-target must be positive.")

    if args.q_reference <= 0:
        raise ValueError("--q-reference must be positive.")

    if args.reps < 1:
        raise ValueError("--reps must be at least 1.")

    if not c_grid:
        raise ValueError("At least one C value is required.")

    if any(not np.isfinite(c_value) for c_value in c_grid):
        raise ValueError("All C values must be finite.")

    if any(c_value <= 0.0 for c_value in c_grid):
        raise ValueError("All C values must be positive.")

    if len(c_grid) != len(set(c_grid)):
        raise ValueError("C values must be unique.")

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
    """Generate the Table 9 CSV, JSON, and Markdown artifacts."""
    args = parse_args()

    c_grid = parse_floats(args.c_grid)

    validate_args(
        args,
        c_grid,
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
        tuning_rows,
    ) = compute_table9_rows(
        source=args.source,
        q_target=args.q_target,
        q_reference=args.q_reference,
        reps=args.reps,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
        c_grid=c_grid,
        kernel_normalization=(args.kernel_normalization),
    )

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    tuning_path = args.results_dir / f"{prefix}_tuning.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    write_csv(
        summary_path,
        summary_rows,
    )

    write_csv(
        tuning_path,
        tuning_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 9",
        "paper_pointer": (PAPER_TABLE9_POINTER),
        "paths": {
            "summary_csv": str(summary_path),
            "tuning_csv": str(tuning_path),
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
            "q_target": (args.q_target),
            "q_reference": (args.q_reference),
            "reps": args.reps,
            "c_grid": c_grid,
            "kernel_normalization": (args.kernel_normalization),
            "normalization_protocol": (
                "The training-kernel normalization scale is applied "
                "to the training, validation, and test kernels."
            ),
            "selection_metric": ("validation_f1"),
            "selection_tie_break": ("smaller_C"),
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "models": list(TABLE9_MODELS),
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
        "tuning_rows": len(tuning_rows),
        "seed": args.seed,
        "q_target": (args.q_target),
        "q_reference": (args.q_reference),
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
        )
    )

    print(f"Wrote {summary_path}")

    print(f"Wrote {tuning_path}")

    print(f"Wrote {json_path}")

    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
