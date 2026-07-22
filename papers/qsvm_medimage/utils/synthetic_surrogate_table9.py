#!/usr/bin/env python3
"""Compute a Table 9-style q=16 QSVM C-tuning comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 9 protocol shape: q=16, DT9-style split, seed 0,
trace-normalized QSVM, and Best-C selected by validation F1.
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
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import feature_states
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import (
    SyntheticSpec,
    compute_metrics,
    load_dataset,
    parse_floats,
)

PAPER_TABLE9_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T9"

TABLE9_MODELS: tuple[str, ...] = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP-448",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-patch32-cls",
}


def normalize_kernel_like_original(
    kernel: np.ndarray, normalization: str
) -> np.ndarray:
    """Match the original qml-medimage QSVM normalization helpers."""
    if normalization == "none":
        return kernel.copy()

    if normalization == "trace":
        if kernel.shape[0] != kernel.shape[1]:
            return kernel.copy()
        trace = float(np.trace(kernel))
        if trace <= 0.0:
            return kernel.copy()
        return kernel / trace

    raise ValueError(f"unknown normalization: {normalization}")


def qsvm_kernels(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    reps: int,
    kernel_normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute train/validation/test kernels once for a C-grid search."""
    train_states = feature_states(X_train, reps=reps)
    val_states = feature_states(X_val, reps=reps)
    test_states = feature_states(X_test, reps=reps)

    K_train = np.abs(train_states.conj() @ train_states.T) ** 2
    np.fill_diagonal(K_train, 1.0)
    K_val = np.abs(val_states.conj() @ train_states.T) ** 2
    K_test = np.abs(test_states.conj() @ train_states.T) ** 2

    return (
        normalize_kernel_like_original(K_train, kernel_normalization),
        normalize_kernel_like_original(K_val, kernel_normalization),
        normalize_kernel_like_original(K_test, kernel_normalization),
    )


def select_best_c_by_validation_f1(tuning_rows: list[dict[str, object]]) -> float:
    """Select highest validation F1; use smaller C as deterministic tie-break."""
    if not tuning_rows:
        raise ValueError("empty C tuning rows")

    best_row = tuning_rows[0]
    for row in tuning_rows[1:]:
        row_f1 = float(row["val_f1"])
        best_f1 = float(best_row["val_f1"])
        if row_f1 > best_f1 or (
            row_f1 == best_f1 and float(row["C"]) < float(best_row["C"])
        ):
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
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Tune C on validation F1, then score the selected QSVM on test data."""
    tuning_rows = []
    for c in c_grid:
        svc = SVC(kernel="precomputed", C=c, random_state=seed)
        svc.fit(K_train, y_train)
        val_pred = svc.predict(K_val)
        tuning_rows.append(
            {
                "C": c,
                "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
            }
        )

    best_c = select_best_c_by_validation_f1(tuning_rows)
    svc = SVC(kernel="precomputed", C=best_c, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = svc.decision_function(K_test)
    metrics = compute_metrics(y_test, y_pred, scores)
    metrics["best_c"] = best_c
    return metrics, tuning_rows


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
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compute one model/q Best-C result plus per-C validation rows."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, X_val, X_test, explained_variance_ratio = preprocess(
        X[idx_train],
        X[idx_val],
        X[idx_test],
        q,
    )
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    K_train, K_val, K_test = qsvm_kernels(
        X_train,
        X_val,
        X_test,
        reps=reps,
        kernel_normalization=kernel_normalization,
    )
    metrics, tuning_rows = score_qsvm_best_c(
        K_train=K_train,
        K_val=K_val,
        K_test=K_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        c_grid=c_grid,
        seed=seed,
    )

    result = {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "model_display": MODEL_DISPLAY[model],
        "q": q,
        "reps": reps,
        "seed": seed,
        "kernel_normalization": kernel_normalization,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        "best_c": metrics["best_c"],
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "f1": metrics["f1"],
    }
    full_tuning_rows = [
        {
            "source": source,
            "synthetic_surrogate": source != "real",
            "model": model,
            "model_display": MODEL_DISPLAY[model],
            "q": q,
            "seed": seed,
            "kernel_normalization": kernel_normalization,
            **row,
        }
        for row in tuning_rows
    ]
    return result, full_tuning_rows


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
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute q=16 rows and q=8 reference rows for delta F1."""
    summary_rows = []
    tuning_rows = []
    for model in TABLE9_MODELS:
        target, target_tuning = compute_model_q_result(
            source=source,
            model=model,
            q=q_target,
            reps=reps,
            seed=seed,
            data_root=data_root,
            synthetic=synthetic,
            c_grid=c_grid,
            kernel_normalization=kernel_normalization,
        )
        reference, reference_tuning = compute_model_q_result(
            source=source,
            model=model,
            q=q_reference,
            reps=reps,
            seed=seed,
            data_root=data_root,
            synthetic=synthetic,
            c_grid=c_grid,
            kernel_normalization=kernel_normalization,
        )
        tuning_rows.extend(target_tuning)
        tuning_rows.extend(reference_tuning)
        summary_rows.append(
            {
                **target,
                "reference_q": q_reference,
                "reference_best_c": reference["best_c"],
                "reference_f1": reference["f1"],
                "delta_f1_vs_reference": float(target["f1"]) - float(reference["f1"]),
            }
        )
    return summary_rows, tuning_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def format_c(value: float) -> str:
    return f"{value:g}"


def format_signed_metric(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}"


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    data = payload["data"]
    lines = [
        "# Synthetic surrogate Table 9 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE9_POINTER}",
        "",
        "Table 9 tunes QSVM C on validation F1 at q=16 and compares test F1 with the same Best-C procedure at q=8.",
        "",
        f"Reference q for this surrogate artifact: `{data['q_reference']}` recalculated on the same source, not copied from the paper caption.",
        "",
        "| Model | Best-C | Acc | F1 | vs q8 ΔF1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {best_c} | {acc} | {f1} | {delta} |".format(
                model=row["model_display"],
                best_c=format_c(row["best_c"]),
                acc=format_metric(row["accuracy"]),
                f1=format_metric(row["f1"]),
                delta=format_signed_metric(row["delta_f1_vs_reference"]),
            )
        )

    lines.extend(
        [
            "",
            "Data source metadata:",
            "",
            "```json",
            json.dumps(payload["data"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def default_prefix(source: str) -> str:
    if source == "synthetic":
        return "synthetic_surrogate_table9"
    if source == "synthetic_file":
        return "synthetic_file_table9"
    return "real_table9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q-target", type=int, default=16)
    parser.add_argument("--q-reference", type=int, default=8)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--c-grid", default="0.01,0.1,1,10,100")
    parser.add_argument(
        "--kernel-normalization", choices=("trace", "none"), default="trace"
    )
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c_grid = parse_floats(args.c_grid)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    summary_rows, tuning_rows = compute_table9_rows(
        source=args.source,
        q_target=args.q_target,
        q_reference=args.q_reference,
        reps=args.reps,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
        c_grid=c_grid,
        kernel_normalization=args.kernel_normalization,
    )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    tuning_path = args.results_dir / f"{prefix}_tuning.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(summary_path, summary_rows)
    write_csv(tuning_path, tuning_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 9",
        "paper_pointer": PAPER_TABLE9_POINTER,
        "paths": {
            "summary_csv": str(summary_path),
            "tuning_csv": str(tuning_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "seed": args.seed,
            "q_target": args.q_target,
            "q_reference": args.q_reference,
            "reps": args.reps,
            "c_grid": c_grid,
            "kernel_normalization": args.kernel_normalization,
            "selection_metric": "validation_f1",
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "models": list(TABLE9_MODELS),
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(
        json.dumps(
            {"rows": len(summary_rows), "tuning_rows": len(tuning_rows)}, indent=2
        )
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {tuning_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
