#!/usr/bin/env python3
"""Generate a Figure 5-style partial qubit sweep for QSVM.

This is not a reproduction of the paper figure when ``--source synthetic`` is
used. It mirrors the Figure 5 protocol shape: q in {2, 3, 4, 5, 6, 8}, C=1,
DT9-style split/preprocessing, trace-normalized QSVM, and seed 0.
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
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import (
    SyntheticSpec,
    compute_metrics,
    load_dataset,
    parse_ints,
)

PAPER_FIGURE5_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F5"

FIGURE5_MODELS: tuple[str, ...] = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP-448",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}

MODEL_STYLE = {
    "medsiglip-448": {"color": "#2166ac", "marker": "o"},
    "rad-dino": {"color": "#e66101", "marker": "s"},
    "vit-patch32-cls": {"color": "#4dac26", "marker": "^"},
}


def normalize_kernel_like_original(kernel: np.ndarray, normalization: str) -> np.ndarray:
    """Match the original qml-medimage QSVM normalization helpers.

    The original ``trace`` and ``cosine`` helpers normalize square train kernels
    and leave rectangular test kernels unchanged. This behavior is unusual but
    it is part of the paper's runnable setup, so the sweep keeps it explicit.
    """
    if normalization == "none":
        return kernel.copy()

    if normalization == "trace":
        if kernel.shape[0] != kernel.shape[1]:
            return kernel.copy()
        trace = float(np.trace(kernel))
        if trace <= 0.0:
            return kernel.copy()
        return kernel / trace

    if normalization == "frobenius":
        norm = float(np.linalg.norm(kernel, ord="fro"))
        if norm <= 0.0:
            return kernel.copy()
        return kernel / norm

    if normalization == "cosine":
        if kernel.shape[0] != kernel.shape[1]:
            return kernel.copy()
        diag = np.diag(kernel).reshape(-1, 1)
        scale = np.sqrt(diag @ diag.T)
        scale[scale == 0.0] = 1.0
        return kernel / scale

    raise ValueError(f"unknown normalization: {normalization}")


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
) -> dict[str, float]:
    """Train and score one fixed-C QSVM point in the qubit sweep."""
    K_train_raw = fidelity_kernel(X_train, reps=reps)
    K_test_raw = fidelity_kernel(X_test, X_train, reps=reps)
    K_train = normalize_kernel_like_original(K_train_raw, kernel_normalization)
    K_test = normalize_kernel_like_original(K_test_raw, kernel_normalization)

    svc = SVC(kernel="precomputed", C=c, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = svc.decision_function(K_test)
    metrics = compute_metrics(y_test, y_pred, scores)
    metrics["q"] = float(q)
    return metrics


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
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    y_train = y[idx_train]
    y_test = y[idx_test]

    rows = []
    for q in q_values:
        X_train, _, X_test, explained_variance_ratio = preprocess(
            X[idx_train],
            X[idx_val],
            X[idx_test],
            q,
        )
        metrics = score_qsvm_for_q(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            q=q,
            reps=reps,
            c=c,
            seed=seed,
            kernel_normalization=kernel_normalization,
        )
        rows.append(
            {
                "source": source,
                "synthetic_surrogate": source != "real",
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "q": q,
                "reps": reps,
                "C": c,
                "seed": seed,
                "kernel_normalization": kernel_normalization,
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "test_class_0": int(np.sum(y_test == 0)),
                "test_class_1": int(np.sum(y_test == 1)),
                "pca_variance_percent": 100.0 * float(explained_variance_ratio),
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "f1": metrics["f1"],
            }
        )
    return rows


def metric_series(
    rows: list[dict[str, object]],
    *,
    model: str,
    metric: str,
) -> tuple[list[int], list[float]]:
    """Return one model's metric values sorted by q."""
    selected = sorted(
        [row for row in rows if row["model"] == model],
        key=lambda row: int(row["q"]),
    )
    return [int(row["q"]) for row in selected], [float(row[metric]) for row in selected]


def padded_ylim(values: list[float], *, lower_floor: float = 0.0) -> tuple[float, float]:
    """Give line plots a small vertical margin without hiding zero F1."""
    low = min(values)
    high = max(values)
    if np.isclose(low, high):
        pad = 0.05
    else:
        pad = 0.08 * (high - low)
    return max(lower_floor, low - pad), min(1.0, high + pad)


def save_plot(path: Path, *, rows: list[dict[str, object]], q_values: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for model in FIGURE5_MODELS:
        q_axis, accuracy = metric_series(rows, model=model, metric="accuracy")
        style = MODEL_STYLE[model]
        axes[0].plot(
            q_axis,
            accuracy,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.8,
            markersize=5.5,
            label=MODEL_DISPLAY[model],
        )

    for model in FIGURE5_MODELS:
        q_axis, f1 = metric_series(rows, model=model, metric="f1")
        style = MODEL_STYLE[model]
        axes[1].plot(
            q_axis,
            f1,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.8,
            markersize=5.5,
            label=MODEL_DISPLAY[model],
        )

    accuracy_values = [float(row["accuracy"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]

    axes[0].set_xlabel("Number of Qubits (q)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_xticks(q_values)
    axes[0].set_ylim(*padded_ylim(accuracy_values, lower_floor=0.0))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Number of Qubits (q)")
    axes[1].set_ylabel("Minority-class F1")
    axes[1].set_xticks(q_values)
    axes[1].set_ylim(*padded_ylim(f1_values, lower_floor=0.0))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    image_name = Path(payload["paths"]["png"]).name
    lines = [
        "# Synthetic surrogate Figure 5 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper figure because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_FIGURE5_POINTER}",
        "",
        "Figure 5 is a partial qubit sweep for QSVM: test accuracy and minority-class F1 versus q for the three embedding models.",
        "",
        f"![Synthetic surrogate Figure 5]({image_name})",
        "",
        "| Model | q | Acc | AUC | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {q} | {acc:.3f} | {auc:.3f} | {f1:.3f} |".format(
                model=row["model_display"],
                q=row["q"],
                acc=row["accuracy"],
                auc=row["auc"],
                f1=row["f1"],
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
        return "synthetic_surrogate_figure5"
    if source == "synthetic_file":
        return "synthetic_file_figure5"
    return "real_figure5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--q-values", default="2,3,4,5,6,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--kernel-normalization",
        choices=("trace", "none", "cosine", "frobenius"),
        default="trace",
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
    q_values = parse_ints(args.q_values)
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
        summary_rows.extend(
            compute_model_sweep_rows(
                source=args.source,
                model=model,
                q_values=q_values,
                reps=args.reps,
                c=args.C,
                seed=args.seed,
                data_root=args.data_root,
                synthetic=synthetic,
                kernel_normalization=args.kernel_normalization,
            )
        )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.results_dir / f"{prefix}.png"
    csv_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    write_csv(csv_path, summary_rows)
    save_plot(png_path, rows=summary_rows, q_values=q_values)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 5",
        "paper_pointer": PAPER_FIGURE5_POINTER,
        "paths": {
            "png": str(png_path),
            "summary_csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "seed": args.seed,
            "q_values": q_values,
            "reps": args.reps,
            "C": args.C,
            "kernel_normalization": args.kernel_normalization,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "models": list(FIGURE5_MODELS),
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps({"rows": len(summary_rows), "seed": args.seed}, indent=2))
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
