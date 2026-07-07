#!/usr/bin/env python3
"""Compute a Table 7-style QSVM kernel-normalization comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 7 protocol shape: q=8, reps=1, C=1, seed=0, and the
same four kernel normalizations.
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
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import SyntheticSpec, compute_metrics, load_dataset

PAPER_TABLE7_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T7"

TABLE7_MODELS: tuple[str, ...] = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

TABLE7_NORMALIZATIONS: tuple[str, ...] = (
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


def safe_scale(value: float) -> float:
    """Avoid division by zero while keeping the normal case unchanged."""
    if value <= 0.0:
        return 1.0
    return float(value)


def normalize_train_test_kernels(
    K_train: np.ndarray,
    K_test: np.ndarray,
    normalization: str,
    *,
    K_test_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one Table 7 normalization using train-derived scale factors.

    The paper table compares fixed-C QSVM performance under different kernel
    normalizations. For the rectangular test kernel, the same train-derived
    scale is used so train and test values stay on the same scale.
    """
    if normalization == "none":
        return K_train.copy(), K_test.copy()

    if normalization == "trace":
        scale = safe_scale(float(np.trace(K_train)))
        return K_train / scale, K_test / scale

    if normalization == "frobenius":
        scale = safe_scale(float(np.linalg.norm(K_train, ord="fro")))
        return K_train / scale, K_test / scale

    if normalization == "cosine":
        train_diag = np.sqrt(np.maximum(np.diag(K_train), 1e-12))
        if K_test_diag is None:
            K_test_diag = np.ones(K_test.shape[0])
        test_diag = np.sqrt(np.maximum(K_test_diag, 1e-12))
        return (
            K_train / np.outer(train_diag, train_diag),
            K_test / np.outer(test_diag, train_diag),
        )

    raise ValueError(f"unknown normalization: {normalization}")


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
) -> dict[str, float]:
    """Train the fixed-C precomputed-kernel QSVM for one normalization."""
    K_train, K_test = normalize_train_test_kernels(
        K_train_raw,
        K_test_raw,
        normalization,
        K_test_diag=K_test_diag,
    )
    svc = SVC(kernel="precomputed", C=c, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = svc.decision_function(K_test)
    return compute_metrics(y_test, y_pred, scores)


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
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, X_test, explained_variance_ratio = preprocess(
        X[idx_train],
        X[idx_val],
        X[idx_test],
        q,
    )
    y_train = y[idx_train]
    y_test = y[idx_test]

    K_train_raw = fidelity_kernel(X_train, reps=reps)
    K_test_raw = fidelity_kernel(X_test, X_train, reps=reps)
    K_test_diag = np.ones(len(y_test))

    rows = []
    for normalization in TABLE7_NORMALIZATIONS:
        metrics = score_qsvm_normalization(
            K_train_raw=K_train_raw,
            K_test_raw=K_test_raw,
            K_test_diag=K_test_diag,
            y_train=y_train,
            y_test=y_test,
            normalization=normalization,
            c=c,
            seed=seed,
        )
        rows.append(
            {
                "source": source,
                "synthetic_surrogate": source != "real",
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "normalization": normalization,
                "q": q,
                "reps": reps,
                "C": c,
                "seed": seed,
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


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "# Synthetic surrogate Table 7 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE7_POINTER}",
        "",
        "Table 7 compares QSVM test metrics for q=8, reps=1, C=1, seed=0 under four kernel normalizations.",
        "",
        "| Model | Norm. | Acc | AUC | F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {norm} | {acc} | {auc} | {f1} |".format(
                model=row["model_display"],
                norm=row["normalization"],
                acc=format_metric(row["accuracy"]),
                auc=format_metric(row["auc"]),
                f1=format_metric(row["f1"]),
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
        return "synthetic_surrogate_table7"
    if source == "synthetic_file":
        return "synthetic_file_table7"
    return "real_table7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        summary_rows.extend(
            compute_table7_rows(
                source=args.source,
                model=model,
                q=args.q,
                reps=args.reps,
                c=args.C,
                seed=args.seed,
                data_root=args.data_root,
                synthetic=synthetic,
            )
        )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(summary_path, summary_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 7",
        "paper_pointer": PAPER_TABLE7_POINTER,
        "paths": {
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "seed": args.seed,
            "q": args.q,
            "reps": args.reps,
            "C": args.C,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "models": list(TABLE7_MODELS),
            "normalizations": list(TABLE7_NORMALIZATIONS),
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps({"rows": len(summary_rows), "seed": args.seed}, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
