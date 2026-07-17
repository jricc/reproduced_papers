#!/usr/bin/env python3
"""Compute a Table 3-style QSVM confusion matrix on surrogate or real data.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 3 protocol shape: one representative MedSigLIP-448
QSVM C=1 test confusion matrix at q=11, seed 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset

PAPER_TABLE3_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T3"


def predict_qsvm_c1(
    *,
    X_train,
    y_train,
    X_test,
    seed: int,
):
    """Predict with the Table 3 QSVM: fidelity kernel, fixed C=1."""
    K_train = fidelity_kernel(X_train)
    K_test = fidelity_kernel(X_test, X_train)
    svc = SVC(kernel="precomputed", C=1.0, random_state=seed)
    svc.fit(K_train, y_train)
    return svc.predict(K_test)


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
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, X_test, explained_variance_ratio = preprocess(
        X[idx_train], X[idx_val], X[idx_test], q
    )
    y_train = y[idx_train]
    y_test = y[idx_test]
    y_pred = predict_qsvm_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        seed=seed,
    )

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "val_samples": int(len(idx_val)),
        "test_samples": int(len(idx_test)),
        "explained_variance_ratio": float(explained_variance_ratio),
        "confusion_matrix": matrix.astype(int).tolist(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def confusion_rows(table3: dict[str, object]) -> list[dict[str, object]]:
    matrix = table3["confusion_matrix"]
    labels = ["class_0_majority", "class_1_minority"]
    return [
        {"true_label": labels[0], "pred_class_0_majority": matrix[0][0], "pred_class_1_minority": matrix[0][1]},
        {"true_label": labels[1], "pred_class_0_majority": matrix[1][0], "pred_class_1_minority": matrix[1][1]},
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    table3 = payload["table3"]
    rows = confusion_rows(table3)
    lines = [
        "# Synthetic surrogate Table 3 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE3_POINTER}",
        "",
        "The paper Table 3 reports a representative MedSigLIP-448 QSVM confusion matrix at q=11, seed 0. In the original repo, q=11 is the MedSigLIP row whose F1=0.586 matches the paper Table 3 text.",
        "",
        "| True label | Pred class 0 majority | Pred class 1 minority |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['true_label']} | {row['pred_class_0_majority']} | {row['pred_class_1_minority']} |"
        )
    lines.extend(
        [
            "",
            f"Accuracy: {table3['accuracy']:.6f}",
            f"Precision: {table3['precision']:.6f}",
            f"Recall: {table3['recall']:.6f}",
            f"F1: {table3['f1']:.6f}",
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
        return "synthetic_surrogate_table3"
    if source == "synthetic_file":
        return "synthetic_file_table3"
    return "real_table3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model", default="medsiglip-448")
    parser.add_argument("--q", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
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
    table3 = compute_table3(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
    )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"{prefix}_confusion_matrix.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 3",
        "paper_pointer": PAPER_TABLE3_POINTER,
        "paths": {
            "confusion_matrix_csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
        },
        "table3": table3,
    }

    write_csv(csv_path, confusion_rows(table3))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps(table3, indent=2, sort_keys=True))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
