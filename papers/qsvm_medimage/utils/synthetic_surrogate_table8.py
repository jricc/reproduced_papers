#!/usr/bin/env python3
"""Compute a Table 8-style 1-DOF vs 3-DOF QSVM comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 8 protocol shape: q=8, reps=1, trace normalization,
C=1, DT9-style split/preprocessing, and seed 0.
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

from lib.quantum_kernel import _H, _apply_1q, _apply_cx, _ry, _rz, fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import SyntheticSpec, compute_metrics, load_dataset

PAPER_TABLE8_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T8"

TABLE8_MODELS: tuple[str, ...] = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

TABLE8_CIRCUITS: tuple[str, ...] = ("1-DOF", "3-DOF")

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def pca_dim_for_circuit(q: int, circuit: str) -> int:
    """Return the PCA dimension used by the original 1-DOF/3-DOF switch."""
    if circuit == "1-DOF":
        return q
    if circuit == "3-DOF":
        return 3 * q
    raise ValueError(f"unknown circuit: {circuit}")


def bsp_3dof_statevector(x: np.ndarray, *, q: int, reps: int = 1) -> np.ndarray:
    """Return the original make_bsp_3dof statevector for one 3*q vector."""
    if len(x) != 3 * q:
        raise ValueError(f"3-DOF expects {3 * q} features for q={q}, got {len(x)}")

    state = np.zeros(2**q, dtype=np.complex128)
    state[0] = 1.0
    for _ in range(reps):
        for qubit in range(q):
            state = _apply_1q(state, _H, qubit, q)
        for qubit in range(q):
            state = _apply_1q(state, _rz(float(x[qubit])), qubit, q)
            state = _apply_1q(state, _ry(float(x[qubit + q])), qubit, q)
        for qubit in range(q - 1):
            state = _apply_cx(state, qubit, qubit + 1, q)
        for qubit in range(q):
            state = _apply_1q(state, _rz(float(x[qubit + 2 * q])), qubit, q)
    return state


def fidelity_kernel_3dof(
    data1: np.ndarray,
    *,
    q: int,
    reps: int,
    data2: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the 3-DOF fidelity kernel used by Table 8."""
    states1 = np.stack([bsp_3dof_statevector(row, q=q, reps=reps) for row in data1])
    if data2 is None:
        gram = states1.conj() @ states1.T
        kernel = np.abs(gram) ** 2
        np.fill_diagonal(kernel, 1.0)
        return kernel

    states2 = np.stack([bsp_3dof_statevector(row, q=q, reps=reps) for row in data2])
    return np.abs(states1.conj() @ states2.T) ** 2


def normalize_kernel_like_original(kernel: np.ndarray, normalization: str) -> np.ndarray:
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


def score_qsvm_circuit(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    q: int,
    circuit: str,
    reps: int,
    c: float,
    seed: int,
    kernel_normalization: str,
) -> dict[str, float]:
    """Train the fixed-C QSVM for one Table 8 circuit."""
    if circuit == "1-DOF":
        K_train_raw = fidelity_kernel(X_train, reps=reps)
        K_test_raw = fidelity_kernel(X_test, X_train, reps=reps)
    elif circuit == "3-DOF":
        K_train_raw = fidelity_kernel_3dof(X_train, q=q, reps=reps)
        K_test_raw = fidelity_kernel_3dof(X_test, q=q, reps=reps, data2=X_train)
    else:
        raise ValueError(f"unknown circuit: {circuit}")

    K_train = normalize_kernel_like_original(K_train_raw, kernel_normalization)
    K_test = normalize_kernel_like_original(K_test_raw, kernel_normalization)

    svc = SVC(kernel="precomputed", C=c, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = svc.decision_function(K_test)
    return compute_metrics(y_test, y_pred, scores)


def compute_table8_rows(
    *,
    source: str,
    model: str,
    q: int,
    reps: int,
    c: float,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    kernel_normalization: str,
) -> list[dict[str, object]]:
    """Compute the 1-DOF and 3-DOF Table 8 rows for one model."""
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
    for circuit in TABLE8_CIRCUITS:
        pca_dim = pca_dim_for_circuit(q, circuit)
        X_train, _, X_test, explained_variance_ratio = preprocess(
            X[idx_train],
            X[idx_val],
            X[idx_test],
            pca_dim,
        )
        metrics = score_qsvm_circuit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            q=q,
            circuit=circuit,
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
                "circuit": circuit,
                "q": q,
                "pca_dim": pca_dim,
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
        "# Synthetic surrogate Table 8 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE8_POINTER}",
        "",
        "Table 8 compares the 1-DOF BSP circuit against the 3-DOF variant at q=8, reps=1, trace normalization, C=1, seed=0.",
        "",
        "| Model | Circuit | Acc | AUC | F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {circuit} | {acc} | {auc} | {f1} |".format(
                model=row["model_display"],
                circuit=row["circuit"],
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
        return "synthetic_surrogate_table8"
    if source == "synthetic_file":
        return "synthetic_file_table8"
    return "real_table8"


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
    parser.add_argument("--kernel-normalization", choices=("trace", "none"), default="trace")
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
    for model in TABLE8_MODELS:
        summary_rows.extend(
            compute_table8_rows(
                source=args.source,
                model=model,
                q=args.q,
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
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(summary_path, summary_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 8",
        "paper_pointer": PAPER_TABLE8_POINTER,
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
            "kernel_normalization": args.kernel_normalization,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "models": list(TABLE8_MODELS),
            "circuits": list(TABLE8_CIRCUITS),
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
