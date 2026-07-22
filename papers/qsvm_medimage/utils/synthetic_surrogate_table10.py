#!/usr/bin/env python3
"""Compute a Table 10-style rank-matched RBF vs QSVM comparison.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 10 protocol shape: MedSigLIP-448, q in {4, 6, 11,
16}, 10 seeds, C=1, and collapse defined as F1 < 0.05.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import effective_rank, feature_states, fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset, parse_ints

PAPER_TABLE10_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T10"

COLLAPSE_THRESHOLD = 0.05
DEFAULT_Q_VALUES = "4,6,11,16"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"

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


def rbf_effective_rank(X_train: np.ndarray, gamma: float) -> float:
    return effective_rank(rbf_kernel(X_train, gamma=gamma))


def find_rank_matched_gamma(
    X_train: np.ndarray,
    target_rank: float,
    *,
    tol: float = 0.05,
    max_iter: int = 50,
) -> float:
    """Binary-search gamma so RBF effective rank matches the quantum target."""
    gamma_low = 1e-6
    gamma_high = 1e3
    rank_low = rbf_effective_rank(X_train, gamma_low)
    rank_high = rbf_effective_rank(X_train, gamma_high)

    if target_rank <= rank_low:
        return gamma_low
    if target_rank >= rank_high:
        return gamma_high

    for _ in range(max_iter):
        gamma_mid = float(np.sqrt(gamma_low * gamma_high))
        rank_mid = rbf_effective_rank(X_train, gamma_mid)
        if abs(rank_mid - target_rank) / target_rank < tol:
            return gamma_mid
        if rank_mid < target_rank:
            gamma_low = gamma_mid
        else:
            gamma_high = gamma_mid

    return float(np.sqrt(gamma_low * gamma_high))


def compute_seed0_quantum_rank(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> float:
    """Compute the fixed rank target used for all seeds of one q."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, _, _ = preprocess(X[idx_train], X[idx_val], X[idx_test], q)
    return effective_rank(fidelity_kernel(X_train))


def compute_qsvm_metrics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    seed: int,
    kernel_normalization: str,
) -> dict[str, float]:
    """Compute the QSVM C=1 row for one seed."""
    train_states = feature_states(X_train)
    test_states = feature_states(X_test)
    K_train = np.abs(train_states.conj() @ train_states.T) ** 2
    np.fill_diagonal(K_train, 1.0)
    K_test = np.abs(test_states.conj() @ train_states.T) ** 2

    K_train = normalize_kernel_like_original(K_train, kernel_normalization)
    K_test = normalize_kernel_like_original(K_test, kernel_normalization)

    svc = SVC(kernel="precomputed", C=c, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def compute_rbf_metrics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    gamma: str | float,
    seed: int,
) -> dict[str, float]:
    """Compute one RBF SVC row for one seed."""
    svc = SVC(kernel="rbf", C=c, gamma=gamma, random_state=seed)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


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
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, X_test, explained_variance_ratio = preprocess(
        X[idx_train],
        X[idx_val],
        X[idx_test],
        q,
    )
    y_train = y[idx_train]
    y_test = y[idx_test]

    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    rbf_scale = compute_rbf_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        c=c,
        gamma="scale",
        seed=seed,
    )

    gamma_star = find_rank_matched_gamma(X_train, target_effective_rank)
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
        kernel_normalization=kernel_normalization,
    )

    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "model_display": MODEL_DISPLAY.get(model, model),
        "q": q,
        "seed": seed,
        "C": c,
        "kernel_normalization": kernel_normalization,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_class_0": int(np.sum(y_test == 0)),
        "test_class_1": int(np.sum(y_test == 1)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        "target_effective_rank": target_effective_rank,
        "gamma_scale": float(gamma_scale),
        "effective_rank_rbf_scale": rbf_effective_rank(X_train, gamma_scale),
        "f1_rbf_scale": rbf_scale["f1"],
        "accuracy_rbf_scale": rbf_scale["accuracy"],
        "collapsed_rbf_scale": bool(rbf_scale["f1"] < collapse_threshold),
        "gamma_star": gamma_star,
        "effective_rank_rbf_star": rbf_effective_rank(X_train, gamma_star),
        "f1_rbf_star": rbf_star["f1"],
        "accuracy_rbf_star": rbf_star["accuracy"],
        "collapsed_rbf_star": bool(rbf_star["f1"] < collapse_threshold),
        "f1_qsvm": qsvm["f1"],
        "accuracy_qsvm": qsvm["accuracy"],
        "collapsed_qsvm": bool(qsvm["f1"] < collapse_threshold),
    }


def mean_bool(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def mean_float(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def summarize_table10(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate seed rows into the Table 10 column layout."""
    summary_rows = []
    q_values = sorted({int(row["q"]) for row in long_rows})
    for q in q_values:
        rows = [row for row in long_rows if int(row["q"]) == q]
        summary_rows.append(
            {
                "q": q,
                "target_effective_rank": float(rows[0]["target_effective_rank"]),
                "collapse_rate_rbf_scale": mean_bool(rows, "collapsed_rbf_scale"),
                "collapse_rate_rbf_star": mean_bool(rows, "collapsed_rbf_star"),
                "collapse_rate_qsvm": mean_bool(rows, "collapsed_qsvm"),
                "f1_mean_rbf_scale": mean_float(rows, "f1_rbf_scale"),
                "f1_mean_rbf_star": mean_float(rows, "f1_rbf_star"),
                "f1_mean_qsvm": mean_float(rows, "f1_qsvm"),
                "n_seeds": len(rows),
            }
        )
    return summary_rows


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


def format_rank(value: float) -> str:
    return f"{value:.2f}"


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    data = payload["data"]
    lines = [
        "# Synthetic surrogate Table 10 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE10_POINTER}",
        "",
        "Table 10 compares default RBF, rank-matched RBF, and QSVM across seeds. Collapse means F1 below the configured threshold.",
        "",
        f"Model: `{data['model']}`. Collapse threshold: `{data['collapse_threshold']}`. All methods use C=`{data['C']}`.",
        "",
        "| q | eff_rank(K_Q) | Collapse (RBF_scale) | Collapse (RBF*) | Collapse (QSVM) | F1 (RBF_scale) | F1 (RBF*) | F1 (QSVM) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {q} | {rank} | {collapse_scale} | {collapse_star} | {collapse_qsvm} | {f1_scale} | {f1_star} | {f1_qsvm} |".format(
                q=row["q"],
                rank=format_rank(row["target_effective_rank"]),
                collapse_scale=format_metric(row["collapse_rate_rbf_scale"]),
                collapse_star=format_metric(row["collapse_rate_rbf_star"]),
                collapse_qsvm=format_metric(row["collapse_rate_qsvm"]),
                f1_scale=format_metric(row["f1_mean_rbf_scale"]),
                f1_star=format_metric(row["f1_mean_rbf_star"]),
                f1_qsvm=format_metric(row["f1_mean_qsvm"]),
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
        return "synthetic_surrogate_table10"
    if source == "synthetic_file":
        return "synthetic_file_table10"
    return "real_table10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model", default="medsiglip-448")
    parser.add_argument("--q-values", default=DEFAULT_Q_VALUES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--rank-seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--collapse-threshold", type=float, default=COLLAPSE_THRESHOLD)
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
    q_values = parse_ints(args.q_values)
    seeds = parse_ints(args.seeds)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    rank_targets = {
        q: compute_seed0_quantum_rank(
            source=args.source,
            model=args.model,
            q=q,
            seed=args.rank_seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )
        for q in q_values
    }

    long_rows = []
    for q in q_values:
        for seed in seeds:
            long_rows.append(
                compute_seed_row(
                    source=args.source,
                    model=args.model,
                    q=q,
                    seed=seed,
                    target_effective_rank=rank_targets[q],
                    data_root=args.data_root,
                    synthetic=synthetic,
                    c=args.C,
                    collapse_threshold=args.collapse_threshold,
                    kernel_normalization=args.kernel_normalization,
                )
            )

    summary_rows = summarize_table10(long_rows)

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.results_dir / f"{prefix}_long.csv"
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(long_path, long_rows)
    write_csv(summary_path, summary_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 10",
        "paper_pointer": PAPER_TABLE10_POINTER,
        "paths": {
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "model": args.model,
            "q_values": q_values,
            "seeds": seeds,
            "rank_seed": args.rank_seed,
            "C": args.C,
            "collapse_threshold": args.collapse_threshold,
            "kernel_normalization": args.kernel_normalization,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "rank_targets": {str(q): rank_targets[q] for q in q_values},
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(
        json.dumps({"rows": len(summary_rows), "seed_rows": len(long_rows)}, indent=2)
    )
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
