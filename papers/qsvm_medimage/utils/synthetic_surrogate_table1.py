#!/usr/bin/env python3
"""Compute a Table 1-style summary on a clearly labelled surrogate dataset.

This is not a reproduction of the paper numbers. It exercises the same
comparison shape while the gated MIMIC-CXR embeddings are unavailable:

- Tier 1: QSVM C=1 vs linear SVM C=1 on PCA-q features.
- Tier 2: QSVM C=1 vs best-C RBF SVM on PCA-q features.

When the real embeddings are available, the intended change is only the data
source: pass ``--source real --data-root ...`` and keep the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

# Allow this script to run from utils/ while importing the local paper package
# and the shared runtime_lib package one level above papers/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.data import load_real_embeddings, make_synthetic_embeddings
from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices

TIER1_CONFIGS: tuple[tuple[str, int], ...] = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("medsiglip-448", 8),
    ("medsiglip-448", 9),
    ("medsiglip-448", 10),
    ("medsiglip-448", 11),
    ("medsiglip-448", 12),
    ("medsiglip-448", 16),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("rad-dino", 8),
    ("rad-dino", 10),
    ("rad-dino", 16),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
    ("vit-patch32-cls", 8),
    ("vit-patch32-cls", 10),
    ("vit-patch32-cls", 16),
)

TIER2_CONFIGS: tuple[tuple[str, int], ...] = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("medsiglip-448", 8),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
)

MODEL_SEED_OFFSETS = {
    # The synthetic source has no real model-specific embeddings. These offsets
    # keep the paper's model labels while avoiding identical surrogate datasets.
    "medsiglip-448": 0,
    "rad-dino": 10_000,
    "vit-patch32-cls": 20_000,
}


@dataclass(frozen=True)
class SyntheticSpec:
    n_samples: int
    ambient_dim: int
    latent_dim: int
    minority_frac: float
    signal: float
    noise: float


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def table_configs() -> list[tuple[str, int]]:
    configs = list(TIER1_CONFIGS)
    for config in TIER2_CONFIGS:
        if config not in configs:
            configs.append(config)
    return configs


def load_dataset(
    *,
    source: str,
    model: str,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the only swappable part of the pipeline: real or surrogate X, y."""
    if source == "real":
        if data_root is None:
            raise ValueError("--data-root is required with --source real")
        return load_real_embeddings(model=model, seed=seed, data_root=data_root)

    synthetic_seed = seed + MODEL_SEED_OFFSETS[model]
    return make_synthetic_embeddings(
        n_samples=synthetic.n_samples,
        ambient_dim=synthetic.ambient_dim,
        latent_dim=synthetic.latent_dim,
        minority_fraction=synthetic.minority_frac,
        signal=synthetic.signal,
        noise=synthetic.noise,
        seed=synthetic_seed,
    )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, scores),
    }


def decision_scores(model: SVC, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def score_linear_svc_c1(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Score the Tier 1 classical baseline: linear SVC with fixed C=1."""
    svc = SVC(kernel="linear", C=1.0, random_state=seed)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    scores = decision_scores(svc, X_test)
    row: dict[str, object] = {"method": "linear", "C": 1.0}
    row.update(compute_metrics(y_test, y_pred, scores))
    return row


def select_rbf_c_by_validation_f1(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> float:
    """Select the Tier 2 RBF C using validation F1."""
    best_c = None
    best_f1 = -1.0
    # Table 1 Tier 2 compares QSVM C=1 against an
    # RBF SVM whose C is selected on validation F1 before test scoring.
    for C in c_grid:
        svc = SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)
        svc.fit(X_train, y_train)
        val_pred = svc.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, zero_division=0))
        if val_f1 > best_f1 or (val_f1 == best_f1 and (best_c is None or C < best_c)):
            best_f1 = val_f1
            best_c = C
    if best_c is None:
        raise RuntimeError("empty C grid")
    return float(best_c)


def score_rbf_svc_best_c(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> dict[str, object]:
    """Score the Tier 2 classical baseline after validation C selection."""
    best_c = select_rbf_c_by_validation_f1(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        c_grid=c_grid,
        seed=seed,
    )
    svc = SVC(kernel="rbf", C=best_c, gamma="scale", random_state=seed)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    scores = decision_scores(svc, X_test)
    row: dict[str, object] = {"method": "rbf", "C": best_c}
    row.update(compute_metrics(y_test, y_pred, scores))
    return row


def score_qsvm_c1(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Score the paper-style QSVM: fidelity kernel with fixed C=1."""
    K_train = fidelity_kernel(X_train)
    K_test = fidelity_kernel(X_test, X_train)
    svc = SVC(kernel="precomputed", C=1.0, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = decision_scores(svc, K_test)
    row: dict[str, object] = {"method": "qsvm", "C": 1.0}
    row.update(compute_metrics(y_test, y_pred, scores))
    return row


def run_one_config(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    c_grid: list[float],
) -> list[dict[str, object]]:
    """Run the three Table 1 comparators for one model/q/seed."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    # Keep split and preprocessing shared across synthetic and real sources so
    # switching to gated embeddings changes only load_dataset().
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, X_val, X_test, evr = preprocess(X[idx_train], X[idx_val], X[idx_test], q)
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    run_metadata = {
        "source": source,
        "synthetic_surrogate": source == "synthetic",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "val_samples": int(len(idx_val)),
        "test_samples": int(len(idx_test)),
        "explained_variance_ratio": float(evr),
    }

    score_rows = []
    qsvm_row = score_qsvm_c1(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed
    )
    score_rows.append({**run_metadata, **qsvm_row, "selected_on": "fixed"})

    linear_row = score_linear_svc_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )
    score_rows.append({**run_metadata, **linear_row, "selected_on": "fixed"})

    rbf_row = score_rbf_svc_best_c(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        c_grid=c_grid,
        seed=seed,
    )
    score_rows.append({**run_metadata, **rbf_row, "selected_on": "validation_f1"})
    return score_rows


def mean_metric(rows: list[dict[str, object]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method]
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize(
    long_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Aggregate seed-level rows into Table 1-style wins and F1 gains."""
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        grouped[(str(row["model"]), int(row["q"]))].append(row)

    summary_rows = []
    for model, q in table_configs():
        rows = grouped[(model, q)]
        qsvm_f1 = mean_metric(rows, "qsvm", "f1")
        linear_f1 = mean_metric(rows, "linear", "f1")
        rbf_f1 = mean_metric(rows, "rbf", "f1")
        tier1_delta = qsvm_f1 - linear_f1
        tier2_delta = qsvm_f1 - rbf_f1
        row = {
            "model": model,
            "q": q,
            "tier1_config": (model, q) in TIER1_CONFIGS,
            "tier2_config": (model, q) in TIER2_CONFIGS,
            "qsvm_f1_mean": qsvm_f1,
            "linear_c1_f1_mean": linear_f1,
            "rbf_best_c_f1_mean": rbf_f1,
            "tier1_f1_gain": tier1_delta,
            "tier2_f1_gain": tier2_delta,
            "tier1_win": bool(tier1_delta > 0) if (model, q) in TIER1_CONFIGS else "",
            "tier2_win": bool(tier2_delta > 0) if (model, q) in TIER2_CONFIGS else "",
        }
        summary_rows.append(row)

    tier1_rows = [row for row in summary_rows if row["tier1_config"]]
    tier2_rows = [row for row in summary_rows if row["tier2_config"]]
    tier1_wins = sum(1 for row in tier1_rows if row["tier1_win"] is True)
    tier2_wins = sum(1 for row in tier2_rows if row["tier2_win"] is True)
    tier1_gains = [float(row["tier1_f1_gain"]) for row in tier1_rows]
    tier2_gains = [float(row["tier2_f1_gain"]) for row in tier2_rows]
    aggregate = {
        "tier1": {
            "wins": tier1_wins,
            "total": len(tier1_rows),
            "mean_f1_gain": float(np.mean(tier1_gains)),
            "comparison": "QSVM C=1 vs linear SVM C=1",
        },
        "tier2": {
            "wins": tier2_wins,
            "total": len(tier2_rows),
            "mean_f1_gain": float(np.mean(tier2_gains)),
            "comparison": "QSVM C=1 vs validation-selected best-C RBF SVM",
        },
        "win_rule": "per-configuration mean test F1 over seeds; win if QSVM mean F1 is greater",
    }
    return summary_rows, aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    tier1 = payload["aggregate"]["tier1"]
    tier2 = payload["aggregate"]["tier2"]
    lines = [
        "# Synthetic surrogate Table 1 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        "| Tier | Comparison | Wins / total | Mean F1 gain |",
        "| --- | --- | ---: | ---: |",
        f"| Tier 1 | {tier1['comparison']} | {tier1['wins']} / {tier1['total']} | {tier1['mean_f1_gain']:.6f} |",
        f"| Tier 2 | {tier2['comparison']} | {tier2['wins']} / {tier2['total']} | {tier2['mean_f1_gain']:.6f} |",
        "",
        f"Win rule: {payload['aggregate']['win_rule']}.",
        "",
        "Data source metadata:",
        "",
        "```json",
        json.dumps(payload["data"], indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def default_prefix(source: str) -> str:
    if source == "synthetic":
        return "synthetic_surrogate_table1"
    return "real_table1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--c-grid", default="0.1,1.0,10.0")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=0.30)
    parser.add_argument("--noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    c_grid = parse_floats(args.c_grid)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    long_rows: list[dict[str, object]] = []
    for model, q in table_configs():
        for seed in seeds:
            long_rows.extend(
                run_one_config(
                    source=args.source,
                    model=model,
                    q=q,
                    seed=seed,
                    data_root=args.data_root,
                    synthetic=synthetic,
                    c_grid=c_grid,
                )
            )

    summary_rows, aggregate = summarize(long_rows)
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
        "aggregate": aggregate,
        "paths": {
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source == "synthetic",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "seeds": seeds,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "c_grid": c_grid,
            "table1_tier1_configs": list(TIER1_CONFIGS),
            "table1_tier2_configs": list(TIER2_CONFIGS),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
