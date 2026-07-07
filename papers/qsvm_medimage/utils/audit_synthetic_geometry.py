#!/usr/bin/env python3
"""Audit embedding and kernel geometry before tuning any synthetic generator.

This script is observational. It loads a fixed data source, applies the paper
preprocessing, and records geometry diagnostics that can be compared with paper
figures/tables. It does not train classifiers and does not modify data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT, Path(__file__).resolve().parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset  # noqa: E402


MODEL_NAMES = ("medsiglip-448", "rad-dino", "vit-patch32-cls")
DEFAULT_Q_VALUES = "4,6,8,11,16"

# Targets transcribed from the paper Table V screenshot used for geometry
# comparison only. Missing entries mean the paper table did not report a value.
PAPER_TABLE5_TARGETS: dict[tuple[str, int], dict[str, float | int | None | str]] = {
    ("medsiglip-448", 4): {
        "linear_comparison": True,
        "pca_var_percent": 32.6,
        "positive_rank": 4,
        "effective_rank": 3.77,
        "lambda_max": 770.6,
    },
    ("medsiglip-448", 6): {
        "linear_comparison": True,
        "pca_var_percent": 41.1,
        "positive_rank": 6,
        "effective_rank": 5.53,
        "lambda_max": 614.8,
    },
    ("medsiglip-448", 11): {
        "linear_comparison": False,
        "pca_var_percent": 56.0,
        "positive_rank": None,
        "effective_rank": 43.04,
        "lambda_max": 468.6,
        "note": "dagger row; rank/effective-rank/lambda_max are not linear PCA-q targets",
    },
    ("medsiglip-448", 16): {
        "linear_comparison": False,
        "pca_var_percent": None,
        "positive_rank": None,
        "effective_rank": 92.13,
        "lambda_max": None,
        "note": "dagger row; not a linear PCA-q target",
    },
    ("rad-dino", 4): {
        "linear_comparison": True,
        "pca_var_percent": 5.6,
        "positive_rank": 4,
        "effective_rank": 3.89,
        "lambda_max": 666.1,
    },
    ("rad-dino", 6): {
        "linear_comparison": True,
        "pca_var_percent": 7.7,
        "positive_rank": 6,
        "effective_rank": 5.85,
        "lambda_max": 475.5,
    },
    ("vit-patch32-cls", 4): {
        "linear_comparison": True,
        "pca_var_percent": 28.4,
        "positive_rank": 4,
        "effective_rank": 3.86,
        "lambda_max": 627.5,
    },
    ("vit-patch32-cls", 6): {
        "linear_comparison": True,
        "pca_var_percent": 34.7,
        "positive_rank": 6,
        "effective_rank": 5.59,
        "lambda_max": 502.3,
    },
}


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def effective_rank_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 1.0
    p = eigenvalues / total
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def positive_rank_from_eigenvalues(eigenvalues: np.ndarray) -> int:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if len(eigenvalues) == 0:
        return 0
    tol = max(1e-12, 1e-10 * float(np.max(eigenvalues)))
    return int(np.sum(eigenvalues > tol))


def components_needed(cumulative: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(cumulative >= threshold)
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def summarize_kernel(kernel: np.ndarray, prefix: str) -> dict[str, float | int]:
    kernel = np.asarray(kernel, dtype=float)
    kernel = 0.5 * (kernel + kernel.T)
    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    diag = np.diag(kernel)
    offdiag = off_diagonal_values(kernel)
    offdiag_stats = finite_stats(offdiag)
    return {
        f"{prefix}_trace": float(np.trace(kernel)),
        f"{prefix}_diag_mean": float(np.mean(diag)),
        f"{prefix}_diag_std": float(np.std(diag)),
        f"{prefix}_offdiag_mean": offdiag_stats["mean"],
        f"{prefix}_offdiag_std": offdiag_stats["std"],
        f"{prefix}_offdiag_min": offdiag_stats["min"],
        f"{prefix}_offdiag_max": offdiag_stats["max"],
        f"{prefix}_positive_rank": positive_rank_from_eigenvalues(eigenvalues),
        f"{prefix}_effective_rank": effective_rank_from_eigenvalues(eigenvalues),
    }


def summarize_linear_full_rank(X_train: np.ndarray) -> dict[str, float | int]:
    # Non-zero eigenvalues of X X^T are the eigenvalues of X^T X.
    eigenvalues = np.linalg.eigvalsh(X_train.T @ X_train)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambda_max = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    trace = float(np.sum(eigenvalues))
    trace_normalized_lambda_max = lambda_max / trace if trace > 0.0 else 0.0
    trace_n_lambda_max = trace_normalized_lambda_max * X_train.shape[0]
    return {
        "linear_full_positive_rank": positive_rank_from_eigenvalues(eigenvalues),
        "linear_full_effective_rank": effective_rank_from_eigenvalues(eigenvalues),
        "linear_full_lambda_max_raw": lambda_max,
        "linear_full_lambda_max_trace_1": trace_normalized_lambda_max,
        "linear_full_lambda_max_trace_n": trace_n_lambda_max,
        "linear_full_lambda_max": lambda_max,
        "linear_full_trace": trace,
    }


def maybe_delta(value: float | int | None, target: float | int | None) -> float | None:
    if value is None or target is None:
        return None
    return float(value) - float(target)


def maybe_ratio(value: float | int | None, target: float | int | None) -> float | None:
    if value is None or target is None or float(target) == 0.0:
        return None
    return float(value) / float(target)


def add_paper_table5_comparison(row: dict[str, object]) -> None:
    target = PAPER_TABLE5_TARGETS.get((str(row["model"]), int(row["q"])))
    row["paper_table5_has_target"] = bool(target)
    if not target:
        return

    paper_pca = target.get("pca_var_percent")
    paper_rank = target.get("positive_rank")
    paper_eff_rank = target.get("effective_rank")
    paper_lambda_max = target.get("lambda_max")
    linear_comparison = bool(target.get("linear_comparison", True))

    current_pca_percent = 100.0 * float(row["pca_explained_variance"])
    current_rank = row.get("linear_full_positive_rank")
    current_eff_rank = row.get("linear_full_effective_rank")
    current_lambda_max = row.get("linear_full_lambda_max")
    current_lambda_max_trace_n = row.get("linear_full_lambda_max_trace_n")

    row["paper_table5_linear_comparison"] = linear_comparison
    row["paper_table5_pca_var_percent"] = paper_pca
    row["paper_table5_positive_rank"] = paper_rank
    row["paper_table5_effective_rank"] = paper_eff_rank
    row["paper_table5_lambda_max"] = paper_lambda_max
    row["paper_table5_lambda_max_note"] = "paper normalization unknown; inspect raw and trace_n"
    row["paper_table5_note"] = target.get("note", "")
    row["delta_pca_var_percent_vs_table5"] = maybe_delta(current_pca_percent, paper_pca)
    row["ratio_pca_var_percent_vs_table5"] = maybe_ratio(current_pca_percent, paper_pca)
    row["delta_positive_rank_vs_table5"] = maybe_delta(current_rank, paper_rank) if linear_comparison else None
    row["delta_effective_rank_vs_table5"] = maybe_delta(current_eff_rank, paper_eff_rank) if linear_comparison else None
    row["ratio_effective_rank_vs_table5"] = maybe_ratio(current_eff_rank, paper_eff_rank) if linear_comparison else None
    row["delta_lambda_max_vs_table5"] = maybe_delta(current_lambda_max, paper_lambda_max) if linear_comparison else None
    row["ratio_lambda_max_vs_table5"] = maybe_ratio(current_lambda_max, paper_lambda_max) if linear_comparison else None
    row["delta_lambda_max_raw_vs_table5"] = maybe_delta(current_lambda_max, paper_lambda_max) if linear_comparison else None
    row["ratio_lambda_max_raw_vs_table5"] = maybe_ratio(current_lambda_max, paper_lambda_max) if linear_comparison else None
    row["delta_lambda_max_trace_n_vs_table5"] = (
        maybe_delta(current_lambda_max_trace_n, paper_lambda_max) if linear_comparison else None
    )
    row["ratio_lambda_max_trace_n_vs_table5"] = (
        maybe_ratio(current_lambda_max_trace_n, paper_lambda_max) if linear_comparison else None
    )


def summarize_features(X: np.ndarray, prefix: str) -> dict[str, float]:
    norms = np.linalg.norm(X, axis=1)
    saturation = np.mean(np.abs(X) >= 0.999)
    row_mean = np.mean(X, axis=1)
    row_std = np.std(X, axis=1)
    return {
        f"{prefix}_feature_mean": float(np.mean(X)),
        f"{prefix}_feature_std": float(np.std(X)),
        f"{prefix}_feature_min": float(np.min(X)),
        f"{prefix}_feature_max": float(np.max(X)),
        f"{prefix}_feature_saturation_fraction": float(saturation),
        f"{prefix}_row_norm_mean": float(np.mean(norms)),
        f"{prefix}_row_norm_std": float(np.std(norms)),
        f"{prefix}_row_mean_mean": float(np.mean(row_mean)),
        f"{prefix}_row_std_mean": float(np.mean(row_std)),
    }


def summarize_pairwise_geometry(X: np.ndarray, prefix: str) -> dict[str, float]:
    if len(X) < 2:
        return {}
    gram = X @ X.T
    squared_norms = np.diag(gram)
    distances2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram
    distances = np.sqrt(np.maximum(distances2, 0.0))
    norms = np.sqrt(np.maximum(squared_norms, 0.0))
    denom = np.outer(norms, norms)
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0.0)
    distance_stats = finite_stats(off_diagonal_values(distances))
    cosine_stats = finite_stats(off_diagonal_values(cosine))
    return {
        f"{prefix}_distance_mean": distance_stats["mean"],
        f"{prefix}_distance_std": distance_stats["std"],
        f"{prefix}_distance_min": distance_stats["min"],
        f"{prefix}_distance_max": distance_stats["max"],
        f"{prefix}_cosine_mean": cosine_stats["mean"],
        f"{prefix}_cosine_std": cosine_stats["std"],
        f"{prefix}_cosine_min": cosine_stats["min"],
        f"{prefix}_cosine_max": cosine_stats["max"],
    }


def pca_spectrum_rows(
    *,
    X_train_raw: np.ndarray,
    model: str,
    seed: int,
    max_components: int,
) -> list[dict[str, object]]:
    scaler = StandardScaler().fit(X_train_raw)
    X_scaled = scaler.transform(X_train_raw)
    n_components = min(max_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components).fit(X_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rows = []
    for index, value in enumerate(pca.explained_variance_ratio_):
        rows.append(
            {
                "model": model,
                "seed": seed,
                "component": index + 1,
                "explained_variance_ratio": float(value),
                "cumulative_explained_variance": float(cumulative[index]),
            }
        )
    return rows


def audit_one_q(
    *,
    X: np.ndarray,
    y: np.ndarray,
    model: str,
    seed: int,
    q: int,
    source: str,
    kernel_samples: int,
    max_quantum_q: int,
    skip_quantum: bool,
) -> dict[str, object]:
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, X_val, X_test, pca_explained = preprocess(
        X[idx_train], X[idx_val], X[idx_test], q
    )
    n_kernel = min(kernel_samples, len(X_train))
    X_kernel = X_train[:n_kernel]

    row: dict[str, object] = {
        "source": source,
        "model": model,
        "seed": seed,
        "q": q,
        "n_samples": int(len(y)),
        "n_features_raw": int(X.shape[1]),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "positive_ratio_all": float(np.mean(y)),
        "positive_ratio_train": float(np.mean(y[idx_train])),
        "pca_explained_variance": float(pca_explained),
        "kernel_samples": int(n_kernel),
    }
    row.update(summarize_features(X_train, "train_minmax"))
    row.update(summarize_pairwise_geometry(X_kernel, "train_minmax"))

    linear_kernel_sample = X_kernel @ X_kernel.T
    row.update(summarize_linear_full_rank(X_train))
    row.update(summarize_kernel(linear_kernel_sample, "linear_sample_kernel"))
    row["linear_rank_validation"] = bool(row["linear_full_positive_rank"] <= q)

    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    rbf_sample = rbf_kernel(X_kernel, gamma=gamma_scale)
    row["rbf_gamma_scale"] = float(gamma_scale)
    row.update(summarize_kernel(rbf_sample, "rbf_sample_kernel"))

    if skip_quantum or q > max_quantum_q:
        row["quantum_sample_kernel_skipped"] = True
        row["quantum_sample_kernel_skip_reason"] = "disabled" if skip_quantum else f"q>{max_quantum_q}"
    else:
        quantum_sample = fidelity_kernel(X_kernel)
        row["quantum_sample_kernel_skipped"] = False
        row["quantum_sample_kernel_skip_reason"] = ""
        row.update(summarize_kernel(quantum_sample, "quantum_sample_kernel"))
    add_paper_table5_comparison(row)
    return row


def audit_dataset(
    *,
    source: str,
    data_root: Path | None,
    models: list[str],
    seeds: list[int],
    q_values: list[int],
    synthetic: SyntheticSpec,
    max_pca_components: int,
    kernel_samples: int,
    max_quantum_q: int,
    skip_quantum: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary_rows = []
    pca_rows = []
    dataset_rows = []
    for model in models:
        for seed in seeds:
            X, y = load_dataset(
                source=source,
                model=model,
                seed=seed,
                data_root=data_root,
                synthetic=synthetic,
            )
            idx_train, idx_val, idx_test = split_indices(y, seed=seed)
            dataset_rows.append(
                {
                    "source": source,
                    "model": model,
                    "seed": seed,
                    "n_samples": int(len(y)),
                    "n_features_raw": int(X.shape[1]),
                    "n_train": int(len(idx_train)),
                    "n_val": int(len(idx_val)),
                    "n_test": int(len(idx_test)),
                    "positive_count": int(np.sum(y == 1)),
                    "negative_count": int(np.sum(y == 0)),
                    "positive_ratio_all": float(np.mean(y)),
                    "positive_ratio_train": float(np.mean(y[idx_train])),
                    "positive_ratio_val": float(np.mean(y[idx_val])),
                    "positive_ratio_test": float(np.mean(y[idx_test])),
                }
            )
            pca_rows.extend(
                pca_spectrum_rows(
                    X_train_raw=X[idx_train],
                    model=model,
                    seed=seed,
                    max_components=max_pca_components,
                )
            )
            for q in q_values:
                if q > min(X.shape[1], len(idx_train)):
                    continue
                summary_rows.append(
                    audit_one_q(
                        X=X,
                        y=y,
                        model=model,
                        seed=seed,
                        q=q,
                        source=source,
                        kernel_samples=kernel_samples,
                        max_quantum_q=max_quantum_q,
                        skip_quantum=skip_quantum,
                    )
                )
    return summary_rows, pca_rows, dataset_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    data = payload["data"]
    paths = payload["paths"]
    lines = [
        "# Synthetic Geometry Audit",
        "",
        "This is an observational audit. It does not tune the generator, train classifiers, or claim reproduction of the gated medical task.",
        "",
        "Purpose: compare geometry before changing the synthetic dataset.",
        "",
        "Inputs:",
        "",
        f"- source: `{data['source']}`",
        f"- data_root: `{data['data_root']}`",
        f"- models: `{', '.join(data['models'])}`",
        f"- seeds: `{data['seeds']}`",
        f"- q_values: `{data['q_values']}`",
        f"- kernel_samples: `{data['kernel_samples']}`",
        f"- max_quantum_q: `{data['max_quantum_q']}`",
        "",
        "Outputs:",
        "",
        f"- dataset summary: `{Path(paths['dataset_csv']).name}`",
        f"- PCA spectrum: `{Path(paths['pca_csv']).name}`",
        f"- q-level geometry: `{Path(paths['summary_csv']).name}`",
        f"- JSON payload: `{Path(paths['json']).name}`",
        "",
        "Key checks to inspect first:",
        "",
        "- `linear_rank_validation` should be true for every q.",
        "- `linear_full_positive_rank` should be at most q after PCA-q.",
        "- `pca_explained_variance`, `linear_full_effective_rank`, and `linear_full_lambda_max` should be compared against Table V before using classification results.",
        "- `linear_full_lambda_max` is the raw value; `linear_full_lambda_max_trace_n` is the same spectrum after trace normalization to `trace(K)=n_train`, which may be closer to the paper's reported scale.",
        "- `paper_table5_linear_comparison` is false for dagger rows whose reported ranks cannot be linear PCA-q kernel ranks.",
        "- `paper_table5_*`, `delta_*_vs_table5`, and `ratio_*_vs_table5` are geometry comparisons transcribed from the paper table; they are not optimisation targets.",
        "- `quantum_sample_kernel_effective_rank` is computed on a subset; use it as a geometry diagnostic, not a final paper number.",
    ]
    path.write_text("\n".join(lines) + "\n")


def default_prefix(source: str) -> str:
    if source == "synthetic_file":
        return "synthetic_file_geometry_audit"
    if source == "synthetic":
        return "synthetic_surrogate_geometry_audit"
    return "real_geometry_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic_file", "synthetic", "real"), default="synthetic_file")
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic_qml_mimic_cxr_embeddings"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--q-values", default=DEFAULT_Q_VALUES)
    parser.add_argument("--max-pca-components", type=int, default=200)
    parser.add_argument("--kernel-samples", type=int, default=200)
    parser.add_argument("--max-quantum-q", type=int, default=11)
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_strings(args.models)
    seeds = parse_ints(args.seeds)
    q_values = parse_ints(args.q_values)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    data_root = args.data_root if args.source != "synthetic" else None
    summary_rows, pca_rows, dataset_rows = audit_dataset(
        source=args.source,
        data_root=data_root,
        models=models,
        seeds=seeds,
        q_values=q_values,
        synthetic=synthetic,
        max_pca_components=args.max_pca_components,
        kernel_samples=args.kernel_samples,
        max_quantum_q=args.max_quantum_q,
        skip_quantum=args.skip_quantum,
    )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.results_dir / f"{prefix}_dataset.csv"
    pca_path = args.results_dir / f"{prefix}_pca.csv"
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    write_csv(dataset_path, dataset_rows)
    write_csv(pca_path, pca_rows)
    write_csv(summary_path, summary_rows)

    payload = {
        "artifact": prefix,
        "paths": {
            "dataset_csv": str(dataset_path),
            "pca_csv": str(pca_path),
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "data_root": str(data_root) if data_root else None,
            "models": models,
            "seeds": seeds,
            "q_values": q_values,
            "max_pca_components": args.max_pca_components,
            "kernel_samples": args.kernel_samples,
            "max_quantum_q": args.max_quantum_q,
            "skip_quantum": args.skip_quantum,
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
        },
        "n_dataset_rows": len(dataset_rows),
        "n_pca_rows": len(pca_rows),
        "n_summary_rows": len(summary_rows),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
