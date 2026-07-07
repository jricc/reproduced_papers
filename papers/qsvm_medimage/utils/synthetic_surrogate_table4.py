#!/usr/bin/env python3
"""Compute a Table 4-style Tier 2 detail table on surrogate or real data.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 4 protocol shape: QSVM C=1 versus best-C RBF SVM,
reported over the seven Tier 2 configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import (
    TIER2_CONFIGS,
    SyntheticSpec,
    load_dataset,
    parse_floats,
    parse_ints,
    score_qsvm_c1,
    score_rbf_svc_best_c,
)

PAPER_TABLE4_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T4"


def metric_mean(rows: list[dict[str, object]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method]
    return float(np.mean(values))


def metric_std(rows: list[dict[str, object]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def rbf_c_values(rows: list[dict[str, object]]) -> str:
    values = sorted({float(row["C"]) for row in rows if row["method"] == "rbf"})
    return ",".join(f"{value:g}" for value in values)


def run_one_table4_config(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    c_grid: list[float],
) -> list[dict[str, object]]:
    """Run the two Table 4 Tier 2 comparators for one model/q/seed."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, X_val, X_test, explained_variance_ratio = preprocess(
        X[idx_train], X[idx_val], X[idx_test], q
    )
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    run_metadata = {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "val_samples": int(len(idx_val)),
        "test_samples": int(len(idx_test)),
        "explained_variance_ratio": float(explained_variance_ratio),
    }

    qsvm_row = score_qsvm_c1(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
    )
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
    return [
        {**run_metadata, **qsvm_row, "selected_on": "fixed"},
        {**run_metadata, **rbf_row, "selected_on": "validation_f1"},
    ]


def summarize_table4(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate seed-level rows into the paper Table 4 column layout."""
    summary_rows = []
    for model, q in TIER2_CONFIGS:
        rows = [row for row in long_rows if row["model"] == model and row["q"] == q]
        qsvm_f1 = metric_mean(rows, "qsvm", "f1")
        rbf_f1 = metric_mean(rows, "rbf", "f1")
        f1_gain = qsvm_f1 - rbf_f1
        relative_gain = float("nan") if rbf_f1 == 0 else 100.0 * f1_gain / rbf_f1
        summary_rows.append(
            {
                "model": model,
                "q": q,
                "qsvm_accuracy_mean": metric_mean(rows, "qsvm", "accuracy"),
                "qsvm_accuracy_std": metric_std(rows, "qsvm", "accuracy"),
                "qsvm_f1_mean": qsvm_f1,
                "qsvm_f1_std": metric_std(rows, "qsvm", "f1"),
                "best_svm_kernel": "rbf",
                "best_svm_selected_c_values": rbf_c_values(rows),
                "best_svm_f1_mean": rbf_f1,
                "best_svm_f1_std": metric_std(rows, "rbf", "f1"),
                "f1_gain": f1_gain,
                "relative_gain_percent": relative_gain,
                "verdict": "QSVM F1 WIN" if f1_gain > 0 else "RBF F1 WIN" if f1_gain < 0 else "F1 TIE",
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean: float, std: float) -> str:
    if np.isnan(std):
        return f"{mean:.3f} +/- NA"
    return f"{mean:.3f} +/- {std:.3f}"


def format_relative_gain(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:+.1f}%"


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "# Synthetic surrogate Table 4 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE4_POINTER}",
        "",
        "Table 4 reports the Tier 2 comparison: QSVM C=1 versus best-C RBF SVM at equal PCA-q dimensionality, with C selected on validation F1 in this surrogate pipeline.",
        "",
        "| Model | q | QSVM acc | QSVM F1 | Best SVM | Best SVM F1 | F1 gain | Rel. gain | Verdict |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {q} | {qacc} | {qf1} | {kernel} C={c_values} | {sf1} | {gain:+.3f} | {rel} | {verdict} |".format(
                model=row["model"],
                q=row["q"],
                qacc=format_mean_std(row["qsvm_accuracy_mean"], row["qsvm_accuracy_std"]),
                qf1=format_mean_std(row["qsvm_f1_mean"], row["qsvm_f1_std"]),
                kernel=row["best_svm_kernel"],
                c_values=row["best_svm_selected_c_values"],
                sf1=format_mean_std(row["best_svm_f1_mean"], row["best_svm_f1_std"]),
                gain=row["f1_gain"],
                rel=format_relative_gain(row["relative_gain_percent"]),
                verdict=row["verdict"],
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
        return "synthetic_surrogate_table4"
    if source == "synthetic_file":
        return "synthetic_file_table4"
    return "real_table4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--c-grid", default="0.1,1.0,10.0")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=1.0)
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
    for model, q in TIER2_CONFIGS:
        for seed in seeds:
            long_rows.extend(
                run_one_table4_config(
                    source=args.source,
                    model=model,
                    q=q,
                    seed=seed,
                    data_root=args.data_root,
                    synthetic=synthetic,
                    c_grid=c_grid,
                )
            )

    summary_rows = summarize_table4(long_rows)
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
        "paper_table": "Table 4",
        "paper_pointer": PAPER_TABLE4_POINTER,
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
            "seeds": seeds,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "c_grid": c_grid,
            "c_selection": "per-seed validation F1 in this surrogate pipeline",
            "table4_configs": list(TIER2_CONFIGS),
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    wins = sum(1 for row in summary_rows if row["verdict"] == "QSVM F1 WIN")
    mean_gain = float(np.mean([row["f1_gain"] for row in summary_rows]))
    print(json.dumps({"qsvm_f1_wins": wins, "total": len(summary_rows), "mean_f1_gain": mean_gain}, indent=2))
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
