#!/usr/bin/env python3
"""Compute a Table 5-style kernel effective-rank diagnostic.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 5 diagnostic shape: PCA variance, linear-kernel rank,
linear-kernel effective rank, and quantum-kernel effective rank.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from lib.quantum_kernel import effective_rank, fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset

PAPER_TABLE5_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T5"

TABLE5_CONFIGS: tuple[tuple[str, int], ...] = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("medsiglip-448", 11),
    ("medsiglip-448", 16),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
)


def count_positive_eigenvalues(kernel: np.ndarray, tol: float = 1e-10) -> int:
    eigenvalues = np.linalg.eigvalsh(kernel)
    return int(np.sum(eigenvalues > tol))


def compute_table5_row(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> dict[str, object]:
    """Compute the Table 5 spectral diagnostics for one model/q pair."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, _, explained_variance_ratio = preprocess(
        X[idx_train], X[idx_val], X[idx_test], q
    )

    linear_kernel = X_train @ X_train.T
    quantum_kernel = fidelity_kernel(X_train)
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        "linear_kernel_positive_rank": count_positive_eigenvalues(linear_kernel),
        "linear_kernel_effective_rank": effective_rank(linear_kernel),
        "quantum_kernel_effective_rank": effective_rank(quantum_kernel),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "# Synthetic surrogate Table 5 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE5_POINTER}",
        "",
        "Table 5 is a spectral diagnostic: it compares the low-rank linear kernel with the richer quantum fidelity kernel after the same PCA-q preprocessing.",
        "",
        "| Model | q | PCA variance % | Linear rank | Linear eff. rank | Quantum eff. rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {q} | {pca_var} | {linear_rank} | {linear_eff} | {quantum_eff} |".format(
                model=row["model"],
                q=row["q"],
                pca_var=format_float(row["pca_variance_percent"]),
                linear_rank=row["linear_kernel_positive_rank"],
                linear_eff=format_float(row["linear_kernel_effective_rank"]),
                quantum_eff=format_float(row["quantum_kernel_effective_rank"]),
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
        return "synthetic_surrogate_table5"
    if source == "synthetic_file":
        return "synthetic_file_table5"
    return "real_table5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
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

    summary_rows = [
        compute_table5_row(
            source=args.source,
            model=model,
            q=q,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
        )
        for model, q in TABLE5_CONFIGS
    ]

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(summary_path, summary_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 5",
        "paper_pointer": PAPER_TABLE5_POINTER,
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
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "table5_configs": list(TABLE5_CONFIGS),
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
