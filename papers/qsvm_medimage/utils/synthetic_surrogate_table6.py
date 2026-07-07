#!/usr/bin/env python3
"""Compute a Table 6-style linear-kernel variance diagnostic.

This is not a reproduction of the paper numbers when ``--source synthetic`` is
used. It mirrors the Table 6 diagnostic shape: linear-kernel mean, standard
deviation, and variance on a 200-sample training subset sorted by class.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from lib.svm_pipeline import preprocess, split_indices
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset

PAPER_TABLE6_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T6"

TABLE6_CONFIGS: tuple[tuple[str, int], ...] = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def select_sorted_training_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    subsample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic training subset and sort it by class label."""
    n_selected = min(subsample_size, len(y_train))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(y_train), size=n_selected, replace=False)
    selected = selected[np.argsort(y_train[selected], kind="stable")]
    return X_train[selected], y_train[selected]


def linear_kernel_stats(kernel: np.ndarray) -> dict[str, float]:
    """Compute the three statistics reported in paper Table 6."""
    return {
        "k_l_mean": float(np.mean(kernel)),
        "k_l_std": float(np.std(kernel)),
        "k_l_var": float(np.var(kernel)),
    }


def compute_table6_row(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    subsample_size: int,
) -> dict[str, object]:
    """Compute the Table 6 linear-kernel statistics for one model/q pair."""
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
    y_train = y[idx_train]
    X_subset, y_subset = select_sorted_training_subset(
        X_train,
        y_train,
        seed=seed,
        subsample_size=subsample_size,
    )
    linear_kernel = X_subset @ X_subset.T
    stats = linear_kernel_stats(linear_kernel)
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "model_display": MODEL_DISPLAY[model],
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "subsample_size": int(len(y_subset)),
        "subsample_class_0": int(np.sum(y_subset == 0)),
        "subsample_class_1": int(np.sum(y_subset == 1)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        **stats,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    lines = [
        "# Synthetic surrogate Table 6 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper numbers because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_TABLE6_POINTER}",
        "",
        "Table 6 is a linear-kernel diagnostic: it reports the mean, standard deviation, and variance of the PCA-q linear kernel on 200 subsampled training samples sorted by class.",
        "",
        "| Model | q | K_L mean | K_L std | K_L var |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            "| {model} | {q} | {mean} | {std} | {var} |".format(
                model=row["model_display"],
                q=row["q"],
                mean=format_float(row["k_l_mean"]),
                std=format_float(row["k_l_std"]),
                var=format_float(row["k_l_var"]),
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
        return "synthetic_surrogate_table6"
    if source == "synthetic_file":
        return "synthetic_file_table6"
    return "real_table6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample-size", type=int, default=200)
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
        compute_table6_row(
            source=args.source,
            model=model,
            q=q,
            seed=args.seed,
            data_root=args.data_root,
            synthetic=synthetic,
            subsample_size=args.subsample_size,
        )
        for model, q in TABLE6_CONFIGS
    ]

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / f"{prefix}_summary.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"
    write_csv(summary_path, summary_rows)

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 6",
        "paper_pointer": PAPER_TABLE6_POINTER,
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
            "subsample_size": args.subsample_size,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "table6_configs": list(TABLE6_CONFIGS),
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
