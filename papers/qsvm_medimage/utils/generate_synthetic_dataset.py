#!/usr/bin/env python3
"""Materialize the synthetic fallback dataset to local ``.npz`` files.

The output layout mirrors the gated Hugging Face embedding dataset closely
enough for local runners:

    <output-root>/medsiglip-448-embeddings/20-seeds/seed_0/data_type9_synthetic.npz
    <output-root>/rad-dino-embeddings/20-seeds/seed_0/data_type9_synthetic.npz
    <output-root>/vit-base-patch32-224-embeddings/20-seeds/seed_0/data_type9_synthetic.npz

This is a synthetic kernel-geometry benchmark, not a medical dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.synthetic_data import (  # noqa: E402
    default_embedding_dim,
    default_n_samples,
    make_synthetic_embeddings,
)

MODEL_LAYOUT = {
    "synthetic_medsiglip": "medsiglip-448-embeddings/20-seeds",
    "synthetic_raddino": "rad-dino-embeddings/20-seeds",
    "synthetic_vit": "vit-base-patch32-224-embeddings/20-seeds",
}


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    for model in models:
        if model not in MODEL_LAYOUT:
            raise ValueError(f"unknown synthetic model: {model}")
    return models


def write_one_dataset(
    *,
    output_root: Path,
    model_name: str,
    seed: int,
    n_samples: int | None,
    embedding_dim: int | None,
    positive_ratio: float,
    n_signal_latents: int | None,
    n_nuisance_latents: int | None,
    signal_strength: float | None,
    noise_std: float | None,
    nuisance_scale: float | None,
    nuisance_decay: float | None,
    nuisance_distribution: str | None,
    nuisance_skew_strength: float | None,
    nuisance_skew_decay: float | None,
    dtype: str,
) -> dict[str, object]:
    """Generate and save one model/seed split."""
    resolved_n_samples = n_samples or default_n_samples(model_name)
    resolved_embedding_dim = embedding_dim or default_embedding_dim(model_name)
    X, y, metadata = make_synthetic_embeddings(
        n_samples=resolved_n_samples,
        embedding_dim=resolved_embedding_dim,
        positive_ratio=positive_ratio,
        n_signal_latents=n_signal_latents,
        n_nuisance_latents=n_nuisance_latents,
        signal_strength=signal_strength,
        noise_std=noise_std,
        nuisance_scale=nuisance_scale,
        nuisance_decay=nuisance_decay,
        nuisance_distribution=nuisance_distribution,
        nuisance_skew_strength=nuisance_skew_strength,
        nuisance_skew_decay=nuisance_skew_decay,
        seed=seed,
        model_name=model_name,
    )

    seed_dir = output_root / MODEL_LAYOUT[model_name] / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    npz_path = seed_dir / "data_type9_synthetic.npz"
    metadata_path = seed_dir / "data_type9_synthetic_metadata.json"

    X_to_save = X.astype(np.float32 if dtype == "float32" else np.float64)
    np.savez_compressed(npz_path, X=X_to_save, y=y.astype(np.int8))
    metadata = {
        **metadata,
        "file": str(npz_path),
        "storage_dtype": dtype,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def write_dataset_index(path: Path, rows: list[dict[str, object]]) -> None:
    index = {
        "source": "synthetic",
        "description": "Materialized synthetic fallback for qsvm_medimage kernel-geometry tests.",
        "rows": rows,
    }
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/synthetic_qml_mimic_cxr_embeddings"))
    parser.add_argument(
        "--models",
        default="synthetic_medsiglip,synthetic_raddino,synthetic_vit",
        help="Comma-separated synthetic model names.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--positive-ratio", type=float, default=0.304)
    parser.add_argument("--n-signal-latents", type=int, default=None)
    parser.add_argument("--n-nuisance-latents", type=int, default=None)
    parser.add_argument("--signal-strength", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--nuisance-scale", type=float, default=None)
    parser.add_argument("--nuisance-decay", type=float, default=None)
    parser.add_argument(
        "--nuisance-distribution",
        choices=("normal", "uniform", "rademacher", "skewed_uniform"),
        default=None,
    )
    parser.add_argument("--nuisance-skew-strength", type=float, default=None)
    parser.add_argument("--nuisance-skew-decay", type=float, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    seeds = parse_ints(args.seeds)

    rows = []
    for model_name in models:
        for seed in seeds:
            rows.append(
                write_one_dataset(
                    output_root=args.output_root,
                    model_name=model_name,
                    seed=seed,
                    n_samples=args.n_samples,
                    embedding_dim=args.embedding_dim,
                    positive_ratio=args.positive_ratio,
                    n_signal_latents=args.n_signal_latents,
                    n_nuisance_latents=args.n_nuisance_latents,
                    signal_strength=args.signal_strength,
                    noise_std=args.noise_std,
                    nuisance_scale=args.nuisance_scale,
                    nuisance_decay=args.nuisance_decay,
                    nuisance_distribution=args.nuisance_distribution,
                    nuisance_skew_strength=args.nuisance_skew_strength,
                    nuisance_skew_decay=args.nuisance_skew_decay,
                    dtype=args.dtype,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "synthetic_dataset_index.json"
    write_dataset_index(index_path, rows)
    print(json.dumps({"files": len(rows), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
