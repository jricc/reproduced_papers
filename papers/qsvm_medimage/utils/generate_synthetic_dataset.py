#!/usr/bin/env python3
"""Materialize the synthetic benchmark as local NPZ files.

The output directory mirrors the layout expected by the real-data loader:

    <output-root>/
        medsiglip-448-embeddings/
            20-seeds/
                seed_0/
                    data_type9_synthetic.npz
                    data_type9_synthetic_metadata.json

        rad-dino-embeddings/
            20-seeds/
                seed_0/
                    data_type9_synthetic.npz
                    data_type9_synthetic_metadata.json

        vit-base-patch32-224-embeddings/
            20-seeds/
                seed_0/
                    data_type9_synthetic.npz
                    data_type9_synthetic_metadata.json

The generated data form a controlled and calibrated kernel benchmark.

They do not reconstruct the distribution of the inaccessible medical
embeddings and have no medical or insurance semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

project_root_string = str(PROJECT_ROOT)

if project_root_string not in sys.path:
    sys.path.insert(
        0,
        project_root_string,
    )


from lib.synthetic_data import (  # noqa: E402
    default_embedding_dim,
    default_n_samples,
    make_synthetic_embeddings,
)

MODEL_LAYOUT = {
    "synthetic_medsiglip": ("medsiglip-448-embeddings/20-seeds"),
    "synthetic_raddino": ("rad-dino-embeddings/20-seeds"),
    "synthetic_vit": ("vit-base-patch32-224-embeddings/20-seeds"),
}

DEFAULT_MODELS = "synthetic_medsiglip,synthetic_raddino,synthetic_vit"

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"


def parse_ints(
    raw: str,
) -> list:
    """Parse a non-empty comma-separated list of unique integers."""
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        ) from error

    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")

    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Values must be unique.")

    return values


def parse_models(
    raw: str,
) -> list:
    """Parse and validate synthetic model names."""
    models = [part.strip() for part in raw.split(",") if part.strip()]

    if not models:
        raise argparse.ArgumentTypeError("At least one model is required.")

    unknown_models = [model for model in models if model not in MODEL_LAYOUT]

    if unknown_models:
        raise argparse.ArgumentTypeError(
            "Unknown synthetic models: " + ", ".join(unknown_models)
        )

    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("Model names must be unique.")

    return models


def validate_generated_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    expected_n_samples: int,
    expected_embedding_dim: int,
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one generated feature matrix and label vector."""
    features = np.asarray(X)

    labels = np.asarray(y)

    if features.ndim != 2:
        raise ValueError(
            "Generated features must be two-dimensional for "
            f"model={model_name}, seed={seed}."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Generated labels must be one-dimensional for "
            f"model={model_name}, seed={seed}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "Generated features and labels have different sample counts "
            f"for model={model_name}, seed={seed}."
        )

    expected_shape = (
        expected_n_samples,
        expected_embedding_dim,
    )

    if features.shape != expected_shape:
        raise ValueError(
            "Generated feature shape is inconsistent. "
            f"Expected {expected_shape}, received {features.shape}, "
            f"model={model_name}, seed={seed}."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Generated features contain non-finite values for "
            f"model={model_name}, seed={seed}."
        )

    unique_labels = np.unique(labels)

    if not np.all(
        np.isin(
            unique_labels,
            np.array(
                [
                    0,
                    1,
                ]
            ),
        )
    ):
        raise ValueError(
            "Generated labels must contain only 0 and 1 for "
            f"model={model_name}, seed={seed}."
        )

    if unique_labels.size != 2:
        raise ValueError(
            "Generated labels must contain both classes for "
            f"model={model_name}, seed={seed}."
        )

    return (
        features,
        labels,
    )


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
    score_noise_std: float | None,
    noise_std: float | None,
    nuisance_scale: float | None,
    nuisance_decay: float | None,
    nuisance_distribution: str | None,
    nuisance_skew_strength: float | None,
    nuisance_skew_decay: float | None,
    dtype: str,
    overwrite: bool,
) -> dict[str, object]:
    """Generate and save one synthetic model and seed combination."""
    resolved_n_samples = (
        default_n_samples(model_name) if n_samples is None else n_samples
    )

    resolved_embedding_dim = (
        default_embedding_dim(model_name) if embedding_dim is None else embedding_dim
    )

    if resolved_n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if resolved_embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")

    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("positive_ratio must be in the interval (0, 1).")

    seed_directory = output_root / MODEL_LAYOUT[model_name] / f"seed_{seed}"

    npz_path = seed_directory / "data_type9_synthetic.npz"

    metadata_path = seed_directory / "data_type9_synthetic_metadata.json"

    if not overwrite and (npz_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            "Synthetic dataset already exists for "
            f"model={model_name}, seed={seed}. "
            "Use --overwrite to replace it."
        )

    (
        X,
        y,
        metadata,
    ) = make_synthetic_embeddings(
        n_samples=resolved_n_samples,
        embedding_dim=resolved_embedding_dim,
        positive_ratio=positive_ratio,
        n_signal_latents=n_signal_latents,
        n_nuisance_latents=n_nuisance_latents,
        signal_strength=signal_strength,
        score_noise_std=score_noise_std,
        noise_std=noise_std,
        nuisance_scale=nuisance_scale,
        nuisance_decay=nuisance_decay,
        nuisance_distribution=(nuisance_distribution),
        nuisance_skew_strength=(nuisance_skew_strength),
        nuisance_skew_decay=(nuisance_skew_decay),
        seed=seed,
        model_name=model_name,
    )

    (
        X,
        y,
    ) = validate_generated_dataset(
        X,
        y,
        expected_n_samples=(resolved_n_samples),
        expected_embedding_dim=(resolved_embedding_dim),
        model_name=model_name,
        seed=seed,
    )

    seed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dtype == "float32":
        storage_dtype = np.float32
    else:
        storage_dtype = np.float64

    X_to_save = X.astype(
        storage_dtype,
        copy=False,
    )

    y_to_save = y.astype(
        np.int8,
        copy=False,
    )

    np.savez_compressed(
        npz_path,
        X=X_to_save,
        y=y_to_save,
    )

    complete_metadata = dict(metadata)

    complete_metadata.update(
        {
            "file": str(npz_path),
            "metadata_file": str(metadata_path),
            "storage_dtype": dtype,
            "stored_shape": list(X_to_save.shape),
            "stored_label_shape": list(y_to_save.shape),
            "stored_class_0": int(np.sum(y_to_save == 0)),
            "stored_class_1": int(np.sum(y_to_save == 1)),
        }
    )

    metadata_path.write_text(
        json.dumps(
            complete_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return complete_metadata


def write_dataset_index(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write the index describing every generated file."""
    if not rows:
        raise ValueError("At least one generated dataset is required.")

    index = {
        "source": "synthetic",
        "description": (
            "Materialized calibrated synthetic benchmark for "
            "qsvm_medimage kernel and pipeline tests."
        ),
        "medical_semantics": False,
        "files": len(rows),
        "rows": rows,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            index,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic_qml_mimic_cxr_embeddings"),
    )

    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated synthetic model names. "
            "Valid names are: " + ", ".join(MODEL_LAYOUT) + "."
        ),
    )

    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.304,
    )

    parser.add_argument(
        "--n-signal-latents",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n-nuisance-latents",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--signal-strength",
        type=float,
        default=None,
        help=("Weight of label-related latent variables in the final embeddings."),
    )

    parser.add_argument(
        "--score-noise-std",
        type=float,
        default=None,
        help=("Standard deviation of noise added to the latent label score."),
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help=("Standard deviation of noise added to the final embedding vectors."),
    )

    parser.add_argument(
        "--nuisance-scale",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-distribution",
        choices=(
            "normal",
            "uniform",
            "rademacher",
            "skewed_uniform",
        ),
        default=None,
    )

    parser.add_argument(
        "--nuisance-skew-strength",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--nuisance-skew-decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dtype",
        choices=(
            "float32",
            "float64",
        ),
        default="float32",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Replace existing NPZ and metadata files."),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    models: list[str],
    seeds: list[int],
) -> None:
    """Validate command-line arguments."""
    if not models:
        raise ValueError("At least one model is required.")

    if not seeds:
        raise ValueError("At least one seed is required.")

    if args.n_samples is not None and args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")

    if args.embedding_dim is not None and args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive.")

    if not 0.0 < args.positive_ratio < 1.0:
        raise ValueError("--positive-ratio must be in the interval (0, 1).")

    if args.n_signal_latents is not None and args.n_signal_latents <= 0:
        raise ValueError("--n-signal-latents must be positive.")

    if args.n_nuisance_latents is not None and args.n_nuisance_latents < 0:
        raise ValueError("--n-nuisance-latents must be non-negative.")

    optional_non_negative_values = (
        (
            "--signal-strength",
            args.signal_strength,
        ),
        (
            "--score-noise-std",
            args.score_noise_std,
        ),
        (
            "--noise-std",
            args.noise_std,
        ),
        (
            "--nuisance-scale",
            args.nuisance_scale,
        ),
        (
            "--nuisance-skew-strength",
            args.nuisance_skew_strength,
        ),
    )

    for (
        option_name,
        option_value,
    ) in optional_non_negative_values:
        if option_value is not None and option_value < 0.0:
            raise ValueError(f"{option_name} must be non-negative.")

    optional_positive_values = (
        (
            "--nuisance-decay",
            args.nuisance_decay,
        ),
        (
            "--nuisance-skew-decay",
            args.nuisance_skew_decay,
        ),
    )

    for (
        option_name,
        option_value,
    ) in optional_positive_values:
        if option_value is not None and option_value <= 0.0:
            raise ValueError(f"{option_name} must be positive.")


def main() -> None:
    """Generate all requested model and seed combinations."""
    args = parse_args()

    models = parse_models(args.models)

    seeds = parse_ints(args.seeds)

    validate_args(
        args,
        models,
        seeds,
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    for model_name in models:
        for seed in seeds:
            metadata = write_one_dataset(
                output_root=args.output_root,
                model_name=model_name,
                seed=seed,
                n_samples=args.n_samples,
                embedding_dim=args.embedding_dim,
                positive_ratio=(args.positive_ratio),
                n_signal_latents=(args.n_signal_latents),
                n_nuisance_latents=(args.n_nuisance_latents),
                signal_strength=(args.signal_strength),
                score_noise_std=(args.score_noise_std),
                noise_std=args.noise_std,
                nuisance_scale=(args.nuisance_scale),
                nuisance_decay=(args.nuisance_decay),
                nuisance_distribution=(args.nuisance_distribution),
                nuisance_skew_strength=(args.nuisance_skew_strength),
                nuisance_skew_decay=(args.nuisance_skew_decay),
                dtype=args.dtype,
                overwrite=args.overwrite,
            )

            rows.append(metadata)

            print(
                "[synthetic-dataset] "
                f"model={model_name} "
                f"seed={seed} "
                f"shape={metadata['stored_shape']} "
                f"dtype={args.dtype}",
                flush=True,
            )

    index_path = args.output_root / "synthetic_dataset_index.json"

    write_dataset_index(
        index_path,
        rows,
    )

    summary = {
        "files": len(rows),
        "models": models,
        "seeds": seeds,
        "index": str(index_path),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
