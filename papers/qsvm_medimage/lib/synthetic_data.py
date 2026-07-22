"""Controlled synthetic embeddings for kernel-geometry reproduction.

The generator in this module is not a medical data simulator. It creates
high-dimensional frozen-style embeddings with imbalanced binary labels so the
runner can exercise the same preprocessing and kernel geometry as the paper
without requiring gated MIMIC-CXR embeddings.
"""

from __future__ import annotations

import numpy as np

MODEL_EMBEDDING_DIMS = {
    "synthetic_medsiglip": 448,
    "synthetic_raddino": 768,
    "synthetic_vit": 768,
}

MODEL_SAMPLE_COUNTS = {
    "synthetic_medsiglip": 2371,
    "synthetic_raddino": 2370,
    "synthetic_vit": 2371,
}

MODEL_FAMILY = {
    "synthetic_medsiglip": "MedSigLIP-448",
    "synthetic_raddino": "RAD-DINO",
    "synthetic_vit": "ViT-Base-Patch32-224 CLS",
}

MODEL_SEED_OFFSETS = {
    # Labels come from the base seed. These offsets only change the embedding
    # projection/noise, mimicking different frozen encoders on the same samples.
    "synthetic_medsiglip": 101,
    "synthetic_raddino": 202,
    "synthetic_vit": 303,
}

MODEL_GEOMETRY_PROFILES = {
    # Calibrated against the paper Table V PCA variance percentages and Table
    # VI linear-kernel moments for q=4 and q=6.
    "synthetic_medsiglip": {
        "n_signal_latents": 8,
        "n_nuisance_latents": 160,
        "signal_strength": 0.7,
        "score_noise_std": 0.4,
        "noise_std": 0.40,
        "nuisance_scale": 3.0,
        "nuisance_decay": 0.91,
        "nuisance_distribution": "skewed_uniform",
        "nuisance_skew_strength": 0.25,
        "nuisance_skew_decay": 0.85,
    },
    "synthetic_raddino": {
        "n_signal_latents": 8,
        "n_nuisance_latents": 96,
        "signal_strength": 1.2,
        "score_noise_std": 0.4,
        "noise_std": 2.30,
        "nuisance_scale": 3.0,
        "nuisance_decay": 0.92,
        "nuisance_distribution": "skewed_uniform",
        "nuisance_skew_strength": 0.40,
        "nuisance_skew_decay": 0.65,
    },
    "synthetic_vit": {
        "n_signal_latents": 8,
        "n_nuisance_latents": 64,
        "signal_strength": 1.5,
        "score_noise_std": 0.4,
        "noise_std": 0.75,
        "nuisance_scale": 3.0,
        "nuisance_decay": 0.92,
        "nuisance_distribution": "skewed_uniform",
        "nuisance_skew_strength": 1.60,
        "nuisance_skew_decay": 0.55,
    },
}

GENERATOR_VERSION = "synthetic_kernel_geometry_v5_qsvm_separable"


def default_embedding_dim(model_name: str, fallback: int = 448) -> int:
    return MODEL_EMBEDDING_DIMS.get(model_name, fallback)


def default_n_samples(model_name: str, fallback: int = 2371) -> int:
    return MODEL_SAMPLE_COUNTS.get(model_name, fallback)


def default_generation_profile(model_name: str) -> dict[str, float | int | str]:
    model_name = _model_name(model_name)
    default_profile = {
        "n_signal_latents": 8,
        "n_nuisance_latents": 64,
        "signal_strength": 0.7,
        "score_noise_std": 0.4,
        "noise_std": 1.0,
        "nuisance_scale": 3.0,
        "nuisance_decay": 0.96,
        "nuisance_distribution": "normal",
        "nuisance_skew_strength": 0.0,
        "nuisance_skew_decay": 1.0,
    }
    return {**default_profile, **MODEL_GEOMETRY_PROFILES.get(model_name, {})}


def _model_name(model_name: str) -> str:
    aliases = {
        "medsiglip-448": "synthetic_medsiglip",
        "rad-dino": "synthetic_raddino",
        "vit-patch32-cls": "synthetic_vit",
        "vit-base-patch32-224": "synthetic_vit",
    }
    return aliases.get(model_name, model_name)


def _random_map(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Create a stable random projection with roughly unit output variance."""
    return rng.standard_normal((rows, cols)) / np.sqrt(rows)


def _sample_nuisance_latents(
    rng: np.random.Generator,
    shape: tuple[int, int],
    distribution: str,
    skew_strength: float,
    skew_decay: float,
) -> np.ndarray:
    """Sample nuisance factors with unit-scale variance."""
    distribution = distribution.lower()
    if distribution == "normal":
        return rng.standard_normal(shape)
    if distribution == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=shape)
    if distribution == "rademacher":
        return rng.choice([-1.0, 1.0], size=shape)
    if distribution == "skewed_uniform":
        base = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=shape)
        per_latent_skew = skew_strength * (skew_decay ** np.arange(shape[1]))
        nuisance = base + per_latent_skew[None, :] * (base * base - 1.0)
        nuisance -= np.mean(nuisance, axis=0, keepdims=True)
        std = np.std(nuisance, axis=0, keepdims=True)
        return nuisance / np.maximum(std, 1e-12)
    raise ValueError(f"unknown nuisance_distribution: {distribution}")


def make_synthetic_embeddings(
    n_samples: int | None = None,
    embedding_dim: int | None = None,
    positive_ratio: float = 0.304,
    n_signal_latents: int | None = None,
    n_nuisance_latents: int | None = None,
    signal_strength: float | None = None,
    score_noise_std: float | None = None,
    noise_std: float | None = None,
    nuisance_scale: float | None = None,
    nuisance_decay: float | None = None,
    nuisance_distribution: str | None = None,
    nuisance_skew_strength: float | None = None,
    nuisance_skew_decay: float | None = None,
    seed: int = 0,
    model_name: str = "synthetic_medsiglip",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create deterministic synthetic embeddings, labels, and metadata.

    Labels are generated from a weak nonlinear score in signal latents. The
    embedding is dominated by nuisance variance, so small-q PCA initially
    captures mostly non-label directions.
    """
    model_name = _model_name(model_name)
    profile = default_generation_profile(model_name)
    if n_samples is None:
        n_samples = default_n_samples(model_name)
    if embedding_dim is None:
        embedding_dim = default_embedding_dim(model_name, fallback=448)
    if n_signal_latents is None:
        n_signal_latents = int(profile["n_signal_latents"])
    if n_nuisance_latents is None:
        n_nuisance_latents = int(profile["n_nuisance_latents"])
    if signal_strength is None:
        signal_strength = float(profile["signal_strength"])
    if score_noise_std is None:
        score_noise_std = float(profile.get("score_noise_std", 1.0))
    if noise_std is None:
        noise_std = float(profile["noise_std"])
    if nuisance_scale is None:
        nuisance_scale = float(profile["nuisance_scale"])
    if nuisance_decay is None:
        nuisance_decay = float(profile["nuisance_decay"])
    if nuisance_distribution is None:
        nuisance_distribution = str(profile["nuisance_distribution"])
    if nuisance_skew_strength is None:
        nuisance_skew_strength = float(profile["nuisance_skew_strength"])
    if nuisance_skew_decay is None:
        nuisance_skew_decay = float(profile["nuisance_skew_decay"])

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("positive_ratio must be in (0, 1)")
    if n_signal_latents < 4:
        raise ValueError("n_signal_latents must be at least 4")
    if n_nuisance_latents <= 0:
        raise ValueError("n_nuisance_latents must be positive")
    if not 0.0 < nuisance_decay <= 1.0:
        raise ValueError("nuisance_decay must be in (0, 1]")
    if nuisance_skew_strength < 0.0:
        raise ValueError("nuisance_skew_strength must be non-negative")
    if not 0.0 < nuisance_skew_decay <= 1.0:
        raise ValueError("nuisance_skew_decay must be in (0, 1]")

    label_rng = np.random.default_rng(seed)
    model_rng = np.random.default_rng(seed + MODEL_SEED_OFFSETS.get(model_name, 0))

    signal_latents = label_rng.standard_normal((n_samples, n_signal_latents))

    # Nonlinear (pairwise-product) label score. This region is linearly
    # inseparable, so a linear SVM at C=1 collapses to majority prediction on the
    # imbalanced task, while the nonlinear quantum fidelity kernel recovers the
    # minority class. ``score_noise_std`` controls how learnable the label is and
    # thus the minority-F1 magnitude the QSVM reaches.
    score_noise = score_noise_std * label_rng.standard_normal(n_samples)
    score = (
        signal_latents[:, 0] * signal_latents[:, 1]
        + 0.6 * signal_latents[:, 2] * signal_latents[:, 3]
        + score_noise
    )
    n_positive = int(round(n_samples * positive_ratio))
    n_positive = min(max(n_positive, 1), n_samples - 1)
    positive_idx = np.argpartition(score, n_samples - n_positive)[
        n_samples - n_positive :
    ]
    y = np.zeros(n_samples, dtype=int)
    y[positive_idx] = 1

    nuisance_latents = _sample_nuisance_latents(
        label_rng,
        (n_samples, n_nuisance_latents),
        nuisance_distribution,
        nuisance_skew_strength,
        nuisance_skew_decay,
    )

    nuisance_scales = nuisance_scale * (nuisance_decay ** np.arange(n_nuisance_latents))
    nuisance_features = nuisance_latents * nuisance_scales
    signal_features = signal_latents * (0.9 ** np.arange(n_signal_latents))

    nuisance_map = _random_map(model_rng, n_nuisance_latents, embedding_dim)
    signal_map = _random_map(model_rng, n_signal_latents, embedding_dim)
    embedding_noise = noise_std * model_rng.standard_normal((n_samples, embedding_dim))
    X = (
        nuisance_features @ nuisance_map
        + signal_strength * (signal_features @ signal_map)
        + embedding_noise
    )

    metadata = {
        "source": "synthetic",
        "generator_version": GENERATOR_VERSION,
        "model_name": model_name,
        "model_family": MODEL_FAMILY.get(model_name, model_name),
        "data_type": 9,
        "n_samples": n_samples,
        "embedding_dim": embedding_dim,
        "positive_ratio": positive_ratio,
        "realized_positive_ratio": float(np.mean(y)),
        "seed": seed,
        "signal_strength": signal_strength,
        "noise_std": noise_std,
        "n_signal_latents": n_signal_latents,
        "n_nuisance_latents": n_nuisance_latents,
        "nuisance_scale": nuisance_scale,
        "nuisance_decay": nuisance_decay,
        "nuisance_distribution": nuisance_distribution,
        "nuisance_skew_strength": nuisance_skew_strength,
        "nuisance_skew_decay": nuisance_skew_decay,
        "score_noise_std": score_noise_std,
        "score_threshold": float(np.min(score[positive_idx])),
        "score_formula": "z0*z1 + 0.6*z2*z3 + eps",
        "label_semantics": "synthetic positive class; not medical or insurance metadata",
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
    }
    return X.astype(np.float64), y, metadata
