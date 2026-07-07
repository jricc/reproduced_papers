"""Data for the QSVM medical-embedding reproduction.

The paper's real inputs are frozen embeddings of MIMIC-CXR chest radiographs from three
medical foundation models, with a binary ``insurance == Private`` label.  Those embeddings
are gated (HuggingFace ``MITCriticalData/qml-mimic-cxr-embeddings``, 80 GB, inherits the
MIMIC PhysioNet data-use agreement) and could not be accessed for this reproduction.

We therefore provide:

1. ``load_real_embeddings`` — used automatically when ``QML_DATA_ROOT`` (a local copy of
   the gated dataset) is present.  This makes the reproduction *paper-accurate* for anyone
   with credentialed access, using the exact parquet layout of the original scripts.

2. ``make_synthetic_embeddings`` — a backwards-compatible wrapper around
   ``lib.synthetic_data.make_synthetic_embeddings``. The new generator creates a controlled
   synthetic benchmark for the kernel-collapse mechanism; it is not a medical data simulator.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .synthetic_data import (
    default_embedding_dim,
    default_n_samples,
    make_synthetic_embeddings as make_synthetic_embeddings_with_metadata,
)


def make_synthetic_embeddings(
    n_samples: int = 2371,
    ambient_dim: int = 768,
    latent_dim: int = 50,
    spectrum_decay: float = 0.9,
    minority_fraction: float = 0.304,
    signal: float = 0.25,
    noise: float = 1.0,
    seed: int = 0,
):
    """Backward-compatible synthetic generator returning only ``X, y``.

    Older reproduction scripts call this function directly with the V4 parameter
    names. New code should prefer ``lib.synthetic_data.make_synthetic_embeddings``,
    which returns metadata too.
    """
    del spectrum_decay  # Kept for old config compatibility.
    n_signal_latents = max(8, min(16, latent_dim // 4 if latent_dim else 8))
    n_nuisance_latents = max(latent_dim, 1)
    X, y, _ = make_synthetic_embeddings_with_metadata(
        n_samples=n_samples,
        embedding_dim=ambient_dim,
        positive_ratio=minority_fraction,
        n_signal_latents=n_signal_latents,
        n_nuisance_latents=n_nuisance_latents,
        signal_strength=signal,
        noise_std=noise,
        nuisance_scale=3.0,
        seed=seed,
        model_name="synthetic_legacy",
    )
    return X, y


# --- Real (gated) data path, mirroring the original repository scripts -----------------

_MODEL_SUBDIRS = {
    "medsiglip-448": "medsiglip-448-embeddings/20-seeds",
    "rad-dino": "rad-dino-embeddings/20-seeds",
    "vit-patch32-cls": "vit-base-patch32-224-embeddings/20-seeds",
}

_SYNTHETIC_MODEL_TO_REAL_LAYOUT = {
    "synthetic_medsiglip": "medsiglip-448",
    "synthetic_raddino": "rad-dino",
    "synthetic_vit": "vit-patch32-cls",
}


def load_real_embeddings(model: str, seed: int, data_root: str | None = None):
    """Load (X, y) from a local copy of the gated MIMIC-CXR embeddings.

    Layout and label match ``scripts/classical_svm_multiseed.py`` in the original repo.
    """
    import pandas as pd

    data_root = data_root or os.environ.get("QML_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError("QML_DATA_ROOT not set; gated embeddings unavailable.")
    base = os.path.join(data_root, _MODEL_SUBDIRS[model], f"seed_{seed}")
    files = sorted(glob.glob(os.path.join(base, "data_type9*.parquet")))
    if model == "vit-patch32-cls":
        files = [f for f in files if "cls" in f] or files
    else:
        files = [f for f in files if "cls" not in f and "gap" not in f] or files
    if not files:
        raise FileNotFoundError(f"No parquet found under {base}")
    df = pd.read_parquet(files[0])
    X = np.stack(df["embedding"].values).astype(np.float64)
    y = (df["new_insurance_type"] == "Private").astype(int).values
    return X, y


def load_synthetic_npz_embeddings(model_name: str, seed: int, data_root: str):
    """Load a materialized synthetic fallback dataset from ``.npz`` files."""
    model = _SYNTHETIC_MODEL_TO_REAL_LAYOUT.get(model_name, model_name)
    if model not in _MODEL_SUBDIRS:
        raise ValueError(f"unknown synthetic model layout: {model_name}")

    base = os.path.join(data_root, _MODEL_SUBDIRS[model], f"seed_{seed}")
    files = sorted(glob.glob(os.path.join(base, "data_type9*.npz")))
    if not files:
        raise FileNotFoundError(f"No synthetic npz found under {base}")

    with np.load(files[0], allow_pickle=False) as data:
        X = data["X"].astype(np.float64)
        y = data["y"].astype(int)
    return X, y


def _synthetic_config(ds: dict, seed: int) -> dict:
    """Normalize old and new synthetic config keys into generator arguments."""
    nested = ds.get("synthetic", {})
    direct = {
        key: ds[key]
        for key in (
            "model_name",
            "n_samples",
            "embedding_dim",
            "ambient_dim",
            "positive_ratio",
            "minority_fraction",
            "n_signal_latents",
            "n_nuisance_latents",
            "signal_strength",
            "signal",
            "noise_std",
            "noise",
            "nuisance_scale",
            "nuisance_decay",
            "nuisance_distribution",
            "nuisance_skew_strength",
            "nuisance_skew_decay",
        )
        if key in ds
    }
    p = {**nested, **direct}

    model_name = p.get("model_name") or ds.get("model_name") or ds.get("model") or "synthetic_medsiglip"
    if model_name == "medsiglip-448":
        model_name = "synthetic_medsiglip"
    elif model_name == "rad-dino":
        model_name = "synthetic_raddino"
    elif model_name == "vit-patch32-cls":
        model_name = "synthetic_vit"

    embedding_dim = p.get("embedding_dim", p.get("ambient_dim"))
    if embedding_dim is None:
        embedding_dim = default_embedding_dim(model_name)

    n_signal_latents = p.get("n_signal_latents")
    n_nuisance_latents = p.get("n_nuisance_latents", p.get("latent_dim"))
    signal_strength = p.get("signal_strength", p.get("signal"))
    noise_std = p.get("noise_std", p.get("noise"))
    nuisance_scale = p.get("nuisance_scale")
    nuisance_decay = p.get("nuisance_decay")
    nuisance_skew_strength = p.get("nuisance_skew_strength")
    nuisance_skew_decay = p.get("nuisance_skew_decay")

    return {
        "n_samples": int(p.get("n_samples", default_n_samples(model_name))),
        "embedding_dim": int(embedding_dim),
        "positive_ratio": float(p.get("positive_ratio", p.get("minority_fraction", 0.304))),
        "n_signal_latents": int(n_signal_latents) if n_signal_latents is not None else None,
        "n_nuisance_latents": int(n_nuisance_latents) if n_nuisance_latents is not None else None,
        "signal_strength": float(signal_strength) if signal_strength is not None else None,
        "noise_std": float(noise_std) if noise_std is not None else None,
        "nuisance_scale": float(nuisance_scale) if nuisance_scale is not None else None,
        "nuisance_decay": float(nuisance_decay) if nuisance_decay is not None else None,
        "nuisance_distribution": p.get("nuisance_distribution"),
        "nuisance_skew_strength": (
            float(nuisance_skew_strength) if nuisance_skew_strength is not None else None
        ),
        "nuisance_skew_decay": float(nuisance_skew_decay) if nuisance_skew_decay is not None else None,
        "seed": int(p.get("seed", seed)),
        "model_name": model_name,
    }


def get_dataset(cfg: dict, seed: int):
    """Resolve (X, y) from config, preferring real gated data when available."""
    ds = cfg.get("data") or cfg.get("dataset", {})
    source = ds.get("mode", ds.get("source", "synthetic"))
    # Explicit gated-data root: dataset.data_root, then --data-root/top-level, then env var.
    data_root = ds.get("data_root") or cfg.get("data_root") or os.environ.get("QML_DATA_ROOT")
    # "auto" only switches to real data when QML_DATA_ROOT is explicitly exported, so the
    # runtime's default repo "data/" dir does not accidentally trigger the gated path.
    env_root = os.environ.get("QML_DATA_ROOT")

    if source == "real":
        return load_real_embeddings(ds.get("model", "medsiglip-448"), seed, data_root), "real"
    if source in {"synthetic_file", "synthetic_npz", "generated_synthetic"}:
        if not data_root:
            raise FileNotFoundError("data_root is required for materialized synthetic data")
        model_name = ds.get("model_name") or ds.get("model") or "synthetic_medsiglip"
        return load_synthetic_npz_embeddings(model_name, seed, data_root), "synthetic_file"
    if source == "auto" and env_root:
        try:
            return load_real_embeddings(ds.get("model", "medsiglip-448"), seed, env_root), "real"
        except (FileNotFoundError, ImportError):
            pass  # fall through to synthetic substitute

    params = _synthetic_config(ds, seed)
    X, y, _ = make_synthetic_embeddings_with_metadata(**params)
    return (X, y), "synthetic"
