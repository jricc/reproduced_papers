"""Data for the QSVM medical-embedding reproduction.

The paper's real inputs are frozen embeddings of MIMIC-CXR chest radiographs from three
medical foundation models, with a binary ``insurance == Private`` label.  Those embeddings
are gated (HuggingFace ``MITCriticalData/qml-mimic-cxr-embeddings``, 80 GB, inherits the
MIMIC PhysioNet data-use agreement) and could not be accessed for this reproduction.

We therefore provide:

1. ``load_real_embeddings`` — used automatically when ``QML_DATA_ROOT`` (a local copy of
   the gated dataset) is present.  This makes the reproduction *paper-accurate* for anyone
   with credentialed access, using the exact parquet layout of the original scripts.

2. ``make_synthetic_embeddings`` — a controlled substitute (labelled V4 synthetic-structural)
   that mimics the *statistical structure* the paper's claims rely on:
     * high ambient dimension (default 768, like ViT/RAD-DINO),
     * a low-dimensional latent manifold with a decaying spectrum (so PCA(q) is meaningful),
     * class imbalance matching the insurance task (~28% positive),
     * a tunable ``signal`` knob controlling how predictable the label is from the features.

   ``signal=0`` reproduces the regime the paper actually operates in: insurance type is
   essentially unpredictable from a chest X-ray, so the task carries almost no learnable
   signal under heavy imbalance.  Larger ``signal`` injects genuine, learnable structure
   for contrast / fair-baseline scrutiny.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def make_synthetic_embeddings(
    n_samples: int = 1000,
    ambient_dim: int = 768,
    latent_dim: int = 50,
    spectrum_decay: float = 0.9,
    minority_fraction: float = 0.28,
    signal: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
):
    """Generate (X, y) substitute foundation-model embeddings.

    Returns
    -------
    X : (n_samples, ambient_dim) float64
    y : (n_samples,) int  (1 = minority / "Private")
    """
    rng = np.random.default_rng(seed)

    # Latent factors with a decaying spectrum (foundation embeddings are low-rank-ish).
    scales = spectrum_decay ** np.arange(latent_dim)
    z = rng.standard_normal((n_samples, latent_dim)) * scales

    # Random near-orthogonal embedding into ambient space + small isotropic noise.
    proj = rng.standard_normal((latent_dim, ambient_dim)) / np.sqrt(latent_dim)
    X = z @ proj + 0.1 * rng.standard_normal((n_samples, ambient_dim))

    # Label from a fixed latent direction; ``signal`` sets the SNR.  At signal=0 the
    # label is independent of X (mimicking insurance type given a chest X-ray).
    w = rng.standard_normal(latent_dim)
    logits = signal * (z @ w) + noise * rng.standard_normal(n_samples)
    # Threshold so exactly ~minority_fraction samples are positive (class 1).
    thresh = np.quantile(logits, 1.0 - minority_fraction)
    y = (logits > thresh).astype(int)

    return X.astype(np.float64), y


# --- Real (gated) data path, mirroring the original repository scripts -----------------

_MODEL_SUBDIRS = {
    "medsiglip-448": "medsiglip-448-embeddings/20-seeds",
    "rad-dino": "rad-dino-embeddings/20-seeds",
    "vit-patch32-cls": "vit-base-patch32-224-embeddings/20-seeds",
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


def get_dataset(cfg: dict, seed: int):
    """Resolve (X, y) from config, preferring real gated data when available."""
    ds = cfg.get("dataset", {})
    source = ds.get("source", "synthetic")
    # Explicit gated-data root: dataset.data_root, then --data-root/top-level, then env var.
    data_root = ds.get("data_root") or cfg.get("data_root") or os.environ.get("QML_DATA_ROOT")
    # "auto" only switches to real data when QML_DATA_ROOT is explicitly exported, so the
    # runtime's default repo "data/" dir does not accidentally trigger the gated path.
    env_root = os.environ.get("QML_DATA_ROOT")

    if source == "real":
        return load_real_embeddings(ds.get("model", "medsiglip-448"), seed, data_root), "real"
    if source == "auto" and env_root:
        try:
            return load_real_embeddings(ds.get("model", "medsiglip-448"), seed, env_root), "real"
        except (FileNotFoundError, ImportError):
            pass  # fall through to synthetic substitute

    p = ds.get("synthetic", {})
    X, y = make_synthetic_embeddings(
        n_samples=p.get("n_samples", 1000),
        ambient_dim=p.get("ambient_dim", 768),
        latent_dim=p.get("latent_dim", 50),
        spectrum_decay=p.get("spectrum_decay", 0.9),
        minority_fraction=p.get("minority_fraction", 0.28),
        signal=p.get("signal", 0.0),
        noise=p.get("noise", 1.0),
        seed=seed,
    )
    return (X, y), "synthetic"
