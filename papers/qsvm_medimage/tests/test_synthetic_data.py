"""Tests for the controlled synthetic fallback dataset."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

from lib.data import load_synthetic_npz_embeddings
from lib.synthetic_data import default_generation_profile, make_synthetic_embeddings

MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "synthetic_surrogate_table1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_surrogate_table1", MODULE_PATH)
table1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = table1
SPEC.loader.exec_module(table1)


def test_synthetic_generation_is_deterministic_for_same_seed():
    X1, y1, meta1 = make_synthetic_embeddings(n_samples=200, embedding_dim=64, seed=7)
    X2, y2, meta2 = make_synthetic_embeddings(n_samples=200, embedding_dim=64, seed=7)

    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)
    assert meta1 == meta2


def test_synthetic_generation_changes_with_seed():
    X1, y1, _ = make_synthetic_embeddings(n_samples=200, embedding_dim=64, seed=7)
    X2, y2, _ = make_synthetic_embeddings(n_samples=200, embedding_dim=64, seed=8)

    assert not np.allclose(X1, X2)
    assert not np.array_equal(y1, y2)


def test_synthetic_shape_ratio_labels_and_finiteness():
    X, y, meta = make_synthetic_embeddings(
        n_samples=1000,
        embedding_dim=128,
        positive_ratio=0.304,
        seed=0,
        model_name="synthetic_medsiglip",
    )

    assert X.shape == (1000, 128)
    assert y.shape == (1000,)
    assert set(np.unique(y)) == {0, 1}
    assert abs(float(np.mean(y)) - 0.304) <= 0.001
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert meta["source"] == "synthetic"
    assert meta["model_name"] == "synthetic_medsiglip"


def test_model_defaults_match_embedding_families():
    X_medsig, _, meta_medsig = make_synthetic_embeddings(
        n_samples=120,
        model_name="synthetic_medsiglip",
    )
    X_raddino, _, meta_raddino = make_synthetic_embeddings(
        n_samples=120,
        model_name="synthetic_raddino",
    )

    assert X_medsig.shape == (120, 1152)
    assert X_raddino.shape == (120, 768)
    assert meta_medsig["generator_version"] == "synthetic_kernel_geometry_v5_qsvm_separable"
    assert meta_raddino["model_family"] == "RAD-DINO"


def test_model_profiles_are_not_identical():
    medsig = default_generation_profile("synthetic_medsiglip")
    raddino = default_generation_profile("synthetic_raddino")
    vit = default_generation_profile("synthetic_vit")

    assert medsig["noise_std"] != raddino["noise_std"]
    assert medsig["nuisance_decay"] != vit["nuisance_decay"]
    assert raddino["n_nuisance_latents"] == 96
    assert vit["nuisance_distribution"] == "skewed_uniform"
    assert vit["nuisance_skew_strength"] > medsig["nuisance_skew_strength"]


def test_skewed_uniform_profile_is_finite_and_records_metadata():
    X, y, meta = make_synthetic_embeddings(
        n_samples=160,
        embedding_dim=64,
        model_name="synthetic_vit",
        seed=11,
    )

    assert X.shape == (160, 64)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == {0, 1}
    assert meta["nuisance_distribution"] == "skewed_uniform"
    assert meta["nuisance_skew_strength"] == 1.60


def test_labels_are_shared_across_synthetic_models_for_same_seed():
    _, y_medsig, _ = make_synthetic_embeddings(
        n_samples=200,
        model_name="synthetic_medsiglip",
        seed=3,
    )
    _, y_vit, _ = make_synthetic_embeddings(
        n_samples=200,
        model_name="synthetic_vit",
        seed=3,
    )

    assert np.array_equal(y_medsig, y_vit)


def test_materialized_synthetic_npz_loader_reads_expected_layout():
    X, y, _ = make_synthetic_embeddings(
        n_samples=20,
        embedding_dim=8,
        model_name="synthetic_medsiglip",
        seed=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        seed_dir = (
            Path(tmp)
            / "medsiglip-448-embeddings"
            / "20-seeds"
            / "seed_0"
        )
        seed_dir.mkdir(parents=True)
        np.savez_compressed(seed_dir / "data_type9_synthetic.npz", X=X, y=y)

        loaded_X, loaded_y = load_synthetic_npz_embeddings(
            "synthetic_medsiglip",
            seed=0,
            data_root=tmp,
        )

    assert np.allclose(loaded_X, X)
    assert np.array_equal(loaded_y, y)


def test_artifact_loader_reads_materialized_synthetic_source():
    X, y, _ = make_synthetic_embeddings(
        n_samples=20,
        embedding_dim=8,
        model_name="synthetic_medsiglip",
        seed=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        seed_dir = (
            Path(tmp)
            / "medsiglip-448-embeddings"
            / "20-seeds"
            / "seed_0"
        )
        seed_dir.mkdir(parents=True)
        np.savez_compressed(seed_dir / "data_type9_synthetic.npz", X=X, y=y)

        loaded_X, loaded_y = table1.load_dataset(
            source="synthetic_file",
            model="medsiglip-448",
            seed=0,
            data_root=Path(tmp),
            synthetic=table1.SyntheticSpec(
                n_samples=5,
                ambient_dim=4,
                latent_dim=2,
                minority_frac=0.2,
                signal=0.0,
                noise=1.0,
            ),
        )

    assert np.allclose(loaded_X, X)
    assert np.array_equal(loaded_y, y)
