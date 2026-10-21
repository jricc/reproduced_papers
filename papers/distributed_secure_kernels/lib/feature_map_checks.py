"""Numerical checks of the paper's feature-map theorems (Sec. 4, Thms 1-6).

Polynomial map (Eq. 3): the paper states ``||psi(x)|| = 1`` (Thm 1) and
``<psi(x)|psi(y)> = (a x.y + c)^d`` (Thm 2) for one constant ``c``, but defines
``c`` from ``||x||``. The check reports the actual norm and compares the
inner product with ``(a x.y + c_x)^d``, ``(a x.y + c_y)^d`` and
``(a x.y + sqrt(c_x) sqrt(c_y))^d``, which is what the multinomial expansion of
Eq. 3 gives when ``c`` differs between ``x`` and ``y``.

RFF maps (Eqs. 4 and 8): the inner product approximates the RBF / Laplacian
kernel in expectation (Thms 4 and 6); the swap test returns its square, which
approximates the same kernel with bandwidth ``sigma / sqrt(2)`` (RBF) or
``alpha / 2`` (Laplacian).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .feature_maps import (
    fourier_feature_map,
    laplacian_frequencies,
    laplacian_kernel_exact,
    polynomial_feature_map,
    polynomial_offset,
    rbf_frequencies,
    rbf_kernel_exact,
)

logger = logging.getLogger(__name__)


def _max_abs(values: list[complex]) -> float:
    array = np.asarray(values)
    finite = np.isfinite(array)
    return float(np.max(np.abs(array[finite]))) if finite.any() else float("nan")


def check_polynomial(
    settings: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Norms and inner products of Eq. 3 against the candidate closed forms."""

    scale = float(settings["scale"])
    results: dict[str, Any] = {}
    for degree in settings["degrees"]:
        for norm in settings["sample_norms"]:
            norms, err_cx, err_cy, err_geom, nan_count = [], [], [], [], 0
            for _ in range(int(settings["n_pairs"])):
                x = rng.normal(size=int(settings["n_features"]))
                y = rng.normal(size=int(settings["n_features"]))
                # Different norms for x and y (uniform in [0.5, 1.5] x norm), so
                # that c_x != c_y as for real data.
                norm_x, norm_y = norm * rng.uniform(0.5, 1.5, size=2)
                x, y = norm_x * x / np.linalg.norm(x), norm_y * y / np.linalg.norm(y)
                psi_x, psi_y = (
                    polynomial_feature_map(x, degree, scale),
                    polynomial_feature_map(y, degree, scale),
                )
                inner = np.vdot(psi_x, psi_y)
                if not np.isfinite(inner):
                    nan_count += 1
                    continue
                c_x, c_y = (
                    polynomial_offset(x, degree, scale),
                    polynomial_offset(y, degree, scale),
                )
                dot = scale * float(x @ y)
                norms.append(float(np.vdot(psi_x, psi_x).real))
                err_cx.append(inner - (dot + c_x) ** degree)
                err_cy.append(inner - (dot + c_y) ** degree)
                geometric = np.conj(complex(c_x) ** 0.5) * complex(c_y) ** 0.5
                err_geom.append(inner - (dot + geometric) ** degree)
            key = f"degree={degree},norm={norm}"
            results[key] = {
                "squared_norm_min": float(np.min(norms)) if norms else float("nan"),
                "sample_norm_range": [0.5 * norm, 1.5 * norm],
                "squared_norm_max": float(np.max(norms)) if norms else float("nan"),
                "max_error_vs_offset_of_x": _max_abs(err_cx),
                "max_error_vs_offset_of_y": _max_abs(err_cy),
                "max_error_vs_geometric_offset": _max_abs(err_geom),
                "non_finite_pairs": nan_count,
            }
            logger.info(
                "EVALUATION_COMPLETED | check=polynomial | %s | %s", key, results[key]
            )
    return results


def check_fourier(settings: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Mean absolute errors of the RBF and Laplacian RFF maps versus D."""

    n_features = int(settings["n_features"])
    left = rng.normal(size=(int(settings["n_pairs"]), n_features))
    right = rng.normal(size=(int(settings["n_pairs"]), n_features))
    sigma, alpha = float(settings["sigma"]), float(settings["alpha"])
    exact_rbf = np.diag(rbf_kernel_exact(left, right, sigma))
    exact_lap = np.diag(laplacian_kernel_exact(left, right, alpha))
    results: dict[str, Any] = {}
    for n_random in settings["n_random_features"]:
        w = rbf_frequencies(n_features, int(n_random), sigma, rng)
        inner_rbf = (fourier_feature_map(left, w) * fourier_feature_map(right, w)).sum(
            1
        )
        w_lap, phases = laplacian_frequencies(n_features, int(n_random), alpha, rng)
        inner_lap = (
            fourier_feature_map(left, w_lap, phases)
            * fourier_feature_map(right, w_lap, phases)
        ).sum(1)
        results[str(n_random)] = {
            "qubits": int(np.ceil(np.log2(2 * int(n_random)))),
            "rbf_inner_product_mae": float(np.abs(inner_rbf - exact_rbf).mean()),
            "rbf_swap_test_value_mae_vs_rbf": float(
                np.abs(inner_rbf**2 - exact_rbf).mean()
            ),
            "rbf_swap_test_value_mae_vs_rbf_sigma_over_sqrt2": float(
                np.abs(inner_rbf**2 - exact_rbf**2).mean()
            ),
            "laplacian_inner_product_mae": float(np.abs(inner_lap - exact_lap).mean()),
            "laplacian_swap_test_value_mae_vs_laplacian": float(
                np.abs(inner_lap**2 - exact_lap).mean()
            ),
            "laplacian_swap_test_value_mae_vs_laplacian_alpha_over_2": float(
                np.abs(inner_lap**2 - exact_lap**2).mean()
            ),
            "squared_norm": float((fourier_feature_map(left[:1], w) ** 2).sum()),
        }
        logger.info(
            "EVALUATION_COMPLETED | check=fourier | D=%s | %s",
            n_random,
            results[str(n_random)],
        )
    return results


def check_feature_maps(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run both checks with the generator seeded from ``cfg["seed"]``."""

    experiment = cfg["experiment"]
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 2]))
    logger.info(
        "TRAINING_STARTED | checks=polynomial,fourier | no training (closed-form maps)"
    )
    polynomial = check_polynomial(experiment["polynomial"], rng)
    logger.info(
        "TRAIN_CHECK_COMPLETED | check=polynomial | settings=%d", len(polynomial)
    )
    fourier = check_fourier(experiment["fourier"], rng)
    logger.info("TRAIN_CHECK_COMPLETED | check=fourier | settings=%d", len(fourier))
    return {"polynomial": polynomial, "fourier": fourier}
