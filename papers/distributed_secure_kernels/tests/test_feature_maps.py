from __future__ import annotations

import common  # noqa: F401
import numpy as np
from distributed_secure_kernels.lib.feature_maps import (
    fourier_feature_map,
    polynomial_feature_map,
    polynomial_offset,
    rbf_frequencies,
    rbf_kernel_exact,
)


def test_polynomial_map_inner_product_uses_both_offsets():
    rng = np.random.default_rng(0)
    x, y = 0.3 * rng.normal(size=3), 0.3 * rng.normal(size=3)
    inner = np.vdot(polynomial_feature_map(x, 1), polynomial_feature_map(y, 1))
    c_x, c_y = polynomial_offset(x, 1, 1.0), polynomial_offset(y, 1, 1.0)
    np.testing.assert_allclose(inner, x @ y + np.sqrt(c_x * c_y))
    assert not np.isclose(inner, x @ y + c_x)  # Theorem 2 with a single c fails


def test_polynomial_map_norm_is_not_one():
    x = np.array([0.6, 0.0, 0.0])
    squared_norm = np.vdot(
        polynomial_feature_map(x, 1), polynomial_feature_map(x, 1)
    ).real
    # Sum of squared amplitudes is a||x||^2 + c = 0.36 + 0.4, not 1 (Theorem 1).
    np.testing.assert_allclose(squared_norm, 0.76)


def test_rbf_map_is_normalised_and_approximates_the_kernel():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(1, 4)), rng.normal(size=(1, 4))
    w = rbf_frequencies(4, 50_000, 2.0, rng)
    psi_x, psi_y = fourier_feature_map(x, w), fourier_feature_map(y, w)
    np.testing.assert_allclose((psi_x**2).sum(), 1.0)
    assert abs((psi_x * psi_y).sum() - rbf_kernel_exact(x, y, 2.0)[0, 0]) < 0.02
