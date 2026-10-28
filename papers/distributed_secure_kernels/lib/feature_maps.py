"""The paper's proposed quantum feature maps (Sec. 4), implemented as written.

- Polynomial (Eq. 3, Theorems 1-2): one amplitude per multi-index
  ``k = (k_1, ..., k_{N+1})`` with ``sum k = d``:
  ``sqrt(a) sqrt(d!) / sqrt(k_1! ... k_{N+1}!) x_1^{k_1} ... x_N^{k_N} sqrt(c)^{k_{N+1}}``,
  with ``c = 1 - a||x||`` for odd ``d`` and ``c = -1 - a||x||`` for even ``d``
  (the authors' ``polynomial_kernel_encoding.py`` reading of the paper's
  "d in N / d in 2N"). ``c`` depends on ``x``; the square root of a negative
  ``c`` is taken as a complex number. No renormalisation is applied, so the
  normalisation claim of Theorem 1 can be tested.
- RBF (Eq. 4, Theorems 3-4): ``1/sqrt(D) sum_j cos(w_j.x)|2j-2> + sin(w_j.x)|2j-1>``
  with ``w_j ~ N(0, sigma^-2 I)``.
- Laplacian (Eq. 8, Theorems 5-6): same with ``w_j ~ Cauchy(0, 1/alpha)`` and
  a phase ``alpha_j ~ U(0, 2 pi)`` added to every argument.

The paper's Theorems state the kernel as the inner product ``<psi(x)|psi(y)>``;
the authors' swap-test estimator ``2 Pr(0) - 1`` gives its squared modulus
(App. A), which the experiment also reports.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np


def multi_indices(length: int, degree: int) -> Iterator[tuple[int, ...]]:
    """All non-negative integer tuples of ``length`` summing to ``degree`` (authors' order)."""

    if length == 1:
        yield (degree,)
        return
    for first in range(degree + 1):
        for rest in multi_indices(length - 1, degree - first):
            yield (first, *rest)


def polynomial_offset(x: np.ndarray, degree: int, scale: float) -> float:
    """``c`` of Eq. 3, read as in the authors' code: odd or even degree."""

    norm = float(np.linalg.norm(x))
    return (1.0 - scale * norm) if degree % 2 else (-1.0 - scale * norm)


def polynomial_feature_map(
    x: np.ndarray, degree: int, scale: float = 1.0
) -> np.ndarray:
    """Eq. 3 amplitudes for one sample (complex, not renormalised).

    Parameters
    ----------
    x : numpy.ndarray
        Sample of length N.
    degree : int
        Polynomial degree d.
    scale : float
        The paper's ``a``. Default value is 1.0.

    Returns
    -------
    numpy.ndarray
        ``comb(N + d, d)`` complex amplitudes.
    """

    offset_root = complex(polynomial_offset(x, degree, scale)) ** 0.5
    amplitudes = []
    for k in multi_indices(len(x) + 1, degree):
        coefficient = math.sqrt(scale) * math.sqrt(math.factorial(degree))
        coefficient /= math.sqrt(math.prod(math.factorial(kl) for kl in k))
        monomial = complex(np.prod(np.asarray(x, dtype=float) ** np.asarray(k[:-1])))
        amplitudes.append(coefficient * monomial * offset_root ** k[-1])
    return np.asarray(amplitudes)


def rbf_frequencies(
    n_features: int, n_random: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """``w_j ~ N(0, sigma^-2 I)``, shape ``(D, N)``."""

    return rng.normal(0.0, 1.0 / sigma, size=(n_random, n_features))


def laplacian_frequencies(
    n_features: int, n_random: int, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``w_j ~ Cauchy(0, 1/alpha)`` (shape ``(D, N)``) and phases ``U(0, 2 pi)`` (shape ``(D,)``)."""

    frequencies = rng.standard_cauchy(size=(n_random, n_features)) / alpha
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_random)
    return frequencies, phases


def fourier_feature_map(
    features: np.ndarray, frequencies: np.ndarray, phases: np.ndarray | None = None
) -> np.ndarray:
    """Eq. 4 / Eq. 8 amplitudes for each row: interleaved cos/sin, scaled by ``1/sqrt(D)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, 2 D)``; amplitude ``2j`` is ``cos``, ``2j + 1`` is ``sin``.
        The state uses ``ceil(log2(2 D))`` qubits (zero padding beyond ``2 D``).
    """

    projection = np.atleast_2d(features) @ frequencies.T
    if phases is not None:
        projection = projection + phases
    n_random = frequencies.shape[0]
    amplitudes = np.empty((projection.shape[0], 2 * n_random))
    amplitudes[:, 0::2] = np.cos(projection)
    amplitudes[:, 1::2] = np.sin(projection)
    return amplitudes / math.sqrt(n_random)


def rbf_kernel_exact(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    """``exp(-||x - y||^2 / (2 sigma^2))``."""

    squared = ((left[:, None, :] - right[None, :, :]) ** 2).sum(-1)
    return np.exp(-squared / (2.0 * sigma**2))


def laplacian_kernel_exact(
    left: np.ndarray, right: np.ndarray, alpha: float
) -> np.ndarray:
    """``exp(-||x - y||_1 / alpha)``."""

    l1 = np.abs(left[:, None, :] - right[None, :, :]).sum(-1)
    return np.exp(-l1 / alpha)
