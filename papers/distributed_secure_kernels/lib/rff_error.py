"""Fig. 5 (App. B.4): relative Frobenius error of the RFF kernels versus D.

Follows the authors' ``other_kernels/kernel_approximation_error.py``:
100 standard-normal samples of dimension d, 5 repetitions (new samples each
time), D from 32 to 32,768. For a given D the frequencies are the same in every
repetition and for every bandwidth (the authors reseed NumPy with 42 inside the
feature map, and ``normal(0, 1/sigma)`` / ``cauchy / alpha`` are rescalings of
the same draws), so one projection ``X W0^T`` per D serves all bandwidths.

The Laplacian map's phases cancel in ``<psi(x)|psi(y)>``
(``cos(a+b)cos(c+b) + sin(a+b)sin(c+b) = cos(a-c)``), so they are omitted.

Computed with torch on ``cfg["device"]``; frequencies are generated in chunks of
``chunk`` rows so that ``D x d`` never sits in memory at d = 100,000.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _projection(
    samples: torch.Tensor, n_random: int, distribution: str, seed: int, chunk: int
) -> torch.Tensor:
    """``samples @ W0^T`` for standard-normal or standard-Cauchy ``W0`` of shape ``(D, d)``."""

    generator = torch.Generator(device=samples.device).manual_seed(seed)
    parts = []
    for start in range(0, n_random, chunk):
        rows = min(chunk, n_random - start)
        shape = (rows, samples.shape[1])
        if distribution == "normal":
            block = torch.randn(
                shape, generator=generator, device=samples.device, dtype=samples.dtype
            )
        else:
            uniform = torch.rand(
                shape, generator=generator, device=samples.device, dtype=samples.dtype
            )
            block = torch.tan(math.pi * (uniform - 0.5))  # standard Cauchy
        parts.append(samples @ block.T)
    return torch.cat(parts, dim=1)


def _approximate(projection: torch.Tensor, inverse_bandwidth: float) -> torch.Tensor:
    """``<psi(x)|psi(y)> = 1/D sum_j cos(w_j.(x - y))`` from the projection."""

    scaled = projection * inverse_bandwidth
    cos, sin = torch.cos(scaled), torch.sin(scaled)
    return (cos @ cos.T + sin @ sin.T) / projection.shape[1]


def _relative_error(exact: torch.Tensor, approx: torch.Tensor) -> float:
    return float(torch.linalg.norm(exact - approx) / torch.linalg.norm(exact))


def run_rff_error(cfg: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """Relative Frobenius errors for every (dimension, D, bandwidth, repetition).

    Parameters
    ----------
    cfg : dict
        Uses ``cfg["experiment"]`` keys ``dimensions``, ``n_samples``,
        ``n_random_features``, ``sigmas``, ``alphas``, ``repetitions``,
        ``frequency_seed`` and ``chunk``; ``cfg["seed"]`` seeds the samples.
    dtype : torch.dtype
        Floating-point precision.

    Returns
    -------
    dict
        ``{"rbf": {d: {sigma: {D: [errors]}}}, "laplacian": {...}}`` with string keys.
    """

    experiment = cfg["experiment"]
    device = torch.device(cfg["device"])
    results: dict[str, Any] = {"rbf": {}, "laplacian": {}}
    total = (
        len(experiment["dimensions"])
        * int(experiment["repetitions"])
        * len(experiment["n_random_features"])
    )
    done = 0
    for dimension in experiment["dimensions"]:
        rbf = {
            str(s): {str(D): [] for D in experiment["n_random_features"]}
            for s in experiment["sigmas"]
        }
        lap = {
            str(a): {str(D): [] for D in experiment["n_random_features"]}
            for a in experiment["alphas"]
        }
        for repetition in range(int(experiment["repetitions"])):
            generator = torch.Generator(device=device).manual_seed(
                int(cfg["seed"]) * 1000 + repetition
            )
            samples = torch.randn(
                (int(experiment["n_samples"]), int(dimension)),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            squared = torch.cdist(samples, samples) ** 2
            l1 = torch.cdist(samples, samples, p=1)
            for n_random in experiment["n_random_features"]:
                seed = int(experiment["frequency_seed"]) + int(n_random)
                normal = _projection(
                    samples, int(n_random), "normal", seed, int(experiment["chunk"])
                )
                cauchy = _projection(
                    samples, int(n_random), "cauchy", seed + 1, int(experiment["chunk"])
                )
                for sigma in experiment["sigmas"]:
                    exact = torch.exp(-squared / (2.0 * sigma**2))
                    rbf[str(sigma)][str(n_random)].append(
                        _relative_error(exact, _approximate(normal, 1.0 / sigma))
                    )
                for alpha in experiment["alphas"]:
                    exact = torch.exp(-l1 / alpha)
                    lap[str(alpha)][str(n_random)].append(
                        _relative_error(exact, _approximate(cauchy, 1.0 / alpha))
                    )
                done += 1
                logger.info(
                    "EVALUATION_COMPLETED | d=%d | repetition=%d | D=%d | %d/%d | rbf_error(sigma=%s)=%.4f | laplacian_error(alpha=%s)=%.4f",
                    dimension,
                    repetition,
                    n_random,
                    done,
                    total,
                    experiment["sigmas"][-1],
                    rbf[str(experiment["sigmas"][-1])][str(n_random)][-1],
                    experiment["alphas"][-1],
                    lap[str(experiment["alphas"][-1])][str(n_random)][-1],
                )
        results["rbf"][str(dimension)] = rbf
        results["laplacian"][str(dimension)] = lap
    return results
