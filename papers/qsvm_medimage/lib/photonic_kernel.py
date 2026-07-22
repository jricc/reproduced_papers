"""Photonic fidelity kernel implemented with MerLin and Perceval.

This module defines a simple native photonic feature map. It is not a
gate-by-gate translation of the qubit BSP circuit.

For q PCA features, the circuit contains:

1. One fixed q-mode interferometer.
2. One feature-dependent phase shifter per mode.

The transformation is:

    U(x) = D(x) W

where W is a fixed Haar-random interferometer and:

    D(x) = diag(
        exp(i * phase_scale * x[0]),
        ...,
        exp(i * phase_scale * x[q - 1])
    )

The fidelity kernel is:

    K(x, y) = |<s|U(x).dagger U(y)|s>|^2

where |s> is a fixed Fock input state.

Two photons are used by default to include multi-photon interference. This
does not imply a computational or predictive advantage.
"""

from __future__ import annotations

import numpy as np


def _haar_unitary(
    n_modes: int,
    seed: int,
):
    """Generate one deterministic Haar-random unitary matrix."""
    import perceval as pcvl

    random_generator = np.random.default_rng(seed)

    random_matrix = (
        random_generator.standard_normal((n_modes, n_modes))
        + 1j * random_generator.standard_normal((n_modes, n_modes))
    ) / np.sqrt(2.0)

    unitary, triangular = np.linalg.qr(random_matrix)

    diagonal = np.diag(triangular)

    magnitudes = np.abs(diagonal)

    phases = np.ones_like(
        diagonal,
        dtype=np.complex128,
    )

    nonzero = magnitudes > 0.0

    phases[nonzero] = diagonal[nonzero] / magnitudes[nonzero]

    return pcvl.Matrix(unitary * phases)


def build_photonic_circuit(
    q: int,
    seed: int = 0,
):
    """Build a fixed interferometer followed by q phase shifters.

    The interferometer distributes the input photons across the optical modes.

    Each PCA feature controls one phase parameter:

        mode 0 -> phi0
        mode 1 -> phi1
        ...
        mode q - 1 -> phi(q - 1)
    """
    import perceval as pcvl

    if q <= 0:
        raise ValueError("q must be positive.")

    circuit = pcvl.Circuit(q)

    # Fixed mode-mixing layer.
    circuit.add(
        0,
        pcvl.Unitary(
            _haar_unitary(
                q,
                seed,
            )
        ),
    )

    # Data-encoding layer.
    for mode in range(q):
        circuit.add(
            mode,
            pcvl.PS(pcvl.P(f"phi{mode}")),
        )

    return circuit


def default_input_state(
    q: int,
    n_photons: int = 2,
):
    """Place photons in distinct modes when possible.

    Examples:

        q=4, n_photons=2 -> [1, 0, 1, 0]
        q=2, n_photons=2 -> [1, 1]
        q=1, n_photons=2 -> [2]
    """
    if q <= 0:
        raise ValueError("q must be positive.")

    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")

    state = [0] * q
    placed_photons = 0

    # First use alternating modes: 0, 2, 4, ...
    for mode in range(0, q, 2):
        if placed_photons == n_photons:
            break

        state[mode] += 1
        placed_photons += 1

    # Then use remaining empty modes.
    for mode in range(q):
        if placed_photons == n_photons:
            break

        if state[mode] == 0:
            state[mode] = 1
            placed_photons += 1

    # If photons still remain, place them in the first mode.
    if placed_photons < n_photons:
        state[0] += n_photons - placed_photons

    return state


def _validate_features(
    features: np.ndarray,
    q: int,
    name: str,
) -> np.ndarray:
    """Validate a matrix of q-dimensional PCA features."""
    array = np.asarray(
        features,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features).")

    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    if array.shape[1] != q:
        raise ValueError(f"{name} has {array.shape[1]} features, but q={q}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array


def _to_numpy(
    kernel,
) -> np.ndarray:
    """Convert a MerLin result to a real float64 NumPy matrix."""
    if hasattr(kernel, "detach"):
        array = kernel.detach().cpu().numpy()
    else:
        array = np.asarray(kernel)

    array = np.real_if_close(
        array,
        tol=1000,
    )

    if np.iscomplexobj(array):
        maximum_imaginary_part = float(np.max(np.abs(array.imag)))

        raise ValueError(
            "Kernel has a non-negligible imaginary component: "
            f"{maximum_imaginary_part:.3e}."
        )

    array = np.asarray(
        array,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(array)):
        raise ValueError("Kernel contains non-finite values.")

    return array


def photonic_fidelity_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray,
    q: int,
    n_photons: int = 2,
    seed: int = 0,
    phase_scale: float = np.pi,
):
    """Compute photonic training and test fidelity kernels.

    PCA features are expected in [-1, 1]. With the default phase scale, the
    features are mapped to optical phases in [-pi, pi].

    Returns
    -------
    K_train
        Square training Gram matrix.

    K_test
        Rectangular test-versus-training Gram matrix.
    """
    import torch
    from merlin.algorithms import (
        FeatureMap,
        FidelityKernel,
    )

    if q <= 0:
        raise ValueError("q must be positive.")

    if n_photons <= 0:
        raise ValueError("n_photons must be positive.")

    if not np.isfinite(phase_scale):
        raise ValueError("phase_scale must be finite.")

    train_features = _validate_features(
        X_train,
        q,
        "X_train",
    )

    test_features = _validate_features(
        X_test,
        q,
        "X_test",
    )

    circuit = build_photonic_circuit(
        q,
        seed=seed,
    )

    feature_map = FeatureMap(
        circuit,
        input_size=q,
        input_parameters=["phi"],
    )

    input_state = default_input_state(
        q,
        n_photons,
    )

    fidelity_kernel = FidelityKernel(
        feature_map,
        input_state,
        n_photons=n_photons,
    )

    train_phases = torch.as_tensor(
        train_features * phase_scale,
        dtype=torch.float32,
    )

    test_phases = torch.as_tensor(
        test_features * phase_scale,
        dtype=torch.float32,
    )

    K_train = _to_numpy(fidelity_kernel.forward(train_phases))

    K_test = _to_numpy(
        fidelity_kernel.forward(
            test_phases,
            train_phases,
        )
    )

    expected_train_shape = (
        len(train_features),
        len(train_features),
    )

    expected_test_shape = (
        len(test_features),
        len(train_features),
    )

    if K_train.shape != expected_train_shape:
        raise RuntimeError(
            "Unexpected training-kernel shape "
            f"{K_train.shape}; expected "
            f"{expected_train_shape}."
        )

    if K_test.shape != expected_test_shape:
        raise RuntimeError(
            "Unexpected test-kernel shape "
            f"{K_test.shape}; expected "
            f"{expected_test_shape}."
        )

    # Remove small numerical asymmetries.
    K_train = 0.5 * (K_train + K_train.T)

    diagonal_error = float(np.max(np.abs(np.diag(K_train) - 1.0)))

    # MerLin evaluates the feature map in float32, so fidelities of identical
    # samples can differ slightly from 1 because of numerical precision.
    diagonal_tolerance = 1e-4

    if diagonal_error > diagonal_tolerance:
        raise RuntimeError(
            "Photonic fidelity kernel does not have a unit diagonal within "
            f"tolerance {diagonal_tolerance:.1e}. "
            f"Maximum error: {diagonal_error:.3e}."
        )

    # Restore the exact mathematical value after validating the numerical error.
    np.fill_diagonal(K_train, 1.0)

    eigenvalues = np.linalg.eigvalsh(K_train)
    minimum_eigenvalue = float(eigenvalues.min())

    # MerLin evaluates the feature map in float32. Small negative eigenvalues can
    # appear from numerical rounding even though a fidelity Gram matrix is
    # positive semidefinite in exact arithmetic.
    psd_tolerance = 1e-4

    if minimum_eigenvalue < -psd_tolerance:
        raise RuntimeError(
            "Photonic training kernel is not positive semidefinite within "
            f"tolerance {psd_tolerance:.1e}. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}."
        )

    return K_train, K_test
