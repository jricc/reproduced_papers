"""Exact CPU implementation of the quantum fidelity kernel in the manuscript.

The manuscript feature map uses:

1. One Ry(x[d]) rotation on each qubit.
2. A CNOT ring connecting neighbouring qubits.
3. One or more repetitions of this complete block.

For q qubits, the entangling layer is:

    CNOT(0, 1)
    CNOT(1, 2)
    ...
    CNOT(q - 2, q - 1)
    CNOT(q - 1, 0)

The fidelity kernel is:

    K(x, y) = |<psi(x) | psi(y)>|^2

where:

    |psi(x)> = U(x)|0...0>

The statevectors are simulated exactly with NumPy. The simulation is noiseless
and does not model finite shots or hardware errors.
"""

from __future__ import annotations

import numpy as np


def _ry(theta: float) -> np.ndarray:
    """Return the single-qubit Ry rotation matrix."""
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)

    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.complex128,
    )


def _validate_feature_vector(
    x: np.ndarray,
    reps: int,
) -> np.ndarray:
    """Validate one input feature vector."""
    features = np.asarray(x, dtype=np.float64)

    if features.ndim != 1:
        raise ValueError("x must be a one-dimensional feature vector.")

    if features.size == 0:
        raise ValueError("x must contain at least one feature.")

    if not np.all(np.isfinite(features)):
        raise ValueError("x contains non-finite values.")

    if reps < 1:
        raise ValueError("reps must be at least 1.")

    return features


def _validate_dataset(
    data: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate a feature matrix with shape (n_samples, n_features)."""
    samples = np.asarray(data, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_samples, n_features)."
        )

    if samples.shape[0] == 0:
        raise ValueError(
            f"{name} must contain at least one sample."
        )

    if samples.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one feature."
        )

    if not np.all(np.isfinite(samples)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return samples


def _apply_single_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply a 2x2 gate to one qubit of a Qiskit-ordered statevector."""
    if gate.shape != (2, 2):
        raise ValueError(
            "A single-qubit gate must have shape (2, 2)."
        )

    if not 0 <= qubit < n_qubits:
        raise ValueError(
            "Qubit index is out of range."
        )

    tensor = state.reshape((2,) * n_qubits)
    qubit_axis = n_qubits - 1 - qubit

    tensor = np.tensordot(
        gate,
        tensor,
        axes=([1], [qubit_axis]),
    )

    tensor = np.moveaxis(
        tensor,
        0,
        qubit_axis,
    )

    return tensor.reshape(-1)


def _apply_cnot(
    state: np.ndarray,
    control: int,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    """Apply a CNOT gate to a Qiskit-ordered statevector."""
    if control == target:
        raise ValueError(
            "CNOT control and target must be different."
        )

    if not 0 <= control < n_qubits:
        raise ValueError(
            "CNOT control is out of range."
        )

    if not 0 <= target < n_qubits:
        raise ValueError(
            "CNOT target is out of range."
        )

    tensor = state.reshape((2,) * n_qubits).copy()

    control_axis = n_qubits - 1 - control
    target_axis = n_qubits - 1 - target

    control_one = [slice(None)] * n_qubits
    control_one[control_axis] = 1

    block = tensor[tuple(control_one)].copy()

    # The fixed control axis is removed from the sliced block.
    if target_axis < control_axis:
        target_axis_in_block = target_axis
    else:
        target_axis_in_block = target_axis - 1

    tensor[tuple(control_one)] = np.flip(
        block,
        axis=target_axis_in_block,
    )

    return tensor.reshape(-1)


def bsp_statevector(
    x: np.ndarray,
    reps: int = 1,
) -> np.ndarray:
    """Encode one sample with Ry rotations and a CNOT ring.

    One repetition applies:

        Ry(x[d]) on every qubit
        CNOT nearest-neighbour chain
        CNOT from the last qubit to the first

    The closing CNOT is omitted for a single-qubit input.
    """
    features = _validate_feature_vector(
        x,
        reps,
    )

    n_qubits = features.size

    state = np.zeros(
        2**n_qubits,
        dtype=np.complex128,
    )
    state[0] = 1.0

    for _ in range(reps):
        # Encode one PCA feature on each qubit.
        for qubit, feature in enumerate(features):
            state = _apply_single_qubit_gate(
                state,
                _ry(float(feature)),
                qubit,
                n_qubits,
            )

        # Connect neighbouring qubits.
        for control in range(n_qubits - 1):
            state = _apply_cnot(
                state,
                control,
                control + 1,
                n_qubits,
            )

        # Close the chain into a ring.
        if n_qubits > 1:
            state = _apply_cnot(
                state,
                n_qubits - 1,
                0,
                n_qubits,
            )

    return state


def feature_states(
    data: np.ndarray,
    circuit: str = "bsp",
    reps: int = 1,
) -> np.ndarray:
    """Encode all samples as statevectors.

    The output shape is:

        (n_samples, 2**n_features)

    The circuit argument is retained for compatibility with the experiment
    runner. Only the manuscript BSP feature map is supported.
    """
    samples = _validate_dataset(
        data,
        "data",
    )

    if circuit != "bsp":
        raise ValueError(
            "Only circuit='bsp' is supported by this manuscript "
            "implementation."
        )

    return np.stack(
        [
            bsp_statevector(
                sample,
                reps=reps,
            )
            for sample in samples
        ]
    )


def fidelity_kernel(
    data1: np.ndarray,
    data2: np.ndarray | None = None,
    circuit: str = "bsp",
    reps: int = 1,
) -> np.ndarray:
    """Compute a quantum fidelity-kernel matrix.

    If data2 is omitted, return the square training Gram matrix.

    If data2 is provided, return the rectangular matrix comparing every row
    of data1 with every row of data2. For SVM prediction, data1 is normally
    the test set and data2 is the training set.
    """
    samples1 = _validate_dataset(
        data1,
        "data1",
    )

    states1 = feature_states(
        samples1,
        circuit=circuit,
        reps=reps,
    )

    if data2 is None:
        overlaps = states1.conj() @ states1.T
        kernel = np.abs(overlaps) ** 2

        # Remove small numerical asymmetries.
        kernel = 0.5 * (
            kernel + kernel.T
        )

        diagonal_error = float(
            np.max(
                np.abs(
                    np.diag(kernel) - 1.0
                )
            )
        )

        if diagonal_error > 1e-10:
            raise RuntimeError(
                "Encoded states are not normalized. "
                f"Maximum diagonal error: {diagonal_error:.3e}."
            )

        np.fill_diagonal(
            kernel,
            1.0,
        )

        return np.asarray(
            kernel,
            dtype=np.float64,
        )

    samples2 = _validate_dataset(
        data2,
        "data2",
    )

    # if samples1.shape[1] != samples2.shaperaise ValueError(
    #         "data1 and data2 must have the same number of features."
    #     )

    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError(
            "data1 and data2 must have the same number of features."
        )

    states2 = feature_states(
        samples2,
        circuit=circuit,
        reps=reps,
    )

    overlaps = states1.conj() @ states2.T
    kernel = np.abs(overlaps) ** 2

    return np.asarray(
        kernel,
        dtype=np.float64,
    )


def effective_rank(
    kernel: np.ndarray,
    psd_tolerance: float = 1e-8,
) -> float:
    """Compute the Shannon effective rank of a square kernel.

    Parameters
    ----------
    kernel
        Square kernel matrix.

    psd_tolerance
        Maximum accepted magnitude for negative eigenvalues caused by
        numerical precision.

        Exact NumPy qubit kernels normally use the default 1e-8 tolerance.
        Float32 photonic kernels may require a larger tolerance such as 1e-4.

    Returns
    -------
    float
        Shannon effective rank.

    Notes
    -----
    For kernel eigenvalues lambda_i:

        p_i = lambda_i / sum_j lambda_j

    and:

        effective_rank = exp(-sum_i p_i * log(p_i))

    Small negative eigenvalues within ``psd_tolerance`` are clipped to zero.
    """
    matrix = np.asarray(
        kernel,
        dtype=np.float64,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(
            "Effective rank requires a square kernel matrix."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Kernel matrix must not be empty."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Kernel contains non-finite values."
        )

    if psd_tolerance < 0.0:
        raise ValueError(
            "psd_tolerance must be non-negative."
        )

    matrix = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    if minimum_eigenvalue < -psd_tolerance:
        raise ValueError(
            "Kernel is not positive semidefinite within tolerance. "
            f"Minimum eigenvalue: {minimum_eigenvalue:.3e}. "
            f"Tolerance: {psd_tolerance:.1e}."
        )

    # Clip only numerical negative eigenvalues accepted by the tolerance.
    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    eigenvalue_sum = float(
        np.sum(eigenvalues)
    )

    if eigenvalue_sum <= 0.0:
        raise ValueError(
            "Kernel has no positive eigenvalue mass."
        )

    probabilities = (
        eigenvalues / eigenvalue_sum
    )

    probabilities = probabilities[
        probabilities > 1e-15
    ]

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    return float(
        np.exp(entropy)
    )
