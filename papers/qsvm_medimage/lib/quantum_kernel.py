"""Faithful CPU reimplementation of the QSVM fidelity kernel from arXiv:2604.24597.

The original repository (github.com/sebasmos/qml-medimage, ``qve/core.py``) builds a
Qiskit ``QuantumCircuit`` and contracts it with cuQuantum on GPU.  The default feature
map is ``make_bsp``:

    H on all qubits;  Rz(x_q) then Ry(x_q) on each qubit;
    CNOT chain (i, i+1);  Rz(x_q) on each qubit.

The quantum kernel is the standard *fidelity* kernel

    K(x, y) = |<0| U(x)^dagger U(y) |0>|^2 = |<psi(x)|psi(y)>|^2 ,  |psi(x)> = U(x)|0>.

This module computes the same kernel on CPU by explicit statevector simulation.  Because
the fidelity is invariant to global phase and to a consistent qubit ordering, this
reproduces the original kernel exactly (up to floating-point error); no GPU / cuQuantum /
qiskit is required.  Verified against an independent dense unitary build in the tests.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Single-qubit gates (Qiskit conventions; global phase irrelevant for fidelity)
# ---------------------------------------------------------------------------
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
                    dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_1q(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a 1-qubit ``gate`` to qubit ``q`` of an n-qubit statevector.

    Qubit 0 is the least-significant index (Qiskit ordering): axis ``n-1-q`` after
    reshaping the length-2**n vector to shape (2,)*n.
    """
    st = state.reshape((2,) * n)
    st = np.tensordot(gate, st, axes=([1], [n - 1 - q]))
    st = np.moveaxis(st, 0, n - 1 - q)
    return st.reshape(-1)


def _apply_cx(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    st = state.reshape((2,) * n)
    ax_c, ax_t = n - 1 - control, n - 1 - target
    sl_c1 = [slice(None)] * n
    sl_c1[ax_c] = 1
    block = st[tuple(sl_c1)]
    block = np.flip(block, axis=(ax_t if ax_t < ax_c else ax_t - 1))
    st[tuple(sl_c1)] = block
    return st.reshape(-1)


def bsp_statevector(x: np.ndarray, reps: int = 1) -> np.ndarray:
    """Return |psi(x)> = U_BSP(x)|0...0> as a length-2**n complex vector."""
    n = len(x)
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    for _ in range(reps):
        for q in range(n):
            state = _apply_1q(state, _H, q, n)
        for q in range(n):
            state = _apply_1q(state, _rz(float(x[q])), q, n)
            state = _apply_1q(state, _ry(float(x[q])), q, n)
        for i in range(n - 1):
            state = _apply_cx(state, i, i + 1, n)
        for q in range(n):
            state = _apply_1q(state, _rz(float(x[q])), q, n)
    return state


def zz_statevector(x: np.ndarray, reps: int = 1) -> np.ndarray:
    """ZZFeatureMap (Havlicek 2019) statevector, for the ablation in the paper."""
    n = len(x)
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0
    for _ in range(reps):
        for q in range(n):
            state = _apply_1q(state, _H, q, n)
        for q in range(n):
            state = _apply_1q(state, _rz(float(x[q])), q, n)
        for i in range(n):
            for j in range(i + 1, n):
                state = _apply_cx(state, i, j, n)
                angle = (np.pi - float(x[i])) * (np.pi - float(x[j]))
                state = _apply_1q(state, _rz(angle), j, n)
                state = _apply_cx(state, i, j, n)
    return state


_FEATURE_MAPS = {"bsp": bsp_statevector, "zz": zz_statevector}


def feature_states(data: np.ndarray, circuit: str = "bsp", reps: int = 1) -> np.ndarray:
    """Stack of statevectors, shape (N, 2**n), one per row of ``data`` (N, n)."""
    fn = _FEATURE_MAPS[circuit]
    return np.stack([fn(row, reps=reps) for row in np.asarray(data, dtype=np.float64)])


def fidelity_kernel(data1: np.ndarray, data2: np.ndarray | None = None,
                    circuit: str = "bsp", reps: int = 1) -> np.ndarray:
    """Quantum fidelity kernel K[i, j] = |<psi(x1_i)|psi(x2_j)>|^2.

    If ``data2`` is None, returns the symmetric train kernel (exact 1.0 on the
    diagonal).  This matches ``qve.core.get_kernel_matrix`` / ``build_qsvm_qc``.
    """
    s1 = feature_states(data1, circuit=circuit, reps=reps)
    if data2 is None:
        gram = s1.conj() @ s1.T
        K = np.abs(gram) ** 2
        np.fill_diagonal(K, 1.0)
        return K
    s2 = feature_states(data2, circuit=circuit, reps=reps)
    return np.abs(s1.conj() @ s2.T) ** 2


def effective_rank(K: np.ndarray) -> float:
    """exp(von Neumann entropy) of the trace-normalised kernel (paper eq.).

    Matches ``rbf_rank_matched_multiseed.eff_rank``.
    """
    tr = np.trace(K)
    if tr <= 0:
        return 1.0
    eig = np.linalg.eigvalsh(K / tr)
    eig = np.maximum(eig, 0.0)
    s = eig.sum()
    if s <= 0:
        return 1.0
    p = eig / s
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))
