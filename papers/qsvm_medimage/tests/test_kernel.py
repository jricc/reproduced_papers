"""Validate the CPU statevector fidelity kernel against an independent dense build."""
from __future__ import annotations

import importlib

import numpy as np

qk = importlib.import_module("lib.quantum_kernel")

# Independent reference: build the BSP unitary as an explicit dense matrix via Kronecker
# products and a hand-written CNOT, then compute the fidelity.  This shares no code path
# with the tensordot simulator in lib.quantum_kernel.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _op_on(gate, q, n):
    # qubit 0 = least significant => leftmost in kron is qubit n-1
    return _kron_list([gate if (n - 1 - k) == q else I2 for k in range(n)])


def _cx(control, target, n):
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bits = [(b >> k) & 1 for k in range(n)]
        if bits[control]:
            bits[target] ^= 1
        nb = sum(bit << k for k, bit in enumerate(bits))
        U[nb, b] = 1.0
    return U


def _bsp_unitary(x):
    n = len(x)
    U = np.eye(2 ** n, dtype=complex)
    for q in range(n):
        U = _op_on(H, q, n) @ U
    for q in range(n):
        U = _op_on(_rz(x[q]), q, n) @ U
        U = _op_on(_ry(x[q]), q, n) @ U
    for i in range(n - 1):
        U = _cx(i, i + 1, n) @ U
    for q in range(n):
        U = _op_on(_rz(x[q]), q, n) @ U
    return U


def _ref_fidelity(x, y):
    n = len(x)
    z = np.zeros(2 ** n, dtype=complex)
    z[0] = 1.0
    psi_x = _bsp_unitary(x) @ z
    psi_y = _bsp_unitary(y) @ z
    return abs(np.vdot(psi_x, psi_y)) ** 2


def test_kernel_matches_reference_small():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        data = rng.uniform(-1, 1, size=(5, n))
        K = qk.fidelity_kernel(data, circuit="bsp")
        for i in range(5):
            for j in range(5):
                assert abs(K[i, j] - _ref_fidelity(data[i], data[j])) < 1e-9


def test_kernel_properties():
    rng = np.random.default_rng(1)
    data = rng.uniform(-1, 1, size=(8, 4))
    K = qk.fidelity_kernel(data, circuit="bsp")
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-8  # PSD up to fp error


def test_effective_rank_bounds():
    rng = np.random.default_rng(2)
    K = qk.fidelity_kernel(rng.uniform(-1, 1, size=(20, 5)), circuit="bsp")
    er = qk.effective_rank(K)
    assert 1.0 <= er <= 20.0
