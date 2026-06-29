"""Photonic (boson-sampling) fidelity kernel via MerLin — the photonic translation.

The paper's QSVM uses a qubit/gate fidelity kernel.  Its photonic counterpart is a
*linear-optical* fidelity kernel: encode the q PCA features as phase shifts inside an
interferometer and estimate

    K(x, y) = |<s| U^dagger(x) U(y) |s>|^2

with at least two indistinguishable photons in the input Fock state ``s`` (so the kernel
genuinely exploits multi-photon interference, not a trivial single-photon linear map).

We build U(x) = W2 . diag(PS(pi * x_i)) . W1 on q modes, with W1, W2 fixed Haar-random
mixing unitaries (the photonic analogue of the BSP entangling layer).  MerLin's
``FidelityKernel`` computes the Gram matrix by SLOS simulation.
"""
from __future__ import annotations

import numpy as np


def build_photonic_circuit(q: int, seed: int = 0):
    import perceval as pcvl

    rng = np.random.default_rng(seed)

    def haar(m):
        z = (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))) / np.sqrt(2)
        Qm, R = np.linalg.qr(z)
        d = np.diag(R)
        return pcvl.Matrix(Qm * (d / np.abs(d)))

    c = pcvl.Circuit(q)
    c.add(0, pcvl.Unitary(haar(q)))
    for i in range(q):
        c.add(i, pcvl.PS(pcvl.P(f"phi{i}")))
    c.add(0, pcvl.Unitary(haar(q)))
    return c


def default_input_state(q: int, n_photons: int = 2):
    """Spread ``n_photons`` across the q modes (alternating), so trainable modes interfere."""
    state = [0] * q
    placed, i = 0, 0
    while placed < n_photons and i < q:
        state[i] = 1
        placed += 1
        i += 2 if i + 2 < q else 1
    if placed < n_photons:  # fall back: stack remaining on mode 0
        state[0] += n_photons - placed
    return state


def photonic_fidelity_kernels(X_tr, X_te, q, n_photons=2, seed=0, phase_scale=np.pi):
    """Return (K_train, K_test) photonic fidelity kernels for the given splits."""
    import torch
    from merlin.algorithms import FeatureMap, FidelityKernel

    circuit = build_photonic_circuit(q, seed=seed)
    fmap = FeatureMap(circuit, input_size=q, input_parameters=["phi"])
    kernel = FidelityKernel(fmap, default_input_state(q, n_photons), n_photons=n_photons)

    def _np(K):
        return K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)

    Xtr = np.asarray(X_tr) * phase_scale
    Xte = np.asarray(X_te) * phase_scale
    K_tr = _np(kernel.forward(torch.as_tensor(Xtr, dtype=torch.float32)))
    K_te = _np(kernel.forward(torch.as_tensor(Xte, dtype=torch.float32),
                              torch.as_tensor(Xtr, dtype=torch.float32)))
    np.fill_diagonal(K_tr, 1.0)
    return K_tr, K_te
