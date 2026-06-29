# Quantum Kernel Advantage over Classical Collapse in Medical Foundation-Model Embeddings — Reproduction

Reproduction of **arXiv:2604.24597** (Cajas Ordóñez et al., 2026).
Original code: https://github.com/sebasmos/qml-medimage

> **Bottom line — partially reproduced 🟠 (V4 synthetic-structural).**
> Every *structural* phenomenon the paper reports is reproduced on substitute embeddings —
> the "classical collapse" of an untuned linear SVM, the non-collapsing QSVM, and a quantum
> kernel effective rank (**69.1**) almost identical to the paper's (**69.80**) at q=11.
> **But** a fair-baseline analysis the paper omits shows the headline "quantum advantage" is a
> **decision-threshold / baseline-fairness artifact, not genuine quantum discrimination**:
> wherever the QSVM beats the untuned linear SVM on minority-class F1, the two have the **same
> test AUC** (~0.62), and a trivial class-weighted linear SVM beats the QSVM on **both** F1 and AUC.

## Reference and Attribution
- Paper: *Quantum Kernel Advantage over Classical Collapse in Medical Foundation Model Embeddings*, arXiv:2604.24597.
- Original repo: `sebasmos/qml-medimage` (Qiskit + cuQuantum, GPU). Kernel code: `qve/core.py`.
- Task: binary `insurance == Private` classification on MIMIC-CXR chest radiographs, using frozen
  embeddings from three medical foundation models (MedSigLIP-448, RAD-DINO, ViT-patch32).

## Original Method
StandardScaler → PCA(q) → MinMaxScaler[−1,1] on the foundation-model embeddings, then a quantum
SVM with the **BSP fidelity kernel** (`make_bsp`): per qubit `H; Rz(x); Ry(x)`, a CNOT chain, then
`Rz(x)`; kernel `K(x,y)=|⟨0|U†(x)U(y)|0⟩|²`. Classifiers (all *untuned*, C=1):
- **Tier 1**: QSVM vs linear SVM (C=1). Claim: QSVM wins minority-F1 in all 18 configs; the linear
  kernel collapses to majority prediction on 90–100 % of seeds.
- **Tier 2**: QSVM vs C-tuned / effective-rank-matched RBF SVM. Claim: QSVM still wins (smaller gain, +0.068 mean).
- Eigenspectrum: quantum-kernel effective rank reaches 69.80 at q=11, far above the linear kernel;
  architecture-dependent concentration onset across qubit counts.

## Reproduction Scope, Claims, and Deviations
| | |
|---|---|
| Targeted | Tier-1 collapse (C1), non-collapsing QSVM (C2), eff-rank gap (C3), Tier-2 rank-matched RBF (C4), concentration vs q (C5), **plus** a fair-baseline adjudication the paper omits (C6) |
| Not targeted | Exact paper numbers (real data is gated — see Data), per-model breakdown, VQC/hybrid-kernel variants |
| **Key deviation** | **Real embeddings are gated and could not be accessed.** We reproduce the *kernel-method behaviour* on a controlled **synthetic foundation-embedding substitute** (V4). The BSP fidelity kernel itself is reproduced faithfully and validated against an independent dense-unitary build. |

### Data
The paper's real inputs (MIMIC-CXR images and the precomputed embeddings on HuggingFace
`MITCriticalData/qml-mimic-cxr-embeddings`) are **gated** — they inherit the MIMIC PhysioNet
data-use agreement (credentialed login + accepting terms) and total ~80 GB. Per the reproduction
policy we do not perform credentialed/human-mediated access → failure class **F1 (dataset inaccessible)**.

Substitute (`lib/data.py::make_synthetic_embeddings`, labelled **V4 synthetic-structural**): high
ambient dimension (768), a low-rank latent manifold with a decaying spectrum (so PCA(q) is
meaningful), class imbalance ≈0.28, and a **`signal` knob** controlling how predictable the label is.
`signal=0` reproduces the regime the paper actually operates in — *insurance type is essentially
unpredictable from a chest X-ray*, so the task carries almost no learnable signal under heavy imbalance.

> The paper's claims are explicitly about a **general** "classical collapse" phenomenon in foundation-model
> embeddings, so a controlled substitute is a fair test of the structural claims. Absolute numbers are
> not comparable to the paper; **qualitative behaviour and the eff-rank scale are**.

A paper-accurate path is retained for credentialed users: `configs/real_medsiglip.json` +
`export QML_DATA_ROOT=/path/to/qml-mimic-cxr-embeddings` (or `--data-root`).

## Install and How to Run
```bash
cd papers/qsvm_medimage
pip install -r requirements.txt          # numpy, scikit-learn, scipy, matplotlib (+pandas/pyarrow for gated path)

# from the repo root:
python implementation.py --paper qsvm_medimage --config configs/defaults.json          # ~2 s smoke
python implementation.py --paper qsvm_medimage --config configs/weak_signal_synth.json  # headline run (~5 min)
python implementation.py --paper qsvm_medimage --config configs/insurance_like_synth.json  # signal=0 (everything collapses)
python implementation.py --paper qsvm_medimage --config configs/photonic_synth.json     # MerLin photonic kernel (~1 min)

# decisive fairness sweep + figures:
python utils/signal_sweep.py --q 11 --signals 0,0.25,0.5,1,2,4 --out results/signal_sweep_q11.csv
python utils/plot_signal_sweep.py --csv results/signal_sweep_q11.csv
python utils/plot_summary.py --run-dir outdir/run_XXXX --highlight-q 11
```
No qiskit / cuQuantum / GPU needed: the BSP fidelity kernel is simulated on CPU with numpy
statevectors (`lib/quantum_kernel.py`); the photonic kernel uses MerLin's `FidelityKernel`.

## Configuration
One JSON config per experiment under `configs/`: `defaults.json` (smoke), `insurance_like_synth.json`
(signal=0), `weak_signal_synth.json` (signal=0.25, headline), `with_signal_synth.json` (signal=2,
learnable), `photonic_synth.json` (MerLin kernel), `real_medsiglip.json` (gated paper-accurate path).
CLI overrides: `--signal --n-samples --q-list --seeds --circuit {bsp,zz} --reps --source {synthetic,real,auto}`.

## Results Obtained and Comparison with the Paper
All reproduction numbers are on **substitute embeddings** (5 seeds, weak-signal regime `signal=0.25`,
which is where the paper's *simultaneous* "linear collapses / QSVM survives" phenomenon appears).

| Paper item | Claim tested | Paper value | Reproduced (substitute) | Label | Result |
|---|---|---|---:|---|---|
| Tier-1 collapse (C1) | untuned linear C=1 collapses | 90–100 % seeds | **100 % collapse, F1=0.000** @q=11 | V4, 5 seeds | reproduced |
| QSVM non-collapse (C2) | QSVM keeps minority F1 | F1=0.343 @q11 | F1=0.084 (>0; collapse 40 %) @q=11 | V4 | qualitative |
| Eff-rank gap (C3) | quantum ≫ linear | 69.80 @q11 | **QSVM 69.1 vs linear 10.8** @q=11 | V4 | quantitative match |
| Tier-2 (C4) | QSVM ≥ rank-matched RBF (F1) | +0.068 mean | QSVM 0.084 vs RBF-matched 0.050 (+0.034) | V4 | qualitative |
| Concentration (C5) | onset across q | q16 weighted-F1 crash | eff_rank 69→219 (q11→q16); QSVM F1 stays low | V4 | reproduced |
| **Fairness (C6, added)** | is it real advantage? | *not tested in paper* | **AUC parity (0.62≈0.62); fair linear wins F1 0.485 & AUC 0.663** | V4 | advantage refuted |

See `results/` (figures) and `outdir/run_*/summary.csv`.

### Why the "advantage" is an artifact (the central finding)
At q=11, weak signal (5 seeds):

| method | minority F1 | test AUC | eff_rank | collapse |
|---|---:|---:|---:|---:|
| linear C=1 (Tier-1 baseline) | 0.000 | 0.615 | 10.8 | 100 % |
| **QSVM (BSP, C=1)** | 0.084 | **0.620** | 69.1 | 40 % |
| RBF rank-matched (Tier-2) | 0.050 | 0.615 | 70.4 | 40 % |
| **linear balanced (fair, C=1)** | **0.485** | **0.663** | 10.8 | 0 % |
| linear tuned+balanced (fair) | 0.511 | 0.659 | 10.8 | 0 % |

- The QSVM and the untuned linear SVM have **essentially identical AUC** (0.62) — i.e. the *same*
  discrimination. The QSVM's nonzero minority-F1 is produced by *where its (concentrated) kernel places
  the decision threshold*, not by extracting more signal. Minority-F1 without an AUC gain is a
  thresholding artifact, magnified by heavy class imbalance on a near-unpredictable task.
- A **fair** classical baseline — a plain linear SVM with `class_weight='balanced'` — eliminates the
  collapse and **beats the QSVM on both F1 and AUC**, with ~7× lower effective rank. No quantum kernel
  is needed to "avoid collapse"; one line of class weighting suffices.
- `signal=0` (truly unpredictable, `insurance_like_synth`): *all* kernels collapse (F1=0, AUC≈0.50) and
  only class weighting produces nonzero F1 — at AUC≈0.50, i.e. a pure artifact.

## Fair Baselines
The paper's Tier-1 baseline (untuned linear C=1) is **unfair** for an imbalanced task: C=1 + standardized
high-dim features makes majority prediction near-optimal for hinge loss. Added fair baselines: `linear_balanced`
(class_weight='balanced', C=1) and `linear_tuned` (balanced, C chosen on validation minority-F1), on the
identical PCA-q features and splits. Both dominate the QSVM. Matching axis: accuracy/generalization on an
imbalanced task; key metric = **AUC** (threshold-independent), not raw minority-F1. → failure class **F6 (baseline unfairness)**.

## MerLin Photonic Extension
`lib/photonic_kernel.py` builds the photonic counterpart with MerLin's `FidelityKernel`: a 2-photon
boson-sampling fidelity kernel `K(x,y)=|⟨s|U†(x)U(y)|s⟩|²`, with `U(x)=W₂·diag(PS(πxᵢ))·W₁` on q modes
(Haar-random fixed meshes = photonic analogue of the BSP entangling layer). Run: `configs/photonic_synth.json`.

Result (weak signal, q=10, 3 seeds): the photonic kernel behaves like the gate QSVM — F1=0.117 (>linear
0.000), AUC=0.604, **eff_rank=125** (even more concentrated than the gate kernel's 55). It is again dominated
by the fair `linear_balanced` baseline (F1=0.424, AUC=0.645). **The photonic modality does not change the
verdict**: same threshold artifact, no AUC advantage.

### Hardware-Aware Settings (photonic)
| Field | Value |
|---|---|
| Computation space | Fock / no-bunching (SLOS) |
| Detector model | photon-number (analytic, shots=0) |
| Photon number | 2 |
| Number of modes | q (4–10 here) |
| Input state | alternating, e.g. `[1,0,1,0,…]` (2 photons) |
| Encoding | phase shifters PS(π·xᵢ), one per PCA feature, between two fixed Haar meshes |
| Measurement | FidelityKernel transition probability |
| Postselection | none |
| Simulator / QPU | MerLin SLOS CPU simulator |
| Seeds | 3 | Wall-clock | ~50 s |

## Limitations
- **Substitute data (V4)**: real gated MIMIC-CXR embeddings not accessed; absolute numbers differ from the
  paper. The synthetic generator is designed to match embedding *structure*, not the exact data distribution.
- The exact simultaneous gap (paper: linear 0.05 / QSVM 0.343) is larger than ours (0.000 / 0.084); on
  substitute data the F1 separation and the AUC parity are the robust, reproducible signatures.
- QSVM evaluated at C=1 (as in the paper); fidelity kernel simulated exactly (no shot noise / hardware noise).

## Tests
`cd papers/qsvm_medimage && pytest -q` — 10 tests: kernel matches an independent dense-unitary build,
kernel PSD/diag/symmetry, signal-knob separability, runner artifacts, CLI.

## Citation and License
Original paper: arXiv:2604.24597. Reproduction released under the repository MIT License.
See `LOG.md` (method, decisions), `INSIGHTS.md` (durable findings), `CONFLUENCE.md` (summary).
