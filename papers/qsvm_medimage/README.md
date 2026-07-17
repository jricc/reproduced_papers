# Quantum Kernel Advantage over Classical Collapse in Medical Foundation-Model Embeddings — Reproduction

Reproduction of **arXiv:2604.24597** (Cajas Ordóñez et al., 2026).
Original code: https://github.com/sebasmos/qml-medimage

> **Bottom line — partially reproduced 🟠 (synthetic substitute).**
> The real MIMIC-CXR embeddings are gated, so this is a reproduction of the paper's *protocol and
> structural phenomenology* on a clearly-labelled synthetic foundation-embedding substitute, not of
> its absolute numbers. The structural signatures reproduce: the untuned linear SVM collapses to
> majority prediction under class imbalance, the quantum fidelity kernel keeps a much higher effective
> rank that grows with the qubit count, and the QSVM recovers non-zero minority-class F1 where the
> linear kernel returns zero. A photonic (MerLin) variant of the kernel is provided and behaves
> comparably. Because the substitute is not the paper's data, the size of the QSVM's F1 margin over
> classical baselines is regime-dependent and is reported here as-measured rather than as a claim about
> the original medical task.

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
| Targeted | Tier-1 collapse (C1), non-collapsing QSVM (C2), eff-rank gap (C3), Tier-2 rank-matched RBF (C4), concentration vs q (C5); plus a photonic (MerLin) variant of the core artifacts |
| Not targeted | Exact paper numbers (real data is gated — see Data), reproduction of the medical claim itself |
| **Key deviation** | **Real embeddings are gated and could not be accessed.** We reproduce the *kernel-method behaviour* on a controlled **synthetic foundation-embedding substitute**. The BSP fidelity kernel itself is reproduced faithfully and validated against an independent dense-unitary build. |
| Additional baselines | Beyond the paper's Tier-1/Tier-2 comparators we also report a class-weighted linear SVM for context on this imbalanced substitute; see *Additional baselines* below. |

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

### Synthetic fallback benchmark

The synthetic dataset is not intended to simulate chest radiographs, patients, or insurance
metadata. It is a controlled high-dimensional embedding benchmark designed to stress-test
the same kernel geometry studied by Ordóñez et al.: after PCA compression, linear kernels
can become structurally low-rank and collapse to majority-class prediction under class
imbalance, while richer nonlinear or photonic fidelity kernels may preserve minority-class
signal.

This fallback is useful for CI, smoke tests, local development, and MerLin photonic-kernel
testing when the gated MIMIC-CXR embedding dataset is unavailable. Claims on the original
medical task require the real MIMIC-CXR embedding dataset.

The current generator (`synthetic_kernel_geometry_v4_table5_table6_calibrated`) mirrors the
experimental structure of the gated embedding files: `data_type9`-sized cohorts by default,
class imbalance near 30.4% positive, 20-seed-compatible deterministic generation, and
model-shaped embedding dimensions (`synthetic_medsiglip`: 1152, `synthetic_raddino`: 768,
`synthetic_vit`: 768). Labels are shared across synthetic model families for the same seed
and sample count; only the frozen-style embedding geometry changes by model. The default
profiles are calibrated against the paper's Table V PCA geometry and Table VI linear-kernel
moments, not against classifier performance.

Recommended local workflow keeps the three stages separate:

```bash
# From papers/qsvm_medimage.

# 1. Generate a materialized synthetic dataset once.
python utils/generate_synthetic_dataset.py \
  --output-root data/synthetic_qml_mimic_cxr_embeddings \
  --models synthetic_medsiglip,synthetic_raddino,synthetic_vit \
  --seeds 0,1,2,3,4,5,6,7,8,9

# 2. Train/evaluate models from the fixed dataset.
python ../../implementation.py \
  --paper qsvm_medimage \
  --config configs/synthetic_file_collapse.json

# 3. Render artifacts from saved training outputs, without retraining.
python utils/render_run_artifacts.py --run-dir outdir/run_XXXX

# Optional: paper-style diagnostics can also read the fixed dataset.
python utils/synthetic_surrogate_table10.py --source synthetic_file --data-root data/synthetic_qml_mimic_cxr_embeddings

# Or run selected table/figure scripts with the same fixed dataset source.
python utils/generate_artifacts_from_dataset.py \
  --source synthetic_file \
  --data-root data/synthetic_qml_mimic_cxr_embeddings \
  --only figure2,table6,table10

# Audit geometry before changing the synthetic generator.
python utils/audit_synthetic_geometry.py \
  --source synthetic_file \
  --data-root data/synthetic_qml_mimic_cxr_embeddings
```

The in-memory configs (`synthetic_smoke.json`, `synthetic_collapse.json`) remain useful for quick
checks. The `synthetic_file_*` configs are preferred when comparing runs, because the dataset is fixed
on disk. The `render_run_artifacts.py` script is the strict no-training artifact step; the
`synthetic_surrogate_*` scripts are paper-style diagnostics that may recompute the specific kernels or
models needed for a table or figure, but from the same fixed dataset.

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

## Reproducing Tables 1–10 and Figures 2–5
Every table and figure of the paper has a dedicated script under `utils/`
(`synthetic_surrogate_table{1..10}.py`, `synthetic_surrogate_figure{2..5}.py`),
each writing `.md`/`.json`/`.csv`/`.png` outputs to `results/`. Regenerate all
of them from one fixed dataset:

```bash
cd papers/qsvm_medimage
python utils/generate_synthetic_dataset.py \
  --output-root data/synthetic_qml_mimic_cxr_embeddings \
  --models synthetic_medsiglip,synthetic_raddino,synthetic_vit --seeds 0,1,2,3,4,5,6,7,8,9
python utils/generate_artifacts_from_dataset.py \
  --source synthetic_file --data-root data/synthetic_qml_mimic_cxr_embeddings \
  --results-dir results --only all      # or --only figure4,table5,table10
```

The full coverage map (paper item → script → protocol → output files), the
per-source output-prefix convention, and the credentialed `--source real` path
are documented in **[`REPRODUCTION_ARTIFACTS.md`](REPRODUCTION_ARTIFACTS.md)**.
As with the rest of this reproduction, on `--source synthetic{,_file}` these
scripts reproduce the paper's *protocol shape* on a labelled substitute, not its
absolute numbers.

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
| Concentration (C5) | onset across q | q16 weighted-F1 crash | eff_rank grows steeply (q11→q16); QSVM F1 stays low | V4 | reproduced |

The `synthetic_file` surrogate (v5 generator, on-disk dataset) is the current reference for Tables 1–10
and Figures 2–5; see `results/synthetic_file_*` and **[`REPRODUCTION_ARTIFACTS.md`](REPRODUCTION_ARTIFACTS.md)**
for per-artifact numbers. On that dataset the linear kernel collapses (minority F1 = 0), the quantum-kernel
effective rank grows from ≈11 at q=4 to several hundred at q≥11 (Table 5), and the QSVM's Tier-1 F1 margin
over the untuned linear baseline is positive on average but regime-dependent across (model, q). Figures live
in `results/`; per-seed rows are in `outdir/run_*/summary.csv`.

### Additional baselines (context on the substitute)
For context on this imbalanced substitute we also report a class-weighted linear SVM alongside the paper's
Tier-1/Tier-2 comparators. It is not part of the paper's protocol and is provided only to characterise the
substitute. At q=11, weak signal (5 seeds):

| method | minority F1 | test AUC | eff_rank | collapse |
|---|---:|---:|---:|---:|
| linear C=1 (Tier-1 baseline) | 0.000 | 0.615 | 10.8 | 100 % |
| QSVM (BSP, C=1) | 0.084 | 0.620 | 69.1 | 40 % |
| RBF rank-matched (Tier-2) | 0.050 | 0.615 | 70.4 | 40 % |
| linear balanced (C=1) | 0.485 | 0.663 | 10.8 | 0 % |
| linear tuned + balanced | 0.511 | 0.659 | 10.8 | 0 % |

On this near-unpredictable, heavily imbalanced substitute the untuned linear C=1 baseline collapses to
majority prediction, whereas the QSVM keeps non-zero minority F1 — the qualitative behaviour the paper
reports. A class-weighted linear SVM also avoids collapse here; on the substitute its threshold-independent
AUC is close to the QSVM's, so the F1 differences between methods are sensitive to the decision threshold
and to the class imbalance. These are properties of the synthetic substitute; the paper's claim is about the
real MIMIC-CXR embeddings, which were not accessible. Metrics reported: minority-class F1 (as in the paper)
and AUC (threshold-independent) for context.

## MerLin Photonic Extension
`lib/photonic_kernel.py` builds the photonic counterpart with MerLin's `FidelityKernel`: a 2-photon
boson-sampling fidelity kernel `K(x,y)=|⟨s|U†(x)U(y)|s⟩|²`, with `U(x)=W₂·diag(PS(πxᵢ))·W₁` on q modes
(Haar-random fixed meshes = photonic analogue of the BSP entangling layer). Run: `configs/photonic_synth.json`.

`utils/photonic_artifacts.py` generates photonic variants of the core artifacts
(`results/photonic_{table1,figure4,effrank}`), comparing the linear SVM, the qubit QSVM, and the photonic
QSVM on the same surrogate data (train/test subsampled for SLOS tractability). On the `synthetic_file`
dataset the photonic QSVM recovers non-zero minority F1 where the linear baseline collapses (beats linear
in 10/10 (model, q) configs, mean +0.28 F1), sits ≈0.09 F1 below the gate-based QSVM, and its effective rank
grows with q (≈15 → 112 for q=4 → 10). The photonic kernel therefore behaves comparably to the gate-based
QSVM: the same qualitative structure is reproduced in the linear-optical modality.

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
- The exact simultaneous gap (paper: linear 0.05 / QSVM 0.343) differs from ours; on substitute data the
  robust, reproducible signatures are the linear-kernel collapse and the quantum-kernel effective-rank growth.
- The QSVM's F1 margin over classical baselines is regime-dependent on the substitute and depends on class
  imbalance and decision threshold; it is reported as-measured, not as a claim about the real medical task.
- QSVM evaluated at C=1 (as in the paper); fidelity kernel simulated exactly (no shot noise / hardware noise).

## Tests
`cd papers/qsvm_medimage && pytest -q` — 73 tests: kernel matches an independent dense-unitary build,
kernel PSD/diag/symmetry, signal-knob separability, runner artifacts, CLI, the `synthetic_surrogate_*`
table/figure helpers, and the photonic artifact driver.

## Citation and License
Original paper: arXiv:2604.24597. Reproduction released under the repository MIT License.
See `LOG.md` (method, decisions), `INSIGHTS.md` (durable findings), `CONFLUENCE.md` (summary).
