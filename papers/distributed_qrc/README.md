# Scalable Quantum Reservoir Computing over Distributed Quantum Architectures — Reproduction

Reproduction of the paper's distributed quantum reservoir computing (QRC) architectures for next-hour Greek power-load forecasting, extended with additional classical baselines and a photonic (MerLin) adaptation.

**Summary.**
- **The paper's main contribution is architectural.** A QRC model is split into small, independent quantum circuits whose measured results are combined classically. No entanglement or quantum communication is needed between devices.
- **Reproduction.** We reproduce the four architectures and the paper's reported trend against its classical reference. The ideal fully quantum cells match the paper closely: the median deviation is 2.6 %; 26/32 configurations are within 5 % and all 32 within 12 %.
- **Advantage.** The authors do not claim a computational advantage. On forecasting accuracy, our additional classical baselines indicate that, in our setting, the gain comes from the kernel readout, which can be computed classically. The same kernel applied directly to the 4-hour window, without any reservoir, performs on par with the quantum models.

**Where to start:** this README, then `notebook.ipynb` (scientific walkthrough, about 35 s on a laptop CPU), then `results/` (one image per paper table and figure, and the comparison tables).

## Reference and Attribution

- **Paper:** Scalable Quantum Reservoir Computing over Distributed Quantum Architectures (arXiv preprint, v1, 6 May 2026; under submission).
- **Authors:** Ioannis Liliopoulos, Georgios D. Varsamis, Konstantinos Rallis, Evangelos Tsipas, Ioannis G. Karafyllidis, Georgios Ch. Sirakoulis, Panagiotis Dimitrakis.
- **DOI/arXiv:** https://arxiv.org/abs/2605.04991
- **Original repository:** none listed; no original code is reused.
- **Attribution:** cite the arXiv preprint above. The dataset is public data from IPTO, the Independent Power Transmission Operator of Greece (ADMIE in Greek, portal admie.gr), as cited by the paper (see Data).

## Original Paper

The paper forecasts the next-hour Greek electricity load (IPTO, 35,064 hourly values) from a 4-hour sliding window. Each model has a fixed reservoir and a trained ridge readout:

```text
  4-hour window (x[t-3], x[t-2], x[t-1], x[t])
        |
        v
  Reservoir (fixed)  ── classical ESN (ReservoirPy), or a network of 4-qubit quantum neurons
        |
        |  measured expectation values = features r(t)
        v
  Readout (trained)  ── linear ridge, or "quantum ridge" (kernel ridge with a fidelity kernel)
        |
        v
  Forecast x̂[t+1]
```

- **Quantum neuron:** a 4-qubit circuit with RX encoding and an RX/CX/RZ ansatz (Fig. 2). It reads data lags and other neurons' outputs; the network is propagated 3 times.
- **Quantum ridge:** kernel ridge whose fidelity kernel uses H and P(2x) layers on 5 or 10 qubits (Fig. 3).
- **Simulation:** reservoirs and kernels are simulated ideally or with hardware-informed noise (ibm_brisbane, ibm_marrakesh, IonQ Aria-1; App. B).

**The four architectures (Table I).** Every arrow carries classical data; no quantum state is shared between components, so each branch can run on its own process, node or QPU:

```text
SRSR   x ──> Reservoir ──> Readout ──> ŷ

MRSR   x ─┬─> Reservoir 1 ─┐
          ├─> Reservoir 2 ─┼─> concatenate ──> Readout ──> ŷ
          └─> Reservoir k ─┘

SRMR   x ──> Reservoir ──> r(t) ─┬─> part 1 ──> Readout 1 ─┐
                                 ├─> part 2 ──> Readout 2 ─┼─> mean ──> ŷ
                                 └─> part m ──> Readout m ─┘

MRMR   x ─┬─> Reservoir 1 ──> Readout 1 ─┐
          ├─> Reservoir 2 ──> Readout 2 ─┼─> mean ──> ŷ
          └─> Reservoir k ──> Readout k ─┘
```

Each architecture has 8 configurations at 10–60 total neurons, each with 3 reservoir types × 3 readout types. Results are reported in Tables IV–VII (best model per architecture), VIII (best per total neurons) and C1–C12 (all cells). **Metrics:** MAE, RMSE and R² on min-max-normalized load, one run per cell.

**Claims:**
- **C1:** the quantum-enhanced variants beat the architecture-matched classical baseline, with average MAE reductions of 25–54 % and a peak of 78.8 %.
- **C2:** the quantum-kernel readout is the most consistent contributor.
- **C3:** distributed architectures match or beat SRSR.
- **C4:** fully quantum models beat hybrids for small reservoirs.
- **C5:** the gains persist under noise.
- **C6:** distributing the reservoir mitigates noise.
- **C7:** the best model is the 3×20 classical reservoirs + quantum kernel hybrid (MAE 0.0113).
- **C8:** the architectures suit multi-QPU deployment (qualitative).

## Reproduction Scope, Claims, and Deviations

**Reproduced:** all four architectures × 8 configurations × the 9-cell grid, on the full data, with the full training set for every kernel fit (n = 24,535), seeds 0, 1, 2 (20 seeds for the classical-baseline spread). Simulation only, with no QPU. Every evidence sweep ran with a plan declared before execution and pinned code.

**Implementation:**
- *Neurons:* exact numpy simulation of the 4-qubit neurons, folded into 16×16 functionals.
- *Kernel:* closed-form fidelity kernel. The feature map has no entangling gate, so the kernel factorizes over qubits.
- *Cross-checks:* both are checked against Qiskit (statevector) and Qiskit Aer (noisy density matrix) in `tests/`.
- *Noise model:* rebuilt from the App. B medians: 1q and 2q depolarizing channels after native gates and a symmetric readout flip (`lib/noise.py`; how each calibration value is used is shown in `TABLE_B1.png` and `TABLE_B2.png`).
- *Memory:* as implemented, the quantum reservoir carries no state between time steps (R7); it acts as a fixed random feature map of the 4-lag window, closer to a quantum extreme learning machine than to a recurrent reservoir. The paper does not specify this detail.

**Deviations and choices** where the paper is silent:

| Item | Paper | Choice here |
|---|---|---|
| Data window (R1) | 35,064 hourly values, dates not given | 2020-10-22 … 2024-10-21, identified from the Table II anchors |
| Normalization (R3) | Min-max "in each dataset" | Min-max fitted on train; per split available as a variant (`--normalization per_split`), headline unchanged |
| Ansatz angles (R5) | "Trainable w" | Fixed, uniform in [0, 2π), seeded (reservoir computing) |
| Observable (R6) | "Pauli Z" | Z⊗4, chosen by eval RMSE (0.021545 vs Z₀ 0.076412) |
| Wiring and update (R7) | k_i ∈ {0, 1, 2} recurrent inputs | Uniform sources, synchronous update, 3 passes, no state between time steps; sequential update is tied within 1 % |
| λ (R10) | Not given | 10-value grid, n-normalized, selected on eval RMSE |
| Distributed merge (R12) | Not given | Contiguous feature partition, mean of instance predictions. `sum` would double the prediction, and giving the full r(t) to every instance would make the linear SRMR cells identical to SRSR, which the paper's tables contradict |
| Classical reservoir (R14) | ReservoirPy, no hyperparameters | Library defaults = paper baseline B0; a tuned ESN is added as a fair baseline |
| Noise construction (R15) | Not given | Depolarizing + readout from the App. B medians; exact expectations, no shots |
| Best-cell selection (R16) | Best of 8 cells, chosen on test | Reported with the paper's rule and with selection on the evaluation split |

**Software and hardware named in the paper:**

| Named in the paper | What we did |
|---|---|
| Qiskit | Used only to cross-check our exact numpy simulation (`tests/`) |
| NVIDIA CUDA-Q | Not used. It would not change the ideal results, since the simulation is exact |
| IBM `ibm_brisbane`, `ibm_marrakesh`; IonQ Aria-1 | Noise profiles only, in the paper too (no QPU runs). Rebuilt from the App. B medians (R15) |
| ReservoirPy | Used (0.4.2) |

**Claim verdicts:**

| Claim | Outcome |
|---|---|
| C1 | Trend reproduced against the paper's classical reference (B0). Not confirmed against the additional classical baselines |
| C2 | Reproduced: the kernel readout is the consistent contributor. The kernel is computable classically in closed form, so this gain is not specific to quantum hardware |
| C3 | Partly confirmed: distributed architectures perform comparably to SRSR, but the best model is not always distributed |
| C4 | Fully quantum cells are stable across sizes; their margin over hybrids is within seed variance |
| C5 | Holds against the paper's reference under our noise model; not confirmed against the additional baselines |
| C6 | Not confirmed: every quantum neuron is the same 4-qubit circuit in all architectures, and with the same noise profile on every component, distribution brings no measurable benefit |
| C7 | Not reproduced (0.0146 vs 0.0113); on par with the additional classical baselines |
| C8 | Qualitative. Consistent with our implementation, in which each component runs independently; not tested on hardware |

## Project Layout

```text
papers/distributed_qrc/
├── README.md, INSIGHTS.md, notebook.ipynb, cli.json, requirements.txt
├── configs/
│   ├── defaults.json                 # smoke run and every shared setting
│   ├── e{1..4}_*_original.json       # paper grids (SRSR, MRSR, SRMR, MRMR)
│   ├── e{1..4}_*_merlin.json         # MerLin adaptation
│   ├── e{1..4}_*_classical.json, b1b3_raw_lags_classical.json, c6_*_noise.json   # fair baselines and controls
│   ├── e{3,4}_*_reduced.json         # fallback with subsampled kernels (not needed in the end)
│   └── diagnostics/                  # gap diagnostics, machine cross-checks, observable/wiring selection
├── lib/       # data, quantum_neuron, noise, reservoirs, kernels, readouts, architectures, metrics, runner,
│              # photonic_merlin (MerLin), baselines_classical (fair baselines), experiment_logging
├── utils/     # run_sweep.py (sweep coordinator), analysis, plotting and verification scripts
├── tests/     # 48 tests (Qiskit, Qiskit Aer and Perceval ground truth, data, solvers, end-to-end runs)
├── results/
│   ├── results_summary.md, fair_comparison_table.md   # comparison tables (generated)
│   ├── fair_comparison.png, headline_reductions.png, merlin_comparison.png
│   ├── paper_items/        # one image per paper table and figure, paper next to ours
│   ├── data/               # machine-readable results (JSON, CSV) used by the notebook and the tables
│   └── checks/             # verification outputs: logging validation, sweep re-verification, cross-checks
└── outdir/    # raw runs and sweeps (git-ignored)
```

## Install and How to Run

From the repository root, in the shared repository venv:

```bash
.venv/bin/python -m pip install -r papers/distributed_qrc/requirements.txt
```

**Smoke run** (`configs/defaults.json`: SRSR 10 neurons, 9 cells, kernels on 4,000 points; about 20 s on a laptop CPU):

```bash
python implementation.py --paper distributed_qrc
```

**Paper grids** (Tables IV–VII, C1–C12), one seed per run. E1/E2 take 1–2 h per seed on a laptop CPU; E3/E4 are intended for a GPU:

```bash
python implementation.py --paper distributed_qrc --config configs/e1_srsr_original.json --seed 0
python implementation.py --paper distributed_qrc --config configs/e3_srmr_original.json --seed 0 --device cuda:0
```

**Sweeps** (seeds 0–2, written to `outdir/sweep_<ID>/`). The evidence sweeps of the reproduction carry the suffix `-R2` (second, fully pre-planned run):

```bash
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-R2 --config configs/e1_srsr_original.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-CLASSICAL --config configs/e1_srsr_classical.json --seeds 0 1 2
```

The same pattern applies to `E1-MERLIN` (`*_merlin.json`), `B1B3-RAW` (`b1b3_raw_lags_classical.json`) and `C6-SRSR` (`c6_srsr_noise.json`, with `--candidates '[{"noise_mapping": "paper", "noise_scale": 10}, {"noise_mapping": "ibm_brisbane", "noise_scale": 10}]'`).

**Analysis** (reads the raw `metrics.json` files, writes to `results/`):

```bash
python papers/distributed_qrc/utils/compare_with_paper.py papers/distributed_qrc/outdir/sweep_E1-R2
python papers/distributed_qrc/utils/headline_statistics.py papers/distributed_qrc/outdir/sweep_E{1,2,3,4}-R2
python papers/distributed_qrc/utils/fair_comparison.py papers/distributed_qrc/outdir
python papers/distributed_qrc/utils/fair_comparison_table.py
python papers/distributed_qrc/utils/documentation_tables.py papers/distributed_qrc/outdir
python papers/distributed_qrc/utils/plot_paper_items.py papers/distributed_qrc/outdir data/distributed_qrc/sources/arxiv_source_v1/tqe_tran.tex
```

**Notebook:**

```bash
python papers/distributed_qrc/utils/build_notebook.py   # regenerate from source
python papers/distributed_qrc/utils/run_notebook.py     # execute in a fresh kernel
```

From inside `papers/distributed_qrc`, use `python ../../implementation.py --config configs/<name>.json`. Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `config_snapshot.json`, `run.log`, `run_status.json`, `metrics.json` (every cell: eval/test MAE, RMSE, R², MAE/RMSE in MW) and `test_predictions.npz`.

## Configuration

**Cell names.** A cell is `<reservoir type>|<readout type>`; these names are used in configs, `metrics.json` and `results/`:

| Reservoir type | In the paper? | Meaning |
|---|---|---|
| `classical` | yes | ReservoirPy reservoir with library defaults (the paper's classical reservoir, baseline B0) |
| `quantum_ideal` | yes | Network of 4-qubit quantum neurons, exact ideal simulation |
| `quantum_noisy` | yes | Same circuits with the hardware-informed noise model (App. B, Table III mapping) |
| `photonic` | no (MerLin adaptation) | One frozen 4-mode, 2-photon chip per neuron |
| `esn_tuned` | no (fair baseline B4) | The same ReservoirPy reservoir as `classical`, with spectral radius, leak rate and input scaling tuned on the eval split (12 settings) |
| `raw_lags` | no (fair baselines B1/B3) | No reservoir; the readout sees the 4-lag window directly |

| Readout type | In the paper? | Meaning |
|---|---|---|
| `linear` | yes (classical ridge, Eq. 5) | Linear ridge; an intercept is added by default (`--linear-intercept false` for the paper's Eq. 5) |
| `quantum_kernel_ideal`, `quantum_kernel_noisy` | yes ("quantum ridge") | Kernel ridge with the paper's fidelity kernel, ideal or noisy; evaluated in closed form |
| `photonic_kernel` | no (MerLin adaptation) | Kernel ridge with a 2-photon fidelity kernel |
| `rbf_kernel` | no (fair baselines B2b/B3) | Kernel ridge with an RBF kernel (γ and λ tuned on eval) |
| `persistence` | no (fair baseline B1) | The forecast is the last observed hour (only with `raw_lags`) |

**Paper-specific flags** (`cli.json` is the authoritative list):

| Flag | Effect | Default |
|---|---|---|
| `--observable {z_all,z0}` | Quantum neuron observable (R6) | `z_all` |
| `--update {synchronous,sequential}` | Reservoir propagation order (R7) | `synchronous` |
| `--normalization {train,per_split}` | Min-max fit (R3) | `train` |
| `--artifact-handling {keep,interpolate}` | Zero/negative loads in the series | `keep` |
| `--merge {mean,sum}` | Merge rule of distributed readouts (R12) | `mean` |
| `--noise-mapping {paper,ibm_brisbane,ibm_marrakesh,ionq_aria1}` | Device per component: Table III, or one device for all | `paper` |
| `--noise-scale X` | Multiplies every App. B error rate | `1.0` |
| `--reservoir-shots N`, `--kernel-shots N` | Finite-shot estimation | exact |
| `--linear-intercept {true,false}` | Intercept of the linear ridge | `true` |
| `--kernel-fit-subsample N` | Training points of the final kernel fit | full split |
| `--neuron-scale X`, `--kernel-scale X` | Photonic encoding scales (MerLin) | `1.0`, `2.0` |

The shared runtime adds `--config`, `--seed`, `--device`, `--outdir`, `--log-level` and `--data-root`.

## Data

- **Source:** IPTO "System Load" data (filetype `RealTimeSCADASystemLoad`): one public daily `.xls` file of hourly load in MWh, listed at https://www.admie.gr/en/market/market-statistics/detail-data and downloaded through the official IPTO File Download API (https://www.admie.gr/en/market/market-statistics/file-download-api), with no login. ADMIE (ΑΔΜΗΕ) is IPTO's Greek acronym.
- **Local path:** `data/distributed_qrc/ipto_greek_hourly_load/` (git-ignored): `raw/RealTimeSCADASystemLoad/` (1,807 daily files, 2020-05-05 … 2025-05-31, 64 MB), `api_listings/` and `SHA256SUMS`. Override with `--data-root`. Verify a copy with:

  ```bash
  cd data/distributed_qrc/ipto_greek_hourly_load && shasum -a 256 -c --quiet SHA256SUMS
  ```

- **Paper window:** 2020-10-22 hour 1 … 2024-10-21 hour 24 (35,064 values). The paper does not state it; it was identified by matching the first and last rows of Table II and the reported length, and the loader enforces it with a SHA-256 fingerprint.
- **Preprocessing:** chronological split 24,539 / 5,258 / 5,267; 4-lag windows built inside each split; min-max scale fitted on train.
- **Caveats:** the window contains 5 zero values (the missing 24th hour on 23-hour days) and 3 negative loads (2024-05-06), kept as published. Interpolating them leaves the headline unchanged. Reading the `.xls` files needs `xlrd`.
- **License:** IPTO publishes no explicit data license; its portal terms provide the content "as is". Cite IPTO and do not commit the raw files.

## Results Obtained and Comparison with the Paper

**One image per paper table and figure** (`results/paper_items/`, paper on the left or in blue, reproduction on the right or in green):

| Paper | Image |
|---|---|
| Tables I, II, III (architectures, data excerpt, QPU mapping) | `TABLE_1.png`, `TABLE_2.png`, `TABLE_3.png` |
| Tables IV–VII (best model per configuration: SRSR, MRSR, SRMR, MRMR) | `TABLE_4.png` … `TABLE_7.png` |
| Table VIII (best model per total neurons) | `TABLE_8.png` |
| Tables B1, B2 (calibration data and how our noise model uses it) | `TABLE_B1.png`, `TABLE_B2.png` |
| Tables C1–C12 (all cells) | `TABLE_C1.png` … `TABLE_C12.png` |
| Figures 1 and A1 (architectures) | `FIGURE_1.png`, `FIGURE_A1.png` (as implemented) |
| Figures 2 and 3 (neuron and kernel circuits) | `FIGURE_2.png`, `FIGURE_3.png` (as implemented, drawn with Qiskit) |
| Figures 4–7 (MAE and RMSE per architecture) | `FIGURE_4.png` … `FIGURE_7.png` (paper's style; in the printed Fig. 5, one RMSE curve repeats the SRSR values of Table C1; we plot Table C4) |

`results/results_summary.md` gives, for every paper item, the paper value, our value, the delta and the verdict, plus the index of all evidence sweeps. The cell-by-cell values are in `results/data/<architecture>_vs_paper.csv`.

**Headline reductions vs the paper's baseline B0** (paper rule: best quantum/hybrid cell per configuration by test R²; test MAE reduction, average / peak, mean over seeds):

| Architecture | Ours | Paper |
|---|---|---|
| SRSR (Table IV) | 23.4 ± 1.5 % / 64.9 % | 28.5 / 69.2 |
| MRSR (Table V) | 26.2 ± 3.6 % / 57.9 % | 24.7 / 63.4 (recomputed from the paper's tables; reported average 25.6) |
| SRMR (Table VI) | 44.1 ± 3.1 % / 75.5 % | 54.2 / 78.8 |
| MRMR (Table VII) | 51.8 ± 5.9 % / 75.4 % | 52.1 / 76.1 |

The peak RMSE reduction for MRMR is 72.5 % (paper 72.3 %); see `results/headline_reductions.png`. **Note:** B0 uses a linear readout, while the best quantum cells use a kernel readout (see Fair Baselines).

**Reproduced:**
- **Ideal quantum reservoir with the ideal kernel readout** (Tables C2, C5, C8, C11): median absolute deviation 2.6 %; 26/32 configurations within 5 % and all 32 within 12 %. The largest deviations are in small multi-readout configurations, where seed variance is highest (maximum 11.9 %, SRMR 25/5).
- The headline reductions vs B0 lie within one seed std of the paper for 6/8 statistics.

**Not reproduced, or only partly:**
- **C7, the best overall model** (MRSR 3x20, classical reservoir + kernel): 0.0146 ± 0.0010 vs the paper's 0.0113.
- **B0 cells** (classical reservoir + linear ridge) are strongly seed-dependent. Over 20 seeds, the paper's values fall inside our range for MRSR 6/8, SRMR 7/8 and MRMR 7/8 configurations, but for SRSR only 4/8 (`results/data/classical_seed_spread_<architecture>.json`).
- **Linear readout on quantum features:** the paper's cells are 10–49 % worse than ours. The missing intercept of the paper's Eq. 5 does not explain this; the cause is unresolved.
- **Noisy cells:** in the paper, noise degrades results by 14–35 %; with our App. B model it barely degrades them. Finite shots over-degrade 3–5×; about ×10 stronger depolarizing noise matches the reservoir side but not the kernel side (`results/data/shot_gap_analysis.json`, `results/data/robustness_verdicts.json`).
- **C6:** no noise-mitigation effect of distribution (`results/data/noise_distribution_comparison.json`; same-device control in Fair Baselines).

**Machines:** E1/E2 ran on a laptop CPU, E3/E4 on NVIDIA L4 GPUs; results agree across machines to 7e-11 (`results/checks/local_vs_gpu_crosscheck.json`).

## Fair Baselines

**Why.** The paper's classical reference (B0) is an untuned ReservoirPy reservoir with a **linear** ridge readout, while the best quantum and hybrid cells use a **kernel** readout. To separate the effect of the reservoir from that of the readout, we added classical models with comparable readouts. They were declared before any run (code: `lib/baselines_classical.py`).

| ID | Baseline | Code name (reservoir \| readout) | Question tested |
|---|---|---|---|
| B1 | Persistence; AR(4) linear ridge on the raw window | `raw_lags` \| `persistence`, `raw_lags` \| `linear` | What does the simplest forecast reach? |
| B2b | RBF kernel ridge on the same reservoir features | `classical` \| `rbf_kernel`, `quantum_ideal` \| `rbf_kernel` | Is the gain specific to the quantum kernel? |
| B3 | The paper's product kernel and an RBF kernel on the raw 4 lags | `raw_lags` \| `quantum_kernel_ideal`, `raw_lags` \| `rbf_kernel` | Does the reservoir help at all? |
| B4 | ReservoirPy ESN tuned on eval (12 settings), with linear and RBF readouts | `esn_tuned` \| `linear`, `esn_tuned` \| `rbf_kernel` | Was the paper's baseline just untuned? |

**Protocol.** Same data, split and seeds (0–2) as the reproduction. On each side, the reported cell is chosen per seed and configuration by eval RMSE, never on test. A cell is "better" if its mean test MAE is lower by more than 2 pooled seed standard deviations; Diebold-Mariano tests (lag 24) complement this.

**Results** (33/33 runs complete; `results/fair_comparison_table.md`, `results/fair_comparison.png`):

| Claim | Outcome | Evidence |
|---|---|---|
| C1 | Confirmed against B0; not confirmed against the additional baselines | The eval-selected quantum/hybrid cell is better in 0/8 configurations of every architecture (ties 4–6/8, otherwise worse). Average MAE difference vs the best additional classical model: −3.1 % (SRSR), −3.9 % (MRSR), −6.3 % (SRMR), −4.4 % (MRMR). Diebold-Mariano favours the classical model in 10/12 tests, the quantum cell in none |
| C2 | Not specific to quantum hardware | On B0 features the product kernel ties with RBF in 32/32 configurations; on quantum features it beats RBF in SRSR/MRSR, where the RBF γ sat at its grid edge |
| C5 | Not confirmed against the additional baselines | The noisy fully quantum cells are never better |
| C6 | Not confirmed | At ×10 noise with the same profile for every component, MRSR beats SRSR in 0/8 configurations and MRMR in 0/3. Under the paper's device mapping it is 4/8 and 1/3, consistent with SRSR being assigned the noisiest profile |
| C7 | Not confirmed | MRSR 3x20 hybrid 0.01459 ± 0.00097 vs 0.01365 ± 0.00032 for the best additional classical cell, within the decision tolerance |

**Key finding.** B3, the paper's product kernel on the 4 raw lags with no reservoir and computed classically, reaches test MAE 0.0138. It matches or beats the eval-selected quantum/hybrid cell in every configuration. In our setting, the reported reductions therefore reflect mainly the difference between a kernel readout and the paper's linear classical reference, rather than the quantum components themselves. We find no evidence of a quantum or photonic accuracy advantage on this task.

**Cost** (readout fit per cell, n = 24,535, laptop CPU / NVIDIA L4): product kernel 50 s / 71 s; RBF with 50 γ×λ candidates 80 s / 118 s; MerLin photonic kernel 44 s / 62 s; linear ridge < 0.1 s.

## Photonic Assessment and MerLin Extension

**Status: `PARTIAL_MERLIN_TRANSLATION`.** An adaptation to linear optics, not the paper's method, in simulation only. Translated: the ideal reservoir and kernel cells and the classical joining of chips (C1–C4, C7). Excluded: the gate-hardware noise claims C5/C6, whose noise models concern specific IBM/IonQ calibrations.

**Mapping** (`lib/photonic_merlin.py`):
- *Neuron:* a frozen 4-mode, 2-photon chip: random interferometer, angle encoding of the neuron's 4 inputs, random interferometer, threshold detection; output ⟨n₀⟩ − 1. The paper's wiring and 3 passes are kept.
- *Kernel:* a 10-mode (or 5-mode) 2-photon chip with 2 re-uploading blocks; |⟨ψ(x')|ψ(x)⟩|² is computed exactly from Fock amplitudes and does not factorize over modes.
- *Rejected:* `merlin.models.ReservoirClassifier` (feed-forward classification workflow, no recurrence); `merlin.FidelityKernel` (pairwise evaluation, infeasible at 24.5k², and it deviated from the exact fidelity, `results/checks/merlin_fidelitykernel_repro.txt`; reported to the MerLin team).
- *Verification:* both components match an independent Perceval computation to 1e-10 (`tests/test_photonic.py`). The encoding scales (neuron 1.0, kernel 2.0) were selected on SRSR 10 on the eval split.

**Results** (full grids, seeds 0–2; `results/merlin_comparison.png`). Tolerance declared before the runs: |Δ| ≤ max(10 %, 2 × pooled std) in ≥ 6/8 configurations.

| Architecture | Photonic + photonic kernel vs gate Q-ideal + kernel (median ΔMAE) | Photonic + linear vs gate Q-ideal + linear | Headline vs B0, MerLin (avg / peak) | Same rule, gate ideal cells |
|---|---|---|---|---|
| SRSR | +2.8 % (8/8 within tolerance) | −3.1 % (8/8) | 23.1 % / 63.7 % | 23.2 % / 64.9 % |
| MRSR | +2.5 % (8/8) | −2.3 % (8/8) | 25.1 % / 56.7 % | 26.1 % / 57.9 % |
| SRMR | +3.3 % (8/8) | −1.1 % (7/8) | 40.7 % / 65.1 % | 44.1 % / 75.5 % |
| MRMR | +3.0 % (8/8) | −0.6 % (8/8) | 46.8 % / 71.7 % | 51.8 % / 75.4 % |

The adaptation **preserves the gate-model behaviour** in all four architectures. Under a strict 10 % criterion, agreement drops in small multi-readout configurations (SRMR 10/2: +106 %), where the 5-mode, 2-photon kernel (15 amplitudes) is less expressive than the 5-qubit kernel (32). Against the additional classical baselines, the MerLin cells are better in 0/8 configurations of every architecture.

## Hardware-Aware Settings

All MerLin results are exact simulation in float64 (MerLin 0.4.2 / Perceval 1.3.1); none ran on hardware. There are no gradients: all photonic parameters are frozen and the readouts are closed-form.

| Component | Computation space | Detector | Photons | Modes | Input state | Encoding | Measurement | Postselection | Simulator | Shots | Wall-clock per run | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Neuron chip (`photonic` reservoir) | `UNBUNCHED` | threshold | 2 | 4 | [1,0,1,0] | phases `1.0·x` on 4 modes between two fixed random interferometers; 3 passes | `mode_expectations`, output ⟨n₀⟩ − 1 | yes: collision-free (2-click) events, renormalized | MerLin exact | none (exact) | E1/E2 18 min (laptop CPU); E3/E4 26 min (L4) | 0, 1, 2 |
| Kernel chip (`photonic_kernel`) | `FOCK` amplitudes | threshold (compute–uncompute) | 2 | 10 (5 for 5-feature instances) | [1,0,0,0,0,1,0,0,0,0] / [1,0,1,0,0] | 2 blocks × ⌈d/m⌉ layers of phases `2.0·x` + fixed interferometers | squared overlap of Fock amplitudes | none | MerLin exact | none (exact) | included above | 0, 1, 2 |

Not modelled: loss, distinguishability and dark counts; the post-selection success probability; the ~6 × 10⁸ sampled compute–uncompute estimates a hardware kernel would need.

## Next Steps

A follow-up proof of concept will use the distributed architectures, with the MerLin photonic reservoir, to validate a distributed hybrid stack: one reservoir per process, then per node, then per QPU, starting with MRMR. It targets the stack itself, not an accuracy or speed gain, and will be labelled as an extension.

## Limitations

- **Simulation only.** No QPU or photonic hardware. The noise model is rebuilt from App. B medians, and its strength had to be scaled ×10 to matter; the paper's noise construction is unknown.
- **3 seeds per cell** (20 for the classical-baseline spread). Several small multi-readout configurations have large seed variance.
- **Unresolved gaps:** C7 (0.0146 vs 0.0113); the paper's linear-readout cells on quantum features are 10–49 % worse than ours; the paper's noisy degradation (14–35 %) is not matched by any single noise scale.
- **Underspecified details** (R5–R7, R10, R12, R14–R16): chosen on the evaluation split where tested, or resolved from the paper's own tables (R12), and exposed as flags.
- **Fair-baseline grids.** The RBF γ and the ESN input scaling were often selected at the edge of their frozen grids. This under-tunes only the classical side, so the comparison is conservative.
- **MerLin adaptation:** 2 photons only; `merlin.FidelityKernel` was not usable.
- **Not tested:** C8 (multi-QPU deployment).

## Tests

```bash
cd papers/distributed_qrc
pytest -q
```

48 tests check the quantum neuron and the fidelity kernel against Qiskit and Qiskit Aer, the photonic neuron and kernel against Perceval, the data loader against the paper's Table II, the ridge, kernel-ridge and RBF solvers against direct solutions, the persistence forecast and ESN grid, and tiny end-to-end runs of every grid. Data tests are skipped when the IPTO data has not been acquired.

## Citation and License

Cite the original paper:

```bibtex
@misc{liliopoulos2026scalableqrc,
  title  = {Scalable Quantum Reservoir Computing over Distributed Quantum Architectures},
  author = {Liliopoulos, Ioannis and Varsamis, Georgios D. and Rallis, Konstantinos and Tsipas, Evangelos and Karafyllidis, Ioannis G. and Sirakoulis, Georgios Ch. and Dimitrakis, Panagiotis},
  year   = {2026},
  eprint = {2605.04991},
  archivePrefix = {arXiv}
}
```

The data is © IPTO (ADMIE), public and provided "as is". The reproduction code follows the repository's license.
