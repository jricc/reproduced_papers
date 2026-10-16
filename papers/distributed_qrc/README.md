# Scalable Quantum Reservoir Computing over Distributed Quantum Architectures — Reproduction

Reproduction of the paper's distributed quantum reservoir computing (QRC) architectures for next-hour Greek power-load forecasting, with a fair classical comparison and a photonic (MerLin) adaptation.

**Status (2026-10-16):**
- the paper grids are reproduced (phase 03);
- the MerLin adaptation is complete (phase 04);
- the fair comparison is complete (phase 05);
- the documentation is complete (phase 06);
- a distributed-stack PoC is planned after the reproduction (see Next Steps).

**Verdict in one line:** the paper's numbers against its own baseline reproduce in trend, but **no quantum advantage survives a fair comparison**. The paper's own product kernel on the raw 4 lags, computed classically with no reservoir, matches or beats every quantum cell.

Full log: `LOG.md`. Durable lessons: `INSIGHTS.md`.

## Reference and Attribution

- **Paper:** Scalable Quantum Reservoir Computing over Distributed Quantum Architectures (arXiv preprint, v1, 6 May 2026; under submission).
- **Authors:** Ioannis Liliopoulos, Georgios D. Varsamis, Konstantinos Rallis, Evangelos Tsipas, Ioannis G. Karafyllidis, Georgios Ch. Sirakoulis, Panagiotis Dimitrakis.
- **DOI/arXiv:** https://arxiv.org/abs/2605.04991
- **Original repository:** none listed; no original code is reused.
- **Attribution:** cite the arXiv preprint above. The dataset is IPTO/ADMIE public data (see Data).

## Original Paper

The paper forecasts the next-hour Greek electricity load (IPTO, 35,064 hourly values) from a 4-hour sliding window. Each model has a reservoir layer and a ridge readout.

**Reservoir:**
- *Classical:* ReservoirPy.
- *Quantum:* a network of N "quantum neurons". Each neuron is a 4-qubit circuit with RX encoding and an RX/CX/RZ ansatz (Fig. 2). It reads data lags and other neurons' outputs, and the network is propagated 3 times.

**Readout:**
- *Classical:* linear ridge.
- *"Quantum ridge":* kernel ridge with a fidelity kernel. Its feature map is H and P(2x) layers on 5 or 10 qubits (Fig. 3).

Both reservoirs and kernels are simulated ideally or with hardware-informed noise (ibm_brisbane, ibm_marrakesh, IonQ Aria-1; App. B).

**Architectures (Table I):**
- *SRSR:* single reservoir, single readout.
- *MRSR:* 2, 3 or 5 reservoirs feeding one readout.
- *SRMR:* one reservoir feeding 2, 3 or 5 readout instances.
- *MRMR:* k reservoirs, each with its own readout.

Each architecture has 8 configurations at 10–60 total neurons. The full grid is 3 reservoir types × 3 readout types per configuration, reported in Tables III–VI and C1–C12.

**Metrics:** MAE, RMSE and R² on min-max-normalized load, with one run per cell.

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

**Reproduced:**
- All four architectures × 8 configurations × the full 9-cell grid.
- Full data and the full training set for every kernel fit (n = 24,535).
- Seeds 0, 1, 2, plus 20 seeds for the classical-baseline spread.
- Simulation only, with no QPU.
- Evidence level V1 (full-data, multi-seed). Every sweep ran with a pre-execution plan and pinned code (`-R2` sweeps).

**Implementation:**
- *Neurons:* exact numpy simulation of the 4-qubit neurons, folded into 16×16 functionals.
- *Kernel:* closed-form fidelity kernel. The feature map has no entangling gate, so the kernel factorizes over qubits.
- *Cross-checks:* both are checked against Qiskit (statevector) and Qiskit Aer (noisy density matrix) in `tests/`.
- *Noise model:* rebuilt from the App. B medians. It uses 1q and 2q depolarizing channels after native gates and a symmetric readout flip. No T1/T2 is added on top of the RB errors (`lib/noise.py`).

**Deviations and choices.** The paper leaves these unspecified; all 17 items are in `LOG.md#Reproducibility Assessment`.

| Item | Paper | Choice here |
|---|---|---|
| Data window (R1) | 35,064 hourly values, dates not given | 2020-10-22 … 2024-10-21, identified from the Table II anchors |
| Normalization (R3) | Min-max "in each dataset" | Primary: min-max fitted on train. Variant: per split (`--normalization per_split`). Headline unchanged; one SRSR-10 cell shifts by more than its seed std (S2-R2) |
| Ansatz angles (R5) | "Trainable w" | Fixed, uniform in [0, 2π), seeded (reservoir computing) |
| Observable (R6) | "Pauli Z" | Z⊗4, chosen on P1 by eval RMSE (S1-R2: 0.021545 vs Z₀ 0.076412) |
| Wiring and update (R7) | k_i ∈ {0, 1, 2} recurrent inputs | Uniform sources, synchronous update, 3 passes. In S1-R2, sequential update (0.021543) is tied within 1 % with synchronous (0.021545); the first-listed candidate is kept |
| λ (R10) | Not given | 10-value grid, n-normalized, selected on eval RMSE |
| Distributed merge (R12) | Not given | Contiguous feature partition, mean of instance predictions. This was resolved in phase 03 by evidence, without a sweep: `sum` would double the prediction, and feeding the full r(t) to every instance would make the linear SRMR cells identical to SRSR, which the paper's tables contradict (paper 0.0750 vs 0.0458; ours 0.0608 vs 0.0438). `--merge sum` is available but was not swept |
| Classical reservoir (R14) | ReservoirPy, no hyperparameters | Library defaults = paper baseline B0; tuned ESN added in phase 05 |
| Noise construction (R15) | Not given | Depolarizing + readout from the App. B medians; no shots |
| Best-cell selection (R16) | Best of 8 cells, chosen on test | Reported with the paper's rule (leakage disclosed) and with eval selection |

**Claim verdicts (summary).** Details are in Results and Fair Baselines.

| Claim | Outcome |
|---|---|
| C1 | Trend reproduced vs the paper's baseline B0. **Unsupported** vs fair matched classical models (phase 05) |
| C2 | The kernel readout is the consistent contributor (reproduced), but the "quantum" kernel is a classical closed-form kernel. **Unsupported** as a quantum advantage |
| C3 | "Comparable" supported; "best model always distributed" unsupported |
| C4 | Fully quantum cells are stable across sizes; their advantage over hybrids is within seed noise |
| C5 | Holds vs B0 under our noise model; **unsupported** vs fair classical models |
| C6 | **Unsupported**: no benefit under a same-device control; the paper's effect is consistent with its device assignment |
| C7 | **Not reproduced** (0.0146 vs 0.0113) and **unsupported** (ties with fair classical models) |
| C8 | Not testable |

## Project Layout

```text
papers/distributed_qrc/
├── README.md, LOG.md, INSIGHTS.md, FEEDBACK.md, CONFLUENCE.md, VISITED_URLS.md
├── ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md   # MerLin-specific feedback
├── cli.json, requirements.txt, notebook.ipynb
├── configs/          # defaults.json, *_original.json (paper), *_merlin.json (phase 04), *_classical.json and c6_*_noise.json (phase 05), gap_*/xcheck_* (diagnostics)
├── lib/              # data, quantum_neuron, noise, reservoirs, kernels, readouts, architectures, metrics, runner, photonic_merlin (phase 04), baselines_classical (phase 05), experiment_logging
├── utils/            # run_sweep.py (sweep coordinator), analysis and plotting scripts, verification scripts
├── tests/            # 48 tests (Qiskit, Qiskit Aer and Perceval ground truth, data, solvers, end-to-end runs)
├── results/          # curated tables, JSON summaries and figures
└── outdir/           # raw runs and sweeps (git-ignored)
```

## Install and How to Run

From the repository root, in the shared repository venv:

```bash
.venv/bin/python -m pip install -r papers/distributed_qrc/requirements.txt
```

**Smoke run.** `configs/defaults.json`: SRSR 10 neurons, 9 cells, kernels on 4,000 points; about 20 s on a laptop CPU.

```bash
python implementation.py --paper distributed_qrc
```

**Paper grids** (Tables III–VI), one seed per run. E1/E2 take 1–2 h per seed on a laptop CPU. E3/E4 are intended for a GPU.

```bash
python implementation.py --paper distributed_qrc --config configs/e1_srsr_original.json --seed 0
python implementation.py --paper distributed_qrc --config configs/e3_srmr_original.json --seed 0 --device cuda:0
```

**Sweeps as run.** Seeds 0–2, sweep evidence in `outdir/sweep_<ID>/`:

```bash
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-R2 --config configs/e1_srsr_original.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-MERLIN --config configs/e1_srsr_merlin.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-CLASSICAL --config configs/e1_srsr_classical.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id B1B3-RAW --config configs/b1b3_raw_lags_classical.json --seeds 0 1 2
python papers/distributed_qrc/utils/run_sweep.py --sweep-id C6-SRSR --config configs/c6_srsr_noise.json --seeds 0 1 2 \
  --candidates '[{"noise_mapping": "paper", "noise_scale": 10}, {"noise_mapping": "ibm_brisbane", "noise_scale": 10}]'
```

The exact command of every sweep is in `LOG.md` (sweep records and Remote Launches).

**Analysis.** Each script reads the raw `metrics.json` files and writes to `results/`:

```bash
python papers/distributed_qrc/utils/compare_with_paper.py papers/distributed_qrc/outdir/sweep_E1-R2      # per-cell vs paper
python papers/distributed_qrc/utils/headline_statistics.py papers/distributed_qrc/outdir/sweep_E{1,2,3,4}-R2
python papers/distributed_qrc/utils/merlin_comparison.py papers/distributed_qrc/outdir                   # phase 04
python papers/distributed_qrc/utils/fair_comparison.py papers/distributed_qrc/outdir                     # phase 05
python papers/distributed_qrc/utils/fair_comparison_table.py
python papers/distributed_qrc/utils/documentation_tables.py papers/distributed_qrc/outdir                # results_summary.md
```

From inside `papers/distributed_qrc`, use `python ../../implementation.py --config configs/<name>.json`.

**Notebook.** `notebook.ipynb` is the scientific walkthrough: claim, sanity checks, a live reduced run, stored full results, fair baselines, controls and MerLin. It runs in about 35 s on a laptop CPU, provided the IPTO data is in `data/distributed_qrc/`.

```bash
python papers/distributed_qrc/utils/build_notebook.py   # regenerate from source
python papers/distributed_qrc/utils/run_notebook.py     # execute in a fresh kernel
```

**Output of a run:**

```text
outdir/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # resolved configuration
├── run.log                # lifecycle, per-instance readout fits, per-cell evaluations
├── run_status.json        # structured status (code commit, lib hash, dataset, selection rule, hashes)
├── metrics.json           # every cell: eval/test MAE, RMSE, R², test MAE/RMSE in MW, selections per instance
└── test_predictions.npz   # test predictions of every cell and the test targets
```

## Configuration

| Config | Content |
|---|---|
| `defaults.json` | Smoke-scale defaults and every shared setting: data window and split, reservoir and readout settings, λ grid, App. B noise values, photonic scales, classical-baseline grids |
| `p1_srsr_original.json`, `p1_srmr_original.json` | P1: SRSR 10 and SRMR 10/2, ideal cells, full-n kernels |
| `e1_srsr_original.json` … `e4_mrmr_original.json` | Paper grids E1–E4 (8 configurations × 9 cells), full-n kernels |
| `e3_srmr_reduced.json`, `e4_mrmr_reduced.json` | RED-tier fallback (kernels fitted on 8,000 points). Not used: E3/E4 ran at full n on GPUs |
| `gap_*`, `xcheck_*` | Phase-03 gap diagnostics and machine cross-checks |
| `p1_srsr_merlin.json`, `e1_srsr_merlin.json` … `e4_mrmr_merlin.json` | Phase-04 MerLin adaptation (scale tuning and grids) |
| `e1_srsr_classical.json` … `e4_mrmr_classical.json`, `b1b3_raw_lags_classical.json`, `c6_*_noise.json` | Phase-05 fair baselines, reservoir-free floors and the C6 control |

**Paper-specific flags.** `cli.json` is the authoritative list:

| Flag | Effect | Default |
|---|---|---|
| `--observable {z_all,z0}` | Quantum neuron observable (R6) | `z_all` |
| `--update {synchronous,sequential}` | Reservoir propagation order (R7) | `synchronous` |
| `--normalization {train,per_split}` | Min-max fit; `per_split` is the paper-faithful reading (R3) | `train` |
| `--artifact-handling {keep,interpolate}` | Zero/negative loads in the series | `keep` |
| `--merge {mean,sum}` | Merge rule of distributed readouts (R12) | `mean` |
| `--noise-mapping {paper,ibm_brisbane,ibm_marrakesh,ionq_aria1}` | Device per component: Table III, or one device for all | `paper` |
| `--noise-scale X` | Multiplies every App. B error rate | `1.0` |
| `--reservoir-shots N`, `--kernel-shots N` | Finite-shot estimation (gap diagnosis) | exact |
| `--linear-intercept {true,false}` | Intercept of the linear ridge (paper Eq. 5 has none) | `true` |
| `--kernel-fit-subsample N` | Training points of the final kernel fit | full split |
| `--neuron-scale X`, `--kernel-scale X` | Phase-04 photonic encoding scales | `1.0`, `2.0` |

The shared runtime adds `--config`, `--seed`, `--device`, `--outdir`, `--log-level` and `--data-root`.

## Data

**Source:**
- IPTO/ADMIE, the Greek transmission system operator. The data type is "System Load" (filetype `RealTimeSCADASystemLoad`): one public daily `.xls` file of hourly "real time load demand" in MWh.
- The files are listed at https://www.admie.gr/en/market/market-statistics/detail-data and downloaded through the official IPTO File Download API (https://www.admie.gr/en/market/market-statistics/file-download-api), with no login.

**Local path:** `data/distributed_qrc/ipto_greek_hourly_load/` under the repository data root (git-ignored). It contains:
- `raw/RealTimeSCADASystemLoad/` (1,807 daily files, 2020-05-05 … 2025-05-31, 64 MB);
- `api_listings/` (API responses and `file_urls.txt`);
- `SHA256SUMS`.

`defaults.json` sets `"data_root": "../../data"`, resolved against this paper folder; override it with `--data-root`. The parsed series is cached in `processed/`.

**Paper window:**
- 2020-10-22 hour 1 … 2024-10-21 hour 24, using the first 24 hourly columns of each file: 35,064 values.
- The paper does not state it. It was identified by matching the first and last rows of the paper's Table II and the reported length exactly.
- The loader enforces it with a SHA-256 fingerprint.

**Preprocessing:**
- Chronological split of the raw values: 24,539 / 5,258 / 5,267.
- 4-lag windows built inside each split, oldest lag first.
- Min-max normalization, with the scale fitted on train (primary).

**Acquisition and verification:** the exact `curl` command is in `LOG.md#Data Acquisition Log`. Verify a copy with:

```bash
cd data/distributed_qrc/ipto_greek_hourly_load && shasum -a 256 -c --quiet SHA256SUMS
```

No fallback or synthetic data was used.

**License:** IPTO publishes no explicit data license; its portal terms provide the content "as is". Cite IPTO and do not commit the raw files.

**Data caveats:** the window contains 5 zero values (the missing 24th hour on 23-hour days) and 3 negative loads (2024-05-06). They are kept as published (paper-faithful). Interpolating them leaves the headline unchanged, but two SRSR-10 cells shift by more than their seed std (S2-R2). Reading the `.xls` files needs `xlrd`.

## Results Obtained and Comparison with the Paper

**Generated tables.** All are full-data runs over seeds 0–2. `results/results_summary.md` is generated by `utils/documentation_tables.py` from the structured artifacts and holds:
- the complete **paper-item table**: paper value, reproduced value, delta, metric and trend agreement, claim support, seeds and comment, for Tables III–VI, C1–C12 (by cell type) and VII, and the C6 comparison;
- the **sweep index**: all 28 evidence sweeps, each with its rule, run counts, completeness and recomputed selection, linking to its full candidate table `outdir/sweep_<ID>/summary.json`.

**Per-cell comparison:** `results/E{1..4}-R2_vs_paper.csv` and figures `results/E{1..4}-R2_grid.png`.

**Headline reductions vs the paper's baseline B0** (paper rule: best quantum/hybrid cell per configuration by test R²). Values are test MAE reductions, average / peak, as mean over seeds:

| Architecture | Ours | Paper |
|---|---|---|
| SRSR (Table III) | 23.4 ± 1.5 % / 64.9 % | 28.5 / 69.2 |
| MRSR (Table IV) | 26.2 ± 3.6 % / 57.9 % | 24.7 / 63.4 (recomputed from the paper's tables; reported average 25.6) |
| SRMR (Table V) | 44.1 ± 3.1 % / 75.5 % | 54.2 / 78.8 |
| MRMR (Table VI) | 51.8 ± 5.9 % / 75.4 % | 52.1 / 76.1 |

The peak RMSE reduction for MRMR is 72.5 % (paper 72.3 %). See `results/headline_reductions.png` and `results/headline_statistics.json`. **Caveat:** B0 is mismatched (see Fair Baselines).

**Reproduced (metric agreement):**
- The ideal quantum reservoir with the ideal kernel readout lies within 12 % of the paper in 32/32 configurations (median Δ −2 % to +3 %).
- The headline reductions vs B0 lie within one seed std of the paper for 6/8 statistics.

**Not reproduced, or only partly:**
- **C7, the best overall model** (MRSR 3x20, classical reservoir + kernel): 0.0146 ± 0.0010 vs the paper's 0.0113.
- **B0 cells** (classical reservoir + linear ridge): they are strongly seed-dependent. Over 20 seeds, the paper's values fall inside our range for MRSR 6/8, SRMR 7/8 and MRMR 7/8 configurations, but for SRSR only 4/8 (`results/GAP-CLASSICAL-*-R2_spread.json`).
- **Linear readout on quantum features:** the paper's cells are 10–49 % worse than ours. The missing intercept of the paper's Eq. 5 does not explain this (S5-R2); the cause is unresolved.
- **Noisy cells:**
  - In the paper, noise degrades results by 14–35 %; with our App.-B model it barely degrades them.
  - Finite shots over-degrade 3–5× (S3-R2).
  - About ×10 stronger depolarizing noise matches the reservoir side but not the kernel side (S4-R2; `results/shot_gap_analysis.json`).
- **C6:** no noise-mitigation effect of distribution (`results/noise_distribution_comparison-R2.json`; same-device control in Fair Baselines).

**Machines:** E1/E2 ran on a laptop CPU, E3/E4 on NVIDIA L4 GPUs. Machine equivalence was checked on the first configuration of E3 and E4 (max |difference| 6.0e-11 and 6.7e-11, identical λ selections; `results/xcheck-R2.json`).

## Fair Baselines

**Paper baseline state: mismatched** (phase 05, `MISSING_OR_UNFAIR`). The paper compares kernel-readout models with an untuned ReservoirPy reservoir plus a **linear** ridge (B0). Its "quantum" kernel is a product state, which we evaluate classically in closed form; this is verified against the Qiskit statevector.

**Baselines added.** They were declared in phase 01 and frozen in `LOG.md#Phase 5 Comparison Plan` before any run. Code: `lib/baselines_classical.py`.

| ID | Baseline | Matching |
|---|---|---|
| B1 | Persistence; AR(4) linear ridge on the raw window | Same window, split, λ protocol; no reservoir |
| B2b | RBF kernel ridge on the same reservoir features (γ × λ selected on eval) | Identical features, readout instances, merge |
| B3 | The paper's product kernel and an RBF kernel on the raw 4 lags | Same window and protocol; no reservoir |
| B4 | ReservoirPy ESN tuned on eval (12 settings of spectral radius × leak rate × input scaling), with linear and RBF readouts | Same total neurons, sub-reservoirs, readout instances, node seeds as B0 |

**Selection.** On each side, the reported cell is chosen per seed and configuration by eval RMSE, never on test. The per-configuration rule is "better" if Δ mean test MAE < −2 × pooled seed std.

**Results.** 33/33 runs, complete and validated: `results/fair_comparison_table.md`, `results/fair_comparison.json`, `results/fair_comparison.png`.

| Claim | Outcome | Evidence |
|---|---|---|
| C1 | **Unsupported** vs fair classical models; supported only vs B0 | The eval-selected quantum/hybrid cell is better in 0/8 configurations of every architecture (ties 4–6/8, otherwise worse). Average MAE reduction vs the best fair classical model: −3.1 % (SRSR), −3.9 % (MRSR), −6.3 % (SRMR), −4.4 % (MRMR). Diebold-Mariano tests (lag 24) favour the classical model in 10/12 seed-level tests at SRSR 10, MRSR 3x20, SRMR 10/2 and MRMR 2x5; none favours the quantum cell |
| C2 | **Unsupported** | The "quantum" kernel is a classical closed-form kernel. On B0 features it ties with RBF in 32/32 configurations; on quantum features it beats RBF in SRSR/MRSR, where the RBF γ sat at its grid edge |
| C5 | **Unsupported** vs fair classical models | The noisy fully quantum cells are never better |
| C6 | **Unsupported** | At ×10 effective noise with the same device for every component, MRSR beats SRSR in 0/8 configurations and MRMR in 0/3. Under the paper's device mapping it is 4/8 and 1/3, which is attributable to SRSR being placed on the noisiest device |
| C7 | **Unsupported** | The MRSR 3x20 hybrid scores 0.01459 ± 0.00097 against 0.01365 ± 0.00032 for the fair classical cell: a tie |

**Key finding.** B3, the paper's product kernel on the 4 raw lags with no reservoir and computed classically, reaches test MAE 0.0138. It matches or beats the eval-selected quantum/hybrid cell in every configuration. The large reductions in the paper come from comparing kernel models with an untuned linear readout on default ReservoirPy features. **No quantum or photonic advantage is supported by this evidence.**

**Wall-clock (readout fit per cell, n = 24,535):**

| Cell type | Laptop CPU | NVIDIA L4 |
|---|---|---|
| Product kernel | 50 s | 71 s |
| RBF (50 γ×λ candidates) | 80 s | 118 s |
| MerLin photonic kernel | 44 s | 62 s |
| Linear ridge | < 0.1 s | — |

Per-sweep times are in `LOG.md#Phase 5 execution ledger`.

## Photonic Assessment and MerLin Extension

**Status: `PARTIAL_MERLIN_TRANSLATION`.** This is an adaptation to linear optics, not the paper's method, in simulation only.
- **Translated:** the ideal reservoir and kernel cells and the classical joining of chips (C1–C4, C7).
- **Excluded:** the gate-hardware noise claims C5/C6. Their noise models concern specific IBM/IonQ calibrations.

**Mapping** (`lib/photonic_merlin.py`):
- *Neuron:* each neuron becomes a frozen 4-mode, 2-photon chip: random interferometer, angle encoding of the neuron's 4 inputs, random interferometer, threshold detection. The output is ⟨n₀⟩ − 1. The paper's wiring and 3 passes are kept.
- *Kernel:* the fidelity kernel becomes a 10-mode (or 5-mode) 2-photon chip with 2 re-uploading blocks. The kernel |⟨ψ(x')|ψ(x)⟩|² is computed exactly from Fock amplitudes and is non-factorizable over modes.

**Rejected alternatives:**
- `merlin.models.ReservoirClassifier`: a feed-forward classification workflow with standardized embeddings and no recurrence.
- A Perceval photonic noise model: it would answer a different question than C5/C6.
- `merlin.FidelityKernel`: it evaluates pairs one at a time, which is infeasible at 24.5k², and it disagreed with Perceval (see `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`).

**Verification:** both components match an independent Perceval computation to 1e-10 (`tests/test_photonic.py`).

**Scale tuning:** the encoding scales were tuned on SRSR 10 (sweep `P1-MERLIN-SCALES`, eval split; complete candidate table in `results/results_summary.md`). The defaults (neuron 1.0, kernel 2.0; mean eval RMSE 0.021318) were selected by the 1 % tie rule. They were tied with neuron 1.0, kernel π (0.021165, lowest), which comes later in the candidate list.

**Results** (full grids, seeds 0–2; `results/merlin_comparison.json`, `results/merlin_comparison.png`):

| Architecture | Photonic + photonic kernel vs gate Q-ideal + kernel (median ΔMAE) | Photonic + linear vs gate Q-ideal + linear | Headline vs B0, MerLin (avg / peak) | Same rule, gate ideal cells |
|---|---|---|---|---|
| SRSR | +2.8 % (8/8 within tolerance) | −3.1 % (8/8) | 23.1 % / 63.7 % | 23.2 % / 64.9 % |
| MRSR | +2.5 % (8/8) | −2.3 % (8/8) | 25.1 % / 56.7 % | 26.1 % / 57.9 % |
| SRMR | +3.3 % (8/8) | −1.1 % (7/8) | 40.7 % / 65.1 % | 44.1 % / 75.5 % |
| MRMR | +3.0 % (8/8) | −0.6 % (8/8) | 46.8 % / 71.7 % | 51.8 % / 75.4 % |

The tolerance was declared before the runs: |Δ| ≤ max(10 %, 2 × pooled std) in ≥ 6/8 configurations.

**What this shows:**
- The adaptation **preserves the gate-model behaviour** in all four architectures.
- Under a strict 10 % criterion, agreement drops in the small multi-readout configurations (for example SRMR 10/2 is +106 %). There, the 5-mode, 2-photon kernel (15 amplitudes) is less expressive than the 5-qubit kernel (32), and seed variance is large.
- Against the fair classical models, the MerLin cells are better in 0/8 configurations of every architecture.

## Hardware-Aware Settings

All MerLin results are exact simulation in float64 (MerLin 0.4.2 / Perceval 1.3.1). None ran on hardware. There are no gradients: all photonic parameters are frozen and the readouts are closed-form.

| Component | Computation space | Detector | Photons | Modes | Input state | Encoding | Measurement | Postselection | Simulator | Shots | Wall-clock per run | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Neuron chip (`photonic` reservoir) | `UNBUNCHED` | threshold | 2 | 4 | [1,0,1,0] | phases `1.0·x` on 4 modes between two fixed random interferometers; 3 passes | `mode_expectations`, output ⟨n₀⟩ − 1 | yes: collision-free (2-click) events, renormalized | MerLin exact | none (exact) | E1/E2 18 min (laptop CPU); E3/E4 26 min (L4) | 0, 1, 2 |
| Kernel chip (`photonic_kernel`) | `FOCK` amplitudes | threshold (compute–uncompute) | 2 | 10 (5 for 5-feature instances) | [1,0,0,0,0,1,0,0,0,0] / [1,0,1,0,0] | 2 blocks × ⌈d/m⌉ layers of phases `2.0·x` + fixed interferometers | squared overlap of Fock amplitudes | none | MerLin exact | none (exact) | included above | 0, 1, 2 |

**Not modelled:**
- loss, distinguishability and dark counts;
- the post-selection success probability;
- the ~6 × 10⁸ sampled compute–uncompute estimates a hardware kernel would need.

Full table: `LOG.md#Hardware-Aware Result Summary`.

## Next Steps

**Follow-up PoC (defined 2026-10-16 with the user, set aside until the reproduction is finished).** The aim is to use the distributed architectures as a simple, MerLin-only **proof of concept that the distributed hybrid stack works**. It does not aim at an accuracy or speed gain. The PoC is labelled as an extension, not as a result of the paper.
- **Stages:**
  - *Local, on a laptop or the L4 GPUs:* multi-core, one reservoir per process.
  - *At TGCC, the same three levels:* multi-core (one reservoir per core), multi-node (one reservoir per node), multi-QPU (one reservoir per QPU).
- **Model:** the phase-04 MerLin photonic reservoir. Each neuron is a frozen 4-mode, 2-photon chip, run in sampled (shots) mode from the local stage onward. The readout is classical (linear ridge). MRMR is the first target: k independent reservoir + readout pipelines followed by a mean. MRSR, which needs a gather of features, comes second.
- **Why it maps well:**
  - k reservoirs run in parallel on k workers or QPUs.
  - Inside a reservoir, the 3 passes need only 3 synchronizations per worker, because every window is independent and can be batched per pass.
  - The number of circuits per reservoir (windows × neurons × 3) is a budget question. A reduced, fixed problem keeps it small, for example MRMR 3×5 with a few hundred windows.
- **Validation per level:**
  - *Exact mode:* bit-equality with the sequential reference. The per-sub-reservoir seeds are already deterministic.
  - *Sampled or QPU mode:* every estimated expectation within ±3σ of the shot noise around the exact value. Bytes exchanged, per-pass synchronization time, QPU time and shots are reported.
- **Open points at TGCC:**
  - how jobs reach the QPU from compute nodes;
  - the number of accessible QPUs;
  - offline installation of MerLin;
  - job arrays vs MPI.
- **Code change needed:** a reservoir-executor interface (sequential / process pool / GPU / scheduler-MPI / remote QPU) at the point where `PhotonicReservoir.transform` evaluates a batch of neurons per pass. Data, readout, merge, metrics and logs are unchanged.

## Limitations

- **Simulation only.** No QPU or photonic hardware. The noise model is rebuilt from App. B medians, and its strength had to be scaled ×10 to matter; the paper's noise construction is unknown.
- **3 seeds per cell** (20 for the classical-baseline spread). Several small multi-readout configurations have large seed variance.
- **Unresolved gaps:**
  - C7 (0.0146 vs 0.0113);
  - the paper's linear-readout cells on quantum features are 10–49 % worse than ours;
  - the paper's noisy degradation (14–35 %) is not matched by any single noise scale.
- **Underspecified details** (R5–R7, R10, R12, R14–R16). The choices were made on the evaluation split where tested, or resolved from the paper's own tables (R12), and are exposed as flags. The merge alternatives were not run as a sweep.
- **Fair-baseline grids.** The RBF γ and the ESN input scaling were selected at the edge of their frozen grids in most cases. This under-tunes only the classical side, so the comparison is conservative.
- **MerLin adaptation:** 2 photons only. `merlin.FidelityKernel` was not usable (see `ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md`).
- **Not tested:** C8 (multi-QPU deployment) and the per-device qubit-reduction argument, which applies only to the kernel readout here.

## Tests

```bash
cd papers/distributed_qrc
pytest -q
```

48 tests check:
- the quantum neuron and the fidelity kernel against Qiskit (statevector) and Qiskit Aer (noisy density matrix);
- the photonic neuron and kernel against Perceval;
- the data loader against the paper's Table II;
- the ridge, kernel-ridge and RBF solvers against direct solutions;
- the persistence forecast, and the ESN grid and tie rules;
- tiny end-to-end runs of the gate, photonic and baseline grids.

Data tests are skipped when the IPTO data has not been acquired.

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

The data is © IPTO/ADMIE, public and provided "as is". The reproduction code follows the repository's license.
