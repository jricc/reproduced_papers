# Scalable Quantum Reservoir Computing over Distributed Quantum Architectures — Reproduction

Reproduction of distributed quantum reservoir computing (QRC) for next-hour Greek power-load forecasting, with additional classical baselines and a photonic (MerLin) adaptation.

**Main message.**
1. **The paper's numbers reproduce.** Against its own baseline we find gains of the same order, and the fully quantum models match the paper within a few percent.
2. **But the gain is not quantum.** It comes from replacing a linear readout by a kernel readout: the same kernel applied directly to the last 4 hours, without any reservoir and computed classically, does as well as every quantum model.
3. **What holds is the distributed architecture.** Small independent circuits combined classically work, also in photonics with MerLin; this is the basis of our follow-up proof of concept.

The quantum advantage we hoped to confirm did not appear in our setting. The paper's architectural idea, small independent quantum circuits combined classically, is simple and works as described, and it opens several research directions (see Next Steps).

**Where to start:** this README, then `notebook.ipynb` (the evidence, about 35 s on a laptop CPU), then `results/`.

## Reference and Attribution

- **Paper:** Scalable Quantum Reservoir Computing over Distributed Quantum Architectures (arXiv preprint, v1, 6 May 2026).
- **Authors:** Ioannis Liliopoulos, Georgios D. Varsamis, Konstantinos Rallis, Evangelos Tsipas, Ioannis G. Karafyllidis, Georgios Ch. Sirakoulis, Panagiotis Dimitrakis.
- **arXiv:** https://arxiv.org/abs/2605.04991
- **Original repository:** none listed; no original code is reused.
- **Data:** public data from IPTO, the Independent Power Transmission Operator of Greece (ADMIE), as cited by the paper.

## Original Paper

The paper forecasts the next hourly Greek electricity load (35,064 hourly values) from the last 4 hours. A fixed reservoir turns the window into features, and only a ridge readout is trained:

```text
  4-hour window x = (x[t-3], x[t-2], x[t-1], x[t])
        |
        v
  Reservoir (fixed)  ── classical echo state network, or a network of 4-qubit quantum neurons
        |
        |  features r(t), one value per neuron
        v
  Readout (trained)  ── linear ridge, or "quantum ridge" (kernel ridge with a quantum fidelity kernel)
        |
        v
  Forecast ŷ = x̂[t+1]
```

The classical reservoir is a ReservoirPy echo state network (ESN), a recurrent network with fixed random weights. A quantum neuron is a 4-qubit circuit (Fig. 2) whose output is a measured expectation value; the quantum ridge uses the kernel K(r, r') = |⟨Φ(r)|Φ(r')⟩|² of a 5- or 10-qubit circuit (Fig. 3, Eq. 6). Everything is simulated, ideally or with noise models of IBM and IonQ devices.

**The four architectures (Table I).** Every arrow carries classical data, so each branch can run on its own node or QPU:

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

Each architecture has 8 sizes (10–60 neurons) × 3 reservoir types × 3 readout types; the metrics are MAE, RMSE and R² on the normalized load, one value per cell. The paper's classical reference, which we call **B0**, is the ESN with library defaults and a linear ridge.

**Claims** of the paper's introduction and conclusions (arXiv v1), plus C6, which motivates our follow-up work:
- **C1:** quantum-enhanced models beat the classical baseline B0, with average MAE reductions of 25.6–54.2 % (Sec. III-B1–B4, pp. 5–8).
- **C2:** the quantum-kernel readout is the most consistent contributor (Sec. I, p. 2).
- **C3:** fully quantum models beat hybrids below 30 neurons and are more stable across sizes (Sec. IV, p. 9).
- **C4:** the gains persist under noise (Sec. I, p. 2).
- **C5:** the best model is a 3×20 classical reservoir + quantum kernel hybrid, MAE 0.0113 (Table VIII, p. 9).
- **C6:** the architectures suit multi-QPU deployment (qualitative; Sec. IV, p. 9).

## Reproduction Scope and Deviations

**Scope:** all four architectures, sizes and cells, on the full data, seeds 0, 1, 2, in simulation. The neurons and the kernel are computed with exact formulas, checked against Qiskit circuit simulations; the noise model is built from the paper's calibration medians (App. B).

**Deviations.** Where the paper is silent (data dates, ansatz angles, observable, wiring, regularization, merge rule, baseline settings, noise construction, best-cell selection), our choices and their justification are listed in [`docs/reproduction_details.md`](docs/reproduction_details.md). Two matter for the reading: the quantum reservoir keeps no memory between time steps, and we select the best cell on the evaluation split in addition to the paper's test-based rule.

**Software named in the paper:** Qiskit is used only to check our simulation; CUDA-Q is not used (it would not change exact results); the IBM and IonQ devices are used as noise profiles only, as in the paper; ReservoirPy 0.4.2 is used for the classical reservoir.

## Project Layout

```text
papers/distributed_qrc/
├── configs/   # paper grids, MerLin, fair baselines, diagnostics/
├── docs/      # reproduction details: deviations, cell names, flags, sweep IDs, commands
├── lib/       # model, data, readouts, noise, MerLin adaptation, baselines, runner
├── utils/     # sweep coordinator, analysis, plotting and verification scripts
├── tests/     # 48 tests
└── results/   # tables and figures; paper_items/ (one image per paper item); data/; checks/
```

## Install and How to Run

From the repository root:

```bash
.venv/bin/python -m pip install -r papers/distributed_qrc/requirements.txt
python implementation.py --paper distributed_qrc                                                    # smoke run, about 20 s
python implementation.py --paper distributed_qrc --config configs/e1_srsr_original.json --seed 0   # one paper grid (SRSR), 1–2 h on a CPU
```

All results come from sweeps (one config over seeds 0–2, collected in `outdir/sweep_<ID>/`), for example:

```bash
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-R2 --config configs/e1_srsr_original.json --seeds 0 1 2
```

Sweep IDs, configs, flags and analysis commands are explained in [`docs/reproduction_details.md`](docs/reproduction_details.md). The notebook is rebuilt and executed with `utils/build_notebook.py` and `utils/run_notebook.py`.

## Configuration

Configs are JSON overlays of `configs/defaults.json`; `cli.json` lists the flags. A model is named by its cell `<reservoir type>|<readout type>`, for example `classical|linear` (B0) or `quantum_ideal|quantum_kernel_ideal` (fully quantum, ideal); the glossary is in `docs/`.

## Data

- **Source:** IPTO "System Load" daily files, from https://www.admie.gr/en/market/market-statistics/detail-data (public, no login), stored in `data/distributed_qrc/` (not committed), with a `SHA256SUMS` file to verify a copy.
- **Window:** 2020-10-22 … 2024-10-21 (35,064 values), not stated in the paper; identified from Table II and enforced by a fingerprint.
- **Splits:** chronological 24,539 / 5,258 / 5,267 (train / eval / test), 4-hour windows, min-max scale fitted on train.

## Results Obtained and Comparison with the Paper

**Claim verdicts.**

| Claim | Against the paper's baseline B0 | Against the additional classical baselines |
|---|---|---|
| C1 | **Reproduced:** average reductions 23–52 % (paper 25.6–54.2 %) | **Not confirmed:** never better, worse by 3–6 % on average |
| C2 | **Reproduced:** the kernel readout drives the gain | **Not quantum:** a classical kernel does as well, even without reservoir |
| C3 | **Partially reproduced:** fully quantum models are stable, but their margin over hybrids is within seed variance | — |
| C4 | **Partially reproduced:** our noise model degrades results far less than the paper's (14–35 %) | **Not confirmed:** noisy models are never better |
| C5 | **Not reproduced:** MAE 0.0146 ± 0.0010, not 0.0113 | Ties with the best classical model (0.0137) |
| C6 | **Not tested:** components run independently, but no hardware was used | — |

**Headline reductions vs B0.** Test MAE reduction `1 − MAE(best quantum/hybrid cell) / MAE(B0)`, averaged over the 8 sizes and at its maximum (peak). The paper does not say how it picks its best cell; the highest test R² reproduces its printed values, except the MRSR average. Ours: mean ± std over 3 seeds.

| Architecture | Ours, average | Ours, peak | Paper, average | Paper, peak |
|---|---|---|---|---|
| SRSR (Table IV) | 23.4 ± 1.5 % | 64.9 ± 12.3 % | 28.5 % | 69.2 % |
| MRSR (Table V) | 26.2 ± 3.6 % | 57.9 ± 17.4 % | 25.6 % | 63.3 % |
| SRMR (Table VI) | 44.1 ± 3.1 % | 75.5 ± 6.5 % | 54.2 % | 78.8 % |
| MRMR (Table VII) | 51.8 ± 5.9 % | 75.4 ± 7.0 % | 52.1 % | not given |

Example, SRSR with 10 neurons: in the paper, B0 has MAE 0.0458 and the best cell 0.0141, so 1 − 0.0141 / 0.0458 = 69.2 %. Every size, with its B0 value and selected cell, is in `results/results_summary.md`.

**Other findings:** the fully quantum cells match the paper closely (test MAE median deviation 2.6 %, 26/32 sizes within 5 %); B0 is unstable across seeds; two gaps remain unexplained (the paper's linear readouts on quantum features are 10–49 % worse than ours, and its noisy cells degrade more). `results/paper_items/` holds one image per paper table and figure, with the paper's values next to ours.

## Fair Baselines

B0 has a **linear** readout, while the best quantum models have a **kernel** readout. To find where the gain comes from, we added two classical controls, declared before any run:

| Control | Question | Answer |
|---|---|---|
| **No reservoir:** the same readouts applied directly to the last 4 hours (a linear ridge, i.e. an AR(4) model, or the paper's kernel) | Is the reservoir needed? | **No.** With a linear readout, the model without reservoir (MAE 0.0163) beats B0 at 28/32 sizes; with the paper's kernel, it (MAE 0.0138) is as good as or better than the best quantum model at every size. |
| **Classical kernel:** an RBF kernel instead of the quantum kernel, on the same reservoir features | Is the gain specific to the quantum kernel? | **No.** On the classical reservoir's features, the RBF kernel does as well at 32/32 sizes. |

The verdicts compare the best quantum model with the best classical one (these controls, B0 and an ESN tuned on the evaluation split), both chosen on the evaluation split; a model is better if its mean test MAE is lower by more than 2 seed standard deviations.

**Conclusion.** The gain needs a kernel readout, not a quantum reservoir: the kernel without any reservoir matches every quantum model. The paper's remark that distribution mitigates noise (Sec. III-B5) does not hold either: with the same noise on every component, MRSR never beats SRSR. Details: `results/fair_comparison_table.md`.

## Photonic Assessment and MerLin Extension

**Status: `PARTIAL_MERLIN_TRANSLATION`** (simulation; the noise claim C4 is excluded). Each quantum neuron becomes a frozen 4-mode, 2-photon chip, and the kernel a 10-mode (or 5-mode) 2-photon chip; both are checked against Perceval to 1e-10.

| Architecture | Photonic vs gate model, kernel cells (median ΔMAE) | Photonic vs gate model, linear cells |
|---|---|---|
| SRSR | +2.8 % (8/8 sizes within tolerance) | −3.1 % (8/8) |
| MRSR | +2.5 % (8/8) | −2.3 % (8/8) |
| SRMR | +3.3 % (8/8) | −1.1 % (7/8) |
| MRMR | +3.0 % (8/8) | −0.6 % (8/8) |

The adaptation preserves the behaviour of the gate model (tolerance declared before the runs: within 10 % or 2 seed std in at least 6/8 sizes) and, like it, does not beat the classical baselines. `merlin.FidelityKernel` was not used: it deviates from the exact fidelity (reported to the MerLin team).

## Hardware-Aware Settings

- **Simulation:** exact, float64, `merlinquantum` 0.4.2 and Perceval 1.3.1; no hardware, no gradients (frozen chips, closed-form readouts).
- **Neuron chip:** `UNBUNCHED`, threshold detection, 2 photons in 4 modes `[1,0,1,0]`, post-selected on 2-click events, no shots; about 18 min per run on a laptop CPU, 26 min on an L4 GPU.
- **Kernel chip:** `FOCK` amplitudes, 2 photons in 10 (or 5) modes, exact overlaps, no shots.
- **Not modelled:** loss, distinguishability, dark counts and the post-selection success rate.

## Next Steps

The paper's architecture is easy to distribute: in MRMR, each branch needs only the input window and returns only its forecast (`lib/distributed.py`; the notebook runs each branch in its own process and gets the same forecast). It opens several directions:
- **Distributed stack:** a proof of concept running the branches on separate processes, then nodes, then QPUs, starting with MRMR and the MerLin reservoir.
- **Kernels that are hard to simulate:** adding entangling gates to the kernel circuit, so that a quantum kernel can differ from classical ones.
- **Reservoirs with memory:** carrying information from one hour to the next, so that the reservoir adds something the input window does not contain.
- **Harder tasks:** longer horizons or inputs without explicit lags, where a reservoir has a real role.
- **Hardware:** real QPUs and photonic chips with more photons, with their noise, loss and shot costs.

## Limitations

- Simulation only; the noise model is rebuilt from calibration medians, and the paper's construction is unknown.
- 3 seeds per cell; small multi-readout sizes have large seed variance.
- The classical baselines were often selected at the edge of their tuning grids, which can only under-tune them.
- MerLin adaptation with 2 photons only.

## Tests

`cd papers/distributed_qrc && pytest -q` runs 48 tests: quantum and photonic components against Qiskit, Qiskit Aer and Perceval, the data loader against Table II, the solvers, and small end-to-end runs.

## Citation and License

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
