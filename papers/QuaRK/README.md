# QuaRK: A Quantum Reservoir Kernel for Time Series Learning - Reproduction

Status: implementation done and tested (phase 02). No paper-size experiment has been run yet.

## Reference and Attribution

- Paper: "QuaRK: A Quantum Reservoir Kernel for Time Series Learning" (arXiv preprint, 2026)
- Authors: Abdallah Aaraba, Soumaya Cherkaoui, Ola Ahmad, Shengrui Wang
- arXiv: [2602.13531](https://arxiv.org/abs/2602.13531) (v1, 14 Feb 2026; cs.LG, quant-ph)
- Original repository: https://github.com/abdo-aary/quark (linked in Sec. 4 of the paper; MIT licence)
- License and attribution notes: cite the paper above when reusing this reproduction.

## Original Paper

According to its abstract, the paper proposes QuaRK, a time-series learning pipeline that couples a quantum reservoir with a kernel-based readout. The reservoir features are expectation values of k-local observables, estimated with classical shadow tomography. Circuit width, depth and measurement budget are exposed as tuning parameters. The paper gives generalization guarantees for dependent (beta-mixing) temporal data and tests the method on synthetic beta-mixing time series.

## Reproduction Scope (including Updates and Deviations)

To be defined in the planning phase.

## Install and How to Run

```bash
pip install -r papers/QuaRK/requirements.txt

# Smoke run (300 windows, two tasks, about 15 s): checks the pipeline, not evidence
python implementation.py --paper QuaRK --config configs/smoke_reduced.json

# QuaRK at the paper's settings (10,000 windows); seed 12345 is the authors' seed
python implementation.py --paper QuaRK --config configs/quark_original.json --seed 12345

# From inside papers/QuaRK (there is no paper-local implementation.py)
python ../../implementation.py --config configs/quark_original.json
```

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `run.log`, `run_status.json`, `config_snapshot.json`, `metrics.json`, `regularization_sweep.npz`, `training_size_sweep.npz` and `fit_examples.npz` (and `features.npz` for `quark_original`). Every run reports two choices of the ridge `reg`: chosen on validation data (our protocol) and chosen on the test set (the authors' rule for Fig. 4, kept only to reproduce that figure).

## Configuration

One config per model; all of them overlay `configs/defaults.json`:

| Config | Model |
|---|---|
| `defaults.json` | QuaRK at the paper's settings on 600 windows (lightweight default) |
| `smoke_reduced.json` | Smoke run: 300 windows, two tasks, short tuning search |
| `quark_original.json` | QuaRK as in the paper and the authors' code: n = 5 qubits, R = 3 sub-reservoirs, all 1- and 2-local Paulis, per-observable shot noise with 1000 shots |
| `quark_exact_original.json` | QuaRK with exact (noise-free) expectations |
| `quark_shadows_original.json` | QuaRK with classical shadows as the paper text describes them (1000 shared random-Pauli snapshots) |
| `raw_window_kernel_classical.json` | Same Matérn kernel ridge readout on the raw window |
| `raw_window_linear_classical.json` | Linear ridge on the raw window |
| `esn_default_classical.json` | Untuned echo state network (315 units) with the Matérn readout |
| `esn_tuned_classical.json` | Echo state network tuned on validation (18 configurations) with the Matérn readout |

Paper-specific flags (`cli.json` is the authoritative list): `--num-windows`, `--estimator`, `--shots`, `--num-qubits`, `--num-reservoirs`. Shared flags include `--seed`, `--outdir` and `--data-root`.

## Data

The paper uses synthetic data only: windows of a VARMA(3,3) process passed through tanh, labelled by three functionals (paper Sec. 4.1 and App. B). The authors publish the exact dataset and the arrays behind Figs. 2–4 as `storage.zip` in release v1.0.0 of their repository. This reproduction uses that file (about 24 MB):

```bash
# From the repository root
mkdir -p data/QuaRK/authors_release_v1.0.0
curl -L -o data/QuaRK/authors_release_v1.0.0/storage.zip \
  https://github.com/abdo-aary/quark/releases/download/v1.0.0/storage.zip
echo "ca1adc9c1a4f359f68e3bc45735552f1358baff086e859b7854c1d43eaca5567  data/QuaRK/authors_release_v1.0.0/storage.zip" | shasum -a 256 -c
unzip -q -o data/QuaRK/authors_release_v1.0.0/storage.zip -d data/QuaRK/authors_release_v1.0.0
```

The data root is `data/` at the repository root, or `$DATA_DIR` when that variable is set. The files used are under `data/QuaRK/authors_release_v1.0.0/storage/`:

| Path | Content |
|---|---|
| `data/synthetic/varma/N=10000__w=25__d=3__s=100/varma__e2_three__N=10000__w=25__d=3__s=100__seed=12345.npz` | `X`: 10,000 windows of 25 steps in 3 dimensions; `y`: the labels of the three tasks (3 × 10,000) |
| `results/reg_sweep/c-shadows/full/N=10000__w=25__d=3__s=100/2026-02-02_14-02-13/` | The authors' run behind Figs. 2–4: features, train/test split, selected kernels, regularization sweep, training-size sweep |

Limitations:
- The release holds one draw of the data (seed 12345). Extra seeds must be generated with the authors' procedure; they are not the authors' files.
- `artifacts/train_size_sweep.pkl` is a Python pickle. Load it only with an unpickler restricted to numpy arrays.
- The release has no licence of its own; the repository is MIT-licensed.

## Results Obtained and Comparison with the Paper

No results yet.

## Limitations

To be written once results are available.

## Tests

```bash
cd papers/QuaRK
pytest -q
```

## Citation and License

Cite arXiv:2602.13531. Code in this folder follows the repository license.
