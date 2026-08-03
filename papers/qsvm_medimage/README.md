# Quantum Kernel Advantage over Classical Collapse — reproduction adaptation

## Scope

This directory adapts the code imported from
[`sebasmos/qml-medimage` at revision `9e80037`](https://github.com/sebasmos/qml-medimage/tree/9e80037305d683b0e70c94b8fa7dd648e1bac82b)
for CPU execution, the open PneumoniaMNIST dataset, and a MerLin 0.4 photonic
kernel.

> **Scope:** the results below are surrogate experiments on raw
> PneumoniaMNIST pixels. They do not reproduce the paper's gated MIMIC-CXR
> foundation-model embeddings.

- Paper: [*Quantum Kernel Advantage over Classical Collapse in Medical
  Foundation Model Embeddings*](https://arxiv.org/abs/2604.24597v1),
  doi:10.48550/arXiv.2604.24597.
- Imported upstream revision: `9e80037305d683b0e70c94b8fa7dd648e1bac82b`.

## Paper summary

The original task is not pneumonia detection. It uses chest X-rays to predict
whether a patient has Private insurance, the minority class, rather than
Medicaid or Medicare coverage.

Each X-ray is converted to a frozen embedding by MedSigLIP-448, RAD-DINO, or a
general-purpose ViT. PCA compresses each embedding to `q` features. The linear
SVM, tuned RBF SVM, and QSVM receive the same reduced features. The QSVM encodes
them into a `q`-qubit state with the BSP (Block-Sparse Parameterization) circuit
and uses a compute-uncompute fidelity kernel.

The primary score is F1 for the minority class. It combines minority precision
and recall and is zero when a classifier always predicts the majority class.

The paper reports 18/18 QSVM wins against an untuned linear SVM and 7/7 wins
against a validation-tuned RBF SVM across its model/qubit configurations. This
repository evaluates the same comparison structure on a different, open-data
surrogate.

## Paper protocol versus imported upstream code

| Point | Paper | Imported upstream code |
|---|---|---|
| MinMax preprocessing | Common preprocessing described | [Default code fits MinMax on training plus held-out data](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/process.py#L6-L21) |
| Training-kernel trace | Divide by the training trace | [Square matrices are divided by their trace](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/core.py#L325-L341) |
| Validation/test cross-kernel | Divide by the same training trace | [Rectangular matrices are returned unchanged](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/core.py#L325-L341) |

The paper's written protocol is in [Section III.C, Equations (2) and
(3)](https://arxiv.org/html/2604.24597v1). The imported script
[normalizes the square and rectangular matrices in separate calls](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L732-L761),
so the rectangular guard makes cross-kernel trace normalization a no-op. The
same path is used for both
[validation and test](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L545-L555).

Fitting MinMaxScaler on held-out data is methodologically invalid because it
leaks held-out information into preprocessing. Its empirical impact is
negligible in this local surrogate. No conclusion about author intent can be
drawn from the code. The cross-kernel trace behavior does, however, contradict
the written protocol of the paper.

## Reproduction scope

| Scope | Status | Meaning |
|---|---|---|
| Reference reproduction | Not run | Original insurance task, gated MIMIC-CXR embeddings, and BSP qubit protocol |
| Open-data CPU surrogate | Completed locally | Raw PneumoniaMNIST pixels with QSVM, linear SVM, and tuned RBF SVM |
| MerLin adaptation | Completed locally | Native photonic fidelity kernel, not the BSP qubit circuit |

The completed local matrix uses at most 500 samples, split into 400 training,
50 validation, and 50 test samples. It evaluates `q` in `{4, 6}` with ten paired
data/split seeds, 0 through 9. The MerLin circuit seed is fixed at 0.

The paper's embedding files remain gated. Running the listed foundation models
on PneumoniaMNIST would be a different frozen-embedding surrogate and is not
part of this minimal adaptation.

## How to run

### Installation

From this paper directory:

```bash
python3 -m venv .venv # Python 3.12
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The combined environment pins MerLin 0.4.0 and a compatible scikit-learn
version.

### Data

The catalogue runtime reads
`../../data/qsvm_medimage/pneumoniamnist_train.pkl`. Direct scripts can instead
use the ignored paper-local file `data/pneumoniamnist_train.pkl` by setting
`DATA_PATH` or `--data_path`.

To prepare the shared copy:

```bash
python scripts/prepare_pneumoniamnist.py \
  --download_path ../../data/qsvm_medimage/pneumoniamnist.npz \
  --output_path ../../data/qsvm_medimage/pneumoniamnist_train.pkl
```

Each 28×28 image is flattened to 784 normalized pixels. The labels are
`normal` and `pneumonia`; `normal` is the minority class in these experiments.
The pixels occupy the upstream `embedding` field for compatibility but are not
medical foundation-model embeddings.

### Catalogue runtime

From this paper directory:

```bash
python ../../implementation.py --list-papers
python ../../implementation.py --paper qsvm_medimage --help
python ../../implementation.py --paper qsvm_medimage
```

The last command runs the small CPU MerLin default from
[`configs/defaults.json`](configs/defaults.json). Paper-specific options are in
[`cli.json`](cli.json). Each run writes its resolved configuration, log,
metrics, and dataset metadata under `outdir/run_YYYYMMDD-HHMMSS/`.

### One-command protocol matrix

From this paper directory, with the virtual environment activated:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=../../data/qsvm_medimage/pneumoniamnist_train.pkl \
Q_VALUES=4,6 \
SEEDS=0,1,2,3,4,5,6,7,8,9 \
MAX_SAMPLES=500 \
CIRCUIT_SEED=0 \
LEAKAGE_MODES=legacy,train_only \
TRACE_MODES=legacy_square_only,train_trace \
RESULT_ROOT=outdir/qsvm-protocol-matrix \
bash scripts/run_table1_adaptation.sh
```

The launcher runs four protocol IDs, computes each classical baseline once per
preprocessing mode, and performs paired aggregation:

| Protocol ID | Preprocessing | QSVM trace | MerLin normalization |
|---|---|---|---|
| `legacy_leak__legacy_trace` | training + held-out | `legacy_square_only` | `none` |
| `train_only__legacy_trace` | training only | `legacy_square_only` | `none` |
| `legacy_leak__train_trace` | training + held-out | `train_trace` | `train_trace` |
| `train_only__train_trace` | training only | `train_trace` | `train_trace` |

Classical scikit-learn baselines have only the two preprocessing variants:
trace modes apply only to precomputed quantum kernels. For MerLin, the common
launcher label `legacy_trace` maps to no normalization. It is only a shared
protocol label and does not imply that MerLin inherits any behavior from the
imported repository.

For each seed, deterministic subsampling precedes a stratified 80/10/10 split.
The QSVM and linear SVM use `C=1`. The RBF SVM selects `C` from
`{0.01, 0.1, 1, 10, 100}` using validation minority F1; the test set is not used
for selection.

## Main local audit finding

> The MinMax leakage has negligible impact on this PneumoniaMNIST surrogate:
> every aggregated minority-F1 change between matching leaky and train-only
> protocols is below 0.005.
>
> The QSVM trace mismatch is decisive. With consistent train-trace scaling,
> minority F1 is zero on every seed at q=4 and q=6, regardless of the
> preprocessing variant.
>
> Because the square QSVM training kernel is trace-normalized in both compared
> trace modes, the controlled difference is the scale of the held-out
> cross-kernel. The favorable held-out scores of the preserved upstream path
> therefore depend on this scale mismatch in the local surrogate.
>
> The paper's central linear-collapse-avoidance mechanism is not reproduced:
> the local linear baseline itself has non-zero minority F1.
>
> This result does not reproduce or refute the paper because the local
> experiment changes the dataset, target, input representation, sample count,
> and evaluated q range.

The comparison tests protocol sensitivity. It does not establish author intent
or whether the paper's MIMIC-CXR results are valid.

## Protocol matrix results

The following values are copied from the completed curated protocol summary.
F1 columns report mean ± sample standard deviation over ten paired data/split
seeds.

| Preprocessing | QSVM trace | q | QSVM F1 | Linear F1 | Q-linear | vs linear W/T/L | Tuned RBF F1 | Q-RBF | vs RBF W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train + held-out | square only | 4 | 0.808947 ± 0.059340 | 0.758009 ± 0.079520 | +0.050938 | 7/0/3 | 0.759963 ± 0.121262 | +0.048984 | 6/1/3 |
| train + held-out | square only | 6 | 0.822105 ± 0.077912 | 0.786008 ± 0.094895 | +0.036097 | 7/0/3 | 0.818913 ± 0.090559 | +0.003193 | 4/1/5 |
| train only | square only | 4 | 0.813727 ± 0.057270 | 0.758009 ± 0.079520 | +0.055718 | 8/0/2 | 0.759963 ± 0.121262 | +0.053763 | 6/1/3 |
| train only | square only | 6 | 0.822105 ± 0.077912 | 0.786008 ± 0.094895 | +0.036097 | 7/0/3 | 0.814960 ± 0.086978 | +0.007145 | 4/1/5 |
| train + held-out | train trace | 4 | 0.000000 ± 0.000000 | 0.758009 ± 0.079520 | -0.758009 | 0/0/10 | 0.759963 ± 0.121262 | -0.759963 | 0/0/10 |
| train + held-out | train trace | 6 | 0.000000 ± 0.000000 | 0.786008 ± 0.094895 | -0.786008 | 0/0/10 | 0.818913 ± 0.090559 | -0.818913 | 0/0/10 |
| train only | train trace | 4 | 0.000000 ± 0.000000 | 0.758009 ± 0.079520 | -0.758009 | 0/0/10 | 0.759963 ± 0.121262 | -0.759963 | 0/0/10 |
| train only | train trace | 6 | 0.000000 ± 0.000000 | 0.786008 ± 0.094895 | -0.786008 | 0/0/10 | 0.814960 ± 0.086978 | -0.814960 | 0/0/10 |

> **W/T/L = Wins / Ties / Losses**, counted seed by seed from the
> first-named model's perspective.

`Q-linear = QSVM F1 - linear F1`; `Q-RBF = QSVM F1 - tuned RBF F1`.
The W/T/L count is not the paper's count of model/qubit configurations.

### Preserved upstream-protocol result

The previous positive QSVM result is retained for provenance. It corresponds
only to `legacy_leak__legacy_trace`: leaky MinMax preprocessing plus
square-only QSVM trace scaling.

| q | QSVM F1 | Linear F1 | Q-linear | vs linear W/T/L | Tuned RBF F1 | Q-RBF | vs RBF W/T/L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.808947 ± 0.059340 | 0.758009 ± 0.079520 | +0.050938 | 7/0/3 | 0.759963 ± 0.121262 | +0.048984 | 6/1/3 |
| 6 | 0.822105 ± 0.077912 | 0.786008 ± 0.094895 | +0.036097 | 7/0/3 | 0.818913 ± 0.090559 | +0.003193 | 4/1/5 |

> **W/T/L = Wins / Ties / Losses**, counted seed by seed from the
> first-named model's perspective.

Historical upstream protocol only: leaky MinMax + square-only QSVM trace
scaling. No local paired significance test has been run.

## MerLin photonic adaptation

MerLin is absent from the imported `sebasmos/qml-medimage` revision. It is a
local photonic adaptation, not an implementation of the BSP qubit circuit, and
it has no mode inherited from upstream. The two evaluated variants are
the **MerLin unnormalized variant** and the **MerLin train-trace variant**.

The local script uses MerLin's `FidelityKernel` to compute exact CPU overlaps
with `shots=None`. The circuit seed is fixed at 0. The table values are copied
from `protocol_summary.csv`. That summary does not contain MerLin-versus-QSVM
rows, so those two requested columns are explicitly marked as not reported
rather than reconstructed.

| Preprocessing | MerLin normalization | q | MerLin F1 | M-QSVM | vs QSVM W/T/L | M-linear | vs linear W/T/L | M-RBF | vs RBF W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train + held-out | unnormalized | 4 | 0.757814 ± 0.131587 | not reported | not reported | -0.000195 | 6/0/4 | -0.002150 | 4/2/4 |
| train + held-out | unnormalized | 6 | 0.770021 ± 0.096427 | not reported | not reported | -0.015987 | 2/3/5 | -0.048891 | 1/1/8 |
| train only | unnormalized | 4 | 0.755433 ± 0.133619 | not reported | not reported | -0.002576 | 6/0/4 | -0.004531 | 4/2/4 |
| train only | unnormalized | 6 | 0.770021 ± 0.096427 | not reported | not reported | -0.015987 | 2/3/5 | -0.044939 | 1/1/8 |
| train + held-out | train trace | 4 | 0.000000 ± 0.000000 | not reported | not reported | -0.758009 | 0/0/10 | -0.759963 | 0/0/10 |
| train + held-out | train trace | 6 | 0.000000 ± 0.000000 | not reported | not reported | -0.786008 | 0/0/10 | -0.818913 | 0/0/10 |
| train only | train trace | 4 | 0.000000 ± 0.000000 | not reported | not reported | -0.758009 | 0/0/10 | -0.759963 | 0/0/10 |
| train only | train trace | 6 | 0.000000 ± 0.000000 | not reported | not reported | -0.786008 | 0/0/10 | -0.814960 | 0/0/10 |

> **W/T/L = Wins / Ties / Losses**, counted seed by seed from the
> first-named model's perspective.

`M-linear = MerLin F1 - linear F1`; `M-RBF = MerLin F1 - tuned RBF F1`.
Without trace normalization, MerLin is in the same broad performance range as
the local QSVM and classical baselines. With train-trace normalization and
fixed `C=1`, it follows the same qualitative zero-minority-F1 pattern as the
local QSVM. This supports consistency of the local comparison; it does not
establish an upstream bug, quantum advantage, or an optimized MerLin result.

Tuning `C`, varying `circuit_seed`, and exploring the photonic feature map,
modes, photon count, normalization, and kernel scale are future work, not part
of this minimal reproduction.

## Limitations

- The reference MIMIC-CXR reproduction has not been run.
- Pneumonia detection differs from the paper's insurance classification task.
- Raw pixels differ from frozen medical foundation-model embeddings.
- The local matrix covers only `q=4` and `q=6`.
- Each seed has only 50 test samples.
- Local seeds control deterministic subsets and splits, not embedding seeds.
- No local paired significance test has been run.
- The paper's linear-collapse result is not reproduced.
- MerLin is not resource-matched to the BSP qubit circuit.
- The MerLin feature map and `C` were not optimized.
- Local audit findings do not establish whether the paper's MIMIC-CXR results
  are valid.
- Paper limitations include noiseless simulation, SVM-only baselines, task
  scope, and post-hoc DT9 selection.

## Result artifacts

- [Per-seed protocol matrix](results/protocol_matrix_n500_q4_q6/protocol_results_per_seed.csv)
- [Aggregated protocol summary CSV](results/protocol_matrix_n500_q4_q6/protocol_summary.csv)
- [Aggregated protocol summary Markdown](results/protocol_matrix_n500_q4_q6/protocol_summary.md)
- [Run provenance](results/protocol_matrix_n500_q4_q6/RUN.md)
- [Historical upstream-protocol-only per-seed result](results/q4_q6_n500_per_seed.csv)

The user executed the completed protocol-matrix experiments. Neither Codex nor
the assistant executed them. The notebook remains a lightweight historical
analysis and does not calculate quantum kernels.

## Manual verification commands

From this paper directory:

```bash
python -m pytest -q tests
python ../../implementation.py --paper qsvm_medimage --help
git diff --check -- README.md
```

These commands are provided for manual use; this documentation update does not
run experiments, tests, or benchmarks.

## Legacy code

Obsolete HPC launchers, generated documentation, and private-path notebooks
were removed after an explicit inventory. Remaining upstream scientific and
analysis scripts are retained for provenance. Some contain original HPC path
defaults and are unsupported; they are not used by the catalogue runner. See
[`legacy/README.md`](legacy/README.md).

## Citation and license

```bibtex
@article{cajas2026qml,
  title   = {Quantum Kernel Advantage over Classical Collapse in Medical
             Foundation Model Embeddings},
  author  = {Cajas Ord\'{o}\~{n}ez, Sebasti\'{a}n Andr\'{e}s and
             Ocampo Osorio, Felipe and
             Koh, Dax Enshan and Al Attrach, Rafi and Marzullo, Aldo and
             Guerra-Adames, Ariel and Andrade, J. Alejandro and Goh, Siong Thye
             and Chen, Chi-Yu and Gorijavolu, Rahul and Yang, Xue and
             Hebdon, Noah Dane and Celi, Leo Anthony},
  journal = {arXiv preprint arXiv:2604.24597},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.24597}
}
```

The original code is attributed above. Its CC BY-NC-SA 4.0 terms are preserved
in [LICENSE](LICENSE), and this adaptation uses the same license.
PneumoniaMNIST is distributed through
[MedMNIST v2](https://medmnist.com/): the dataset is CC BY 4.0 and the MedMNIST
code is Apache-2.0. The archived dataset release is available on
[Zenodo](https://zenodo.org/records/10519652).
