# Quantum Kernel Advantage over Classical Collapse — reproduction adaptation

This directory adapts
[`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage) for CPU
execution, the open PneumoniaMNIST dataset, and a MerLin 0.4 photonic kernel.

> **Scope:** the results below are surrogate experiments on raw
> PneumoniaMNIST pixels. They do not reproduce the paper's gated
> MIMIC-CXR foundation-model embeddings.

## Reference and attribution

- Paper: [*Quantum Kernel Advantage over Classical Collapse in Medical
  Foundation Model Embeddings*](https://arxiv.org/abs/2604.24597v1), doi:10.48550/arXiv.2604.24597.
- Authors: Sebastián Andrés Cajas Ordóñez, Felipe Ocampo Osorio, Dax Enshan Koh,
  Rafi Al Attrach, Aldo Marzullo, Ariel Guerra-Adames, J. Alejandro Andrade,
  Siong Thye Goh, Chi-Yu Chen, Rahul Gorijavolu, Xue Yang, Noah Dane Hebdon,
  and Leo Anthony Celi
- Original code: [`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage)
- Imported upstream revision: `9e80037305d683b0e70c94b8fa7dd648e1bac82b`

## Original paper

The original task is not pneumonia detection. It uses chest X-rays to predict
whether a patient has Private insurance, the minority class, rather than
Medicaid or Medicare coverage.

Each X-ray is first passed through an image model that was already trained on a
large collection of images. The model converts the image into a fixed-size list
of numbers called an **embedding**. **Frozen** means that the image model's
weights are not updated during this experiment: it is used only as a fixed
feature extractor. The paper uses embeddings from MedSigLIP-448, RAD-DINO and a
general-purpose ViT.

Principal component analysis (PCA) then compresses each embedding to `q`
numbers. All compared classifiers receive these same `q` features. A support
vector machine (SVM) learns a boundary between the two classes. The linear SVM
can learn only a straight boundary in this reduced space, while the radial
basis function (RBF) SVM can learn a curved one. The QSVM encodes the `q`
numbers into a `q`-qubit state with the circuit that the paper calls BSP
(Block-Sparse Parameterization). Its quantum kernel is the squared overlap
between two encoded states: a larger value means that the two inputs look more
similar to this circuit. An SVM is then fitted from the resulting similarity
matrix.

The primary score is F1 for the minority class. It combines minority precision
(how often a minority prediction is correct) and recall (how many minority
examples are found). This score is zero if a model always predicts the majority
class, even when its overall accuracy appears reasonable.

In the paper's primary comparisons, trace normalization applies to the QSVM
kernel only. It divides the square training Gram matrix, the table of all
train-to-train similarities, by its trace, which is the sum of its diagonal.
The paper specifies that the same training trace must scale the test--train
matrix. The linear and RBF baselines use their ordinary scikit-learn kernels
without trace normalization. This kernel scaling is separate from the common
StandardScaler, PCA and MinMax preprocessing applied to the input features.

The preserved upstream code differs from the paper on this point: it scales
the square QSVM training matrix but leaves rectangular validation and test
cross-kernels unscaled. The local results below retain and label this historical
behavior instead of silently correcting it.

Its main claim is that the trace-normalized QSVM with `C=1` obtains a higher
minority-class F1 than the untuned linear SVM for every tested model/qubit
configuration. It reports 18 wins in 18 QSVM-versus-linear configurations,
while the linear model has minority F1 equal to zero on 90--100% of seeds. The
paper also reports seven wins in seven comparisons with a validation-tuned RBF
baseline at equal PCA dimension.

This repository focuses on that comparison. The CPU surrogate changes the task
and input representation; the separate MerLin path also changes the quantum
implementation.

## Reproduction scope

| Scope | Status | Meaning |
|---|---|---|
| Reference reproduction | Not run | Original task, gated MIMIC-CXR embeddings and qubit protocol |
| Open-data CPU surrogate | Implemented | Raw PneumoniaMNIST pixels with QSVM, linear SVM and RBF SVM |
| MerLin adaptation | Matched surrogate grid implemented | Native photonic fidelity kernel, not the paper's BSP qubit kernel |

The upstream scientific pipeline is preserved as the initial baseline. Local
changes are limited to CPU execution, alternative-data compatibility and the
separate MerLin script.

## Protocol-sensitivity audit

The local launcher can evaluate a 2×2 sensitivity matrix for two audited
implementation choices:

- MinMax preprocessing:
  - `legacy`: fit MinMaxScaler on training plus held-out data;
  - `train_only`: fit MinMaxScaler on training data only, without clipping
    held-out values.
- Trace scaling:
  - `legacy_square_only`: divide only square QSVM training kernels by their own
    trace;
  - `train_trace`: divide each training kernel and its matching rectangular
    cross-kernel by the same training-kernel trace.

The four protocol IDs are:

| Protocol ID | Preprocessing | QSVM trace protocol | MerLin normalization |
|---|---|---|---|
| `legacy_leak__legacy_trace` | training + held-out | `legacy_square_only` | `none` |
| `train_only__legacy_trace` | training only | `legacy_square_only` | `none` |
| `legacy_leak__train_trace` | training + held-out | `train_trace` | `train_trace` |
| `train_only__train_trace` | training only | `train_trace` | `train_trace` |

`legacy_leak__legacy_trace` retains the historical upstream protocol.
`train_only__train_trace` is the corrected local protocol. Comparing these
modes tests sensitivity to two implementation choices; it does not establish
which behavior the original authors intended.

Classical scikit-learn linear and RBF baselines have only the two preprocessing
variants because trace scaling applies to precomputed quantum kernels, not to
ordinary scikit-learn kernels. Their result for a given preprocessing mode is
therefore reused for both trace rows. MerLin does not have a historical
QSVM-style square-only trace mode: its historical mode is
`kernel_normalization=none`, while its corrected mode is `train_trace`.

The full protocol matrix has not yet been executed. No corrected numerical
result is reported in this README.

### Why not use the paper's embedding models here?

It is possible to use the listed model implementations. The
[MedSigLIP-448](https://huggingface.co/google/medsiglip-448),
[RAD-DINO](https://huggingface.co/microsoft/rad-dino) and
[ViT-patch32](https://huggingface.co/google/vit-base-patch32-224-in21k) model
pages are available, although MedSigLIP access requires accepting its terms of
use. Reference inputs still require either authorized MIMIC-CXR images and
insurance metadata to regenerate the embeddings, or access to the authors'
seed-specific precomputed files. Those
[precomputed embeddings announced by the
paper](https://huggingface.co/datasets/MITCriticalData/qml-mimic-cxr-embeddings)
are manually gated, and this environment has no authorized copy.

We could instead run these models on PneumoniaMNIST. That would be a useful
future **frozen-embedding surrogate** and would be closer to the paper than raw
pixels. It would still change the images, prediction task and embeddings, and
would require a documented resize, channel conversion, model revision and
preprocessing protocol. It was therefore kept out of this first minimal CPU
adaptation rather than presented as the reference experiment.

## Installation

Conda is not required. From this directory:

```bash
python3 -m venv .venv # Python 3.12
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The combined environment pins MerLin 0.4.0 and a compatible scikit-learn
version. The original repository used scikit-learn 1.6.1; the current MerLin
environment uses scikit-learn 1.9.0.

## Data

The two launch paths use distinct configurable locations. The shared catalogue
runtime follows the repository convention and reads
`../../data/qsvm_medimage/pneumoniamnist_train.pkl`. Direct paper scripts can
instead use the ignored paper-local file `data/pneumoniamnist_train.pkl` by
setting `DATA_PATH` or `--data_path` explicitly.

The catalogue default downloads the official PneumoniaMNIST archive when its
shared copy is missing, verifies its checksum, and prepares the training split.
To prepare that shared copy explicitly:

```bash
python scripts/prepare_pneumoniamnist.py \
  --download_path ../../data/qsvm_medimage/pneumoniamnist.npz \
  --output_path ../../data/qsvm_medimage/pneumoniamnist_train.pkl
```

This creates `data/qsvm_medimage/pneumoniamnist_train.pkl` under the repository
data root. Each 28×28 image is flattened to 784 normalized pixels. The labels
are `normal` and `pneumonia`; `normal` is the minority class in the experiments
below.

These pixels are accepted by the upstream `embedding` input field for
compatibility, but they are not frozen medical foundation-model embeddings.

## Catalogue runtime

The paper is discoverable through the repository-wide runtime. From this paper
directory, with the virtual environment activated:

```bash
python ../../implementation.py --list-papers
python ../../implementation.py --help
python ../../implementation.py
```

The last command runs the small CPU MerLin default from
[`configs/defaults.json`](configs/defaults.json). From the repository root, the
equivalent explicit command is:

```bash
python implementation.py --paper qsvm_medimage
```

Each run writes its resolved configuration, log, metrics, and dataset metadata
under:

```text
outdir/run_YYYYMMDD-HHMMSS/
|-- config_snapshot.json
|-- run.log
|-- metrics_summary.csv
`-- dataset_info.json
```

Paper-specific CLI options are declared in [`cli.json`](cli.json). Global
options such as `--config`, `--outdir`, `--seed`, `--device`, `--dtype`, and
`--data-root` are provided by the shared runtime.

## Direct experiment scripts

### CPU QSVM, classical baselines, and MerLin

From the repository root, the following single command runs the complete
protocol matrix. It runs the four protocol IDs, computes each classical
baseline once per preprocessing mode, and then performs paired aggregation:

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
bash papers/qsvm_medimage/scripts/run_table1_adaptation.sh
```

When already inside this paper directory, use
`bash scripts/run_table1_adaptation.sh` and paper-relative paths instead.
Default values for `LEAKAGE_MODES` and `TRACE_MODES` include all four
combinations.

The aggregator
[`scripts/aggregate_protocol_matrix.py`](scripts/aggregate_protocol_matrix.py)
writes:

```text
RESULT_ROOT/
|-- protocol_results_per_seed.csv
|-- protocol_summary.csv
`-- protocol_summary.md
```

The per-seed CSV preserves every paired model/baseline comparison. The summary
CSV contains mean minority F1, sample standard deviation, paired mean delta and
seed-level W/T/L. The Markdown file documents the same summary and its
perspective. These files are generated outputs under the selected
`RESULT_ROOT`; they are not committed numerical results.

For each seed, the seed controls both deterministic subsampling and the
stratified 80/10/10 split. It is not statistically equivalent to the paper's
seed-specific embedding datasets. The same seed is also passed to SVC
probability calibration, which is used for AUC; minority F1 is computed from
hard predictions. The QSVM and linear SVM use `C=1`; for each seed and `q`, the
RBF SVM selects `C` from `{0.01, 0.1, 1, 10, 100}` by validation minority-class
F1. The test split is not used for this selection.
For the same `q` and seed, the launcher also runs MerLin on the same N=500
subset, split and PCA dimension. Its circuit seed remains fixed at zero.

### MerLin photonic fidelity kernel

A kernel is a similarity score between two inputs. MerLin's
[`FidelityKernel`](https://merlinquantum.ai/0.4/user_guide/kernels.html) encodes
each PCA vector into a photonic quantum state and computes the squared overlap
between two states. A value near one means that the two states are very
similar. Repeating this calculation for every pair builds the Gram matrices
that a scikit-learn SVM can use with `kernel="precomputed"`.

The local script evaluates these overlaps exactly on CPU (`shots=None`) with a
fixed photonic feature map. This is a native MerLin construction, not a
translation of the paper's qubit BSP circuit. See also the
[MerLin 0.4 documentation](https://merlinquantum.ai/0.4/index.html).

The combined command above runs the matched grid. To run only one of its MerLin
configurations:

```bash
XDG_DATA_HOME=/tmp/qsvm-merlin-data \
MPLCONFIGDIR=/tmp/qsvm-merlin-mpl \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python scripts/merlin_fidelity_kernel.py \
  --data_path data/pneumoniamnist_train.pkl \
  --output_dir outdir/n500-q4-q6/merlin/q_4/seed_0 \
  --pca_dim 4 \
  --seed 0 \
  --circuit_seed 0 \
  --max_samples 500
```

Here `seed` controls the data subset and split, while `circuit_seed` controls
the fixed random parameters of the photonic feature map. With `pca_dim=4`, the
MerLin map uses five optical modes and the Fock input state `[1, 0, 1, 0, 1]`.
At `pca_dim=6`, it uses seven modes and `[1, 0, 1, 0, 1, 0, 1]`.

The script writes:

```text
output_dir/
|-- metrics_summary.csv
`-- dataset_info.json
```

## Results obtained

### Ten-seed CPU surrogate at q=4 and q=6, N=500

The values in this section come from the previously completed historical
`legacy_leak__legacy_trace` surrogate run. They are not results from the new
four-protocol matrix.

Each seed used 500 samples split into 400 training, 50 validation and 50 test
images. All four model paths used the same subset and split for a given seed
and `q`. The table below first tests the paper's Claim 1 with QSVM and its two
classical baselines. Values are mean test minority-class F1 ± sample standard
deviation across ten data/split seeds.

| q | QSVM | Linear | Q−linear | vs linear W/T/L | Tuned RBF | Q−RBF | vs RBF W/T/L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.809 ± 0.059 | 0.758 ± 0.080 | +0.051 | 7/0/3 | 0.760 ± 0.121 | +0.049 | 6/1/3 |
| 6 | 0.822 ± 0.078 | 0.786 ± 0.095 | +0.036 | 7/0/3 | 0.819 ± 0.091 | +0.003 | 4/1/5 |

W/T/L = Wins / Ties / Losses, counted seed by seed from the first-named model's
perspective.

Here the first-named model is QSVM. W/T/L measures stability across the ten
local data splits; it is not the paper's count of model/qubit configurations.

The QSVM has the highest mean minority F1 at both dimensions, so the direction
of the average difference agrees with the paper in these two surrogate
configurations. The result is not systematic, however. At `q=6`, the mean gain
over RBF is only `0.003`, and the QSVM loses to RBF on five of ten paired seeds.
No model has test minority F1 equal to zero, so the paper's linear-collapse
behavior is not reproduced. No local significance test has been run.

The other metrics do not show the same pattern. At `q=4` and `q=6`, mean QSVM
test AUC is `0.894` and `0.905`, below both the linear SVM (`0.955`, `0.960`) and
RBF SVM (`0.945`, `0.959`). At `q=6`, QSVM accuracy is also lower than both
baselines.

One diagnostic anomaly remains: QSVM training minority F1 is zero in all 20
runs even though its validation and test F1 are nonzero. The preserved upstream
path trace-normalizes each square training kernel but leaves the rectangular
validation and test cross-kernels unscaled. This scale mismatch is consistent
with the unusual train/held-out behavior, but this benchmark does not establish
that it is the only cause.

The paper itself notes that its DT9 data stratum was chosen after preliminary
experiments because it gave the strongest quantum results, and that its
non-collapse Tier-1 result for q ≥ 10 was validated only on DT9. These caveats
limit how broadly the paper's result can be generalized, but they do not by
themselves explain the local difference: this experiment uses q=4 and q=6, a
different dataset and target, raw pixels rather than frozen embeddings, and the
preserved upstream protocol concerns described below.

### Matched MerLin photonic adaptation

MerLin used the same N=500 subset, split and PCA dimension for every `q`--seed
pair. Its circuit seed was fixed at zero. The deltas and W/T/L counts below are
from MerLin's perspective.

| q | MerLin | M−QSVM | vs QSVM W/T/L | M−linear | vs linear W/T/L | M−RBF | vs RBF W/T/L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.758 ± 0.132 | −0.051 | 4/0/6 | −0.0002 | 6/0/4 | −0.002 | 4/2/4 |
| 6 | 0.770 ± 0.096 | −0.052 | 3/0/7 | −0.016 | 2/3/5 | −0.049 | 1/1/8 |

W/T/L = Wins / Ties / Losses, counted seed by seed from the first-named model's
perspective.

MerLin is nearly tied with the classical baselines on mean minority F1 at
`q=4`, but is lower than all three comparison models at `q=6`. It is lower than
QSVM on average at both dimensions. Its mean test AUC is `0.953` at `q=4` and
`0.960` at `q=6`, compared with `0.894` and `0.905` for QSVM. This illustrates
why the primary metric must remain explicit: higher AUC did not produce higher
minority F1 under the fitted SVM decisions.

All MerLin matrices passed the configured shape, finiteness, symmetry,
diagonal, range and PSD checks. One float32 self-overlap was `1.000132`; the
validation tolerance was therefore raised from `1e-4` to `2e-4`. The value is
recorded in `dataset_info.json` and the kernel is not clipped or rounded.

This matched result is useful for comparing local behavior, but it is still a
photonic adaptation and not evidence for or against the paper's qubit-BSP
claim.

The 80 detailed, sanitized records used by this README and the notebook are
kept in
[`results/q4_q6_n500_per_seed.csv`](results/q4_q6_n500_per_seed.csv).

## Notebook

[`notebook.ipynb`](notebook.ipynb) focuses on the paper's primary claim. It
explains the N=500 paired comparison and discusses the matched MerLin results
separately as a photonic adaptation. It loads only the curated CSV and performs
no kernel calculation, so it can be read or executed quickly on CPU.

## Important limitations

- PneumoniaMNIST pixels are not equivalent to the paper's frozen MIMIC-CXR
  embeddings.
- The MerLin fidelity kernel is a native photonic adaptation, not a faithful
  implementation or resource match of the BSP qubit circuit.
- The N=500 benchmark has only 50 test images per seed, including 11--14
  minority examples.
- The protocol-matrix workflow evaluates both historical preprocessing
  (MinMax fitted on training plus held-out data) and corrected train-only
  preprocessing. The historical behavior remains available and is not silently
  replaced.
- The workflow also evaluates both historical QSVM square-only trace scaling
  and corrected train-trace scaling of matching train/cross pairs. The full
  corrected matrix has not yet been run, so this README makes no corrected
  numerical claim. See [AUDIT.md](AUDIT.md).
- The paper's written BSP circuit and the preserved upstream circuit are not
  identical. This repository currently follows the upstream implementation.
- Installing MerLin changed the environment from scikit-learn 1.6.1 to 1.9.0.
- No result here reproduces the paper's all-configuration or statistical
  significance claims.

## Tests and verification

The MerLin script was checked for syntax and import/CLI construction. The shared
catalogue runtime smoke completed and wrote its configuration snapshot, log,
metrics, and dataset metadata. The user also ran the complete local historical
N=500 grid: 80 records covering four models, two dimensions and ten data/split
seeds. The full four-protocol sensitivity matrix has not been run. Larger
reference grids remain manual because exact kernel construction becomes
expensive with sample count and qubit count.

Fast local checks can be run from this directory:

```bash
python -m pytest -q tests
python ../../implementation.py --paper qsvm_medimage --help
```

## Legacy code

Obsolete HPC launchers, generated documentation, and private-path notebooks
were removed after an explicit inventory. The remaining upstream scientific and
analysis scripts are retained for provenance. Some still contain original HPC
path defaults and are unsupported; they are not used by the catalogue runner.
The supported quick paths are the CPU/MerLin commands documented above. See
[`legacy/README.md`](legacy/README.md) for the archived-code policy and the
provenance of helpers removed from the active API.

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
