# Quantum Kernel Advantage over Classical Collapse — reproduction adaptation

This directory adapts
[`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage) for CPU
execution, the open PneumoniaMNIST dataset, and a MerLin 0.4 photonic kernel.
It belongs to a fork of
[`merlinquantum/reproduced_papers`](https://github.com/merlinquantum/reproduced_papers).

> **Scope:** the results below are surrogate experiments on raw
> PneumoniaMNIST pixels. They do not reproduce the paper's controlled
> MIMIC-CXR medical-foundation-model embeddings.

## Reference and attribution

- Paper: [*Quantum Kernel Advantage over Classical Collapse in Medical
  Foundation Model Embeddings*](https://arxiv.org/abs/2604.24597v1)
- Authors: Sebastián Andrés Cajas Ordóñez, Felipe Ocampo Osorio, Dax Enshan Koh,
  Rafi Al Attrach, Aldo Marzullo, Ariel Guerra-Adames, J. Alejandro Andrade,
  Siong Thye Goh, Chi-Yu Chen, Rahul Gorijavolu, Xue Yang, Noah Dane Hebdon,
  and Leo Anthony Celi
- Publication: arXiv preprint `arXiv:2604.24597v1`, 2026,
  [doi:10.48550/arXiv.2604.24597](https://doi.org/10.48550/arXiv.2604.24597)
- Original code: [`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage)
- Imported upstream revision: `9e80037305d683b0e70c94b8fa7dd648e1bac82b`
- Local changes and attribution: [NOTICE.md](NOTICE.md)
- Detailed scientific audit: [AUDIT.md](AUDIT.md)
- Current roadmap: [PLAN.md](PLAN.md)

## Original paper

The paper studies binary classification from frozen medical image embeddings.
It compares a QSVM using a qubit BSP feature map with classical linear and RBF
SVMs after PCA dimensionality reduction.

Its main claim is that the trace-normalized QSVM with `C=1` obtains a higher
minority-class F1 than the untuned linear SVM for every tested model/qubit
configuration. The paper also reports an advantage over a validation-tuned RBF
baseline at equal PCA dimension.

This repository focuses on that comparison, but current local constraints
change both the data representation and the quantum implementation.

## Reproduction scope

| Scope | Status | Meaning |
|---|---|---|
| Reference reproduction | Not run | Original task, controlled embeddings and qubit protocol |
| Open-data CPU surrogate | Implemented | Raw PneumoniaMNIST pixels with QSVM, linear SVM and RBF SVM |
| MerLin adaptation | Initial smoke implemented | Native photonic fidelity kernel, not the paper's BSP qubit kernel |

The upstream scientific pipeline is preserved as the initial baseline. Local
changes are limited to CPU execution, alternative-data compatibility and the
separate MerLin script.

## Installation

Conda is not required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The combined environment pins MerLin 0.4.0 and a compatible scikit-learn
version. The original repository used scikit-learn 1.6.1; the current MerLin
environment uses scikit-learn 1.9.0.

## Data

The catalogue default downloads the official PneumoniaMNIST archive when it is
missing, verifies its checksum, and prepares the training split. To prepare it
explicitly:

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

### CPU QSVM and classical baselines

The following reproduces the reviewed `q=4`, ten-seed surrogate comparison:

```bash
Q_VALUES=4 \
SEEDS=0,1,2,3,4,5,6,7,8,9 \
MAX_SAMPLES=100 \
RESULT_ROOT=outdir/q4-seed-stability \
bash scripts/run_table1_adaptation.sh
```

For each seed, the seed controls both deterministic subsampling and the
stratified 80/10/10 split. It is not an embedding-generation seed like those
used in the paper.

### MerLin photonic fidelity kernel

Run the reviewed CPU smoke:

```bash
XDG_DATA_HOME=/tmp/qsvm-merlin-data \
MPLCONFIGDIR=/tmp/qsvm-merlin-mpl \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python scripts/merlin_fidelity_kernel.py \
  --output_dir outdir/merlin/pca_2/seed_0 \
  --pca_dim 2 \
  --seed 0 \
  --circuit_seed 0 \
  --max_samples 100
```

Here `seed` controls the data subset and split, while `circuit_seed` controls
the fixed random parameters of the photonic feature map. With `pca_dim=2`, the
MerLin map uses three optical modes and the Fock input state `[1, 0, 1]`.

The script writes:

```text
output_dir/
|-- metrics_summary.csv
`-- dataset_info.json
```

## Results obtained

### Ten-seed CPU surrogate at q=4

All models used the same 100-sample seed/split pairs. The reported uncertainty
is the sample standard deviation across ten seeds.

| Model | Test minority F1, mean ± std | Zero-F1 seeds |
|---|---:|---:|
| QSVM, `C=1`, upstream trace path | 0.610 ± 0.356 | 2/10 |
| Linear SVM, `C=1` | 0.597 ± 0.351 | 2/10 |
| Validation-tuned RBF SVM | 0.630 ± 0.373 | 2/10 |

The QSVM mean is `+0.013` above the linear SVM and `−0.020` below the tuned
RBF. Paired against the linear SVM, it records one win, eight ties and one loss.
This small raw-pixel surrogate therefore does not reproduce the paper's broad
QSVM-advantage claim.

### MerLin smoke at PCA dimension 2, seed 0

| Split | Accuracy | Minority F1 | AUC |
|---|---:|---:|---:|
| Train | 0.8625 | 0.000 | 0.9499 |
| Validation | 0.9000 | 0.000 | 1.0000 |
| Test | 0.8000 | 0.000 | 1.0000 |

The kernel computation took 0.771 seconds and the complete run 0.843 seconds.
The Gram matrices passed the configured shape, finiteness, symmetry, diagonal,
range and PSD-tolerance checks.

The classifier nevertheless predicted only the majority class. The test AUC
of 1.0 means its scores ranked the two normal examples correctly, but its hard
decision threshold detected neither one. With only ten test images and one
seed, this is a technical smoke, not evidence for or against a photonic
advantage. No matching local two-component, seed-0 qubit/classical artifact
remains for an exact paired comparison.

Small, sanitized values used by this README are kept in
[`results/curated_results.csv`](results/curated_results.csv) and
[`results/merlin_q2_seed0.json`](results/merlin_q2_seed0.json).

## Notebook

[`notebook.ipynb`](notebook.ipynb) loads only the curated artifacts, explains
the minority-class F1 and plots the CPU and MerLin results. It performs no
kernel calculation and can therefore be read or executed quickly on CPU.

## Important limitations

- PneumoniaMNIST pixels are not equivalent to the paper's frozen MIMIC-CXR
  embeddings.
- The MerLin fidelity kernel is a native photonic adaptation, not a faithful
  implementation or resource match of the BSP qubit circuit.
- The current sample cap leaves only ten test images, including two minority
  examples for seed 0.
- The preserved upstream path fits its final MinMax transform using training
  plus held-out data. This is a preprocessing leak.
- The upstream trace path divides square training kernels by their trace while
  leaving rectangular cross-kernels unscaled. This behavior is preserved and
  documented in [AUDIT.md](AUDIT.md).
- Installing MerLin changed the environment from scikit-learn 1.6.1 to 1.9.0.
- No result here reproduces the paper's all-configuration or statistical
  significance claims.

## Tests and verification

The MerLin script was checked for syntax and import/CLI construction. The same
PCA-dimension-2 settings were run by the user and produced the reported
artifacts. Full QSVM grids are intentionally user-run because exact kernel
construction becomes expensive with sample count and qubit count.

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
The supported quick paths are the CPU/MerLin commands documented above.

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

The original code attribution and CC BY-NC-SA 4.0 terms are preserved. See
[LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). PneumoniaMNIST has its own data
and code terms, summarized in [NOTICE.md](NOTICE.md).
