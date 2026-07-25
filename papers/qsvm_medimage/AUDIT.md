# Audit and Discussion Notes

## Purpose

This file preserves the main conclusions of the repository, paper, upstream,
and MerLin audit performed on 24 July 2026. It is a discussion record, not a
claim that the listed issues have been corrected or that experiments have been
validated.

No test, training, evaluation, dataset download, benchmark, or figure generation
was run during the audit.

## Current preservation decision

- Keep the scientific behavior of the authors' pipeline as the initial
  baseline.
- Limit local edits inside upstream code to what is needed for serial/CPU
  execution and alternative-dataset compatibility.
- Keep integration code, configs, and result curation outside the upstream
  implementation when possible.
- Use the first CPU/open-data runs to characterize the preserved pipeline.
- Do not silently fix the audited scientific concerns before those baseline
  tests and a discussion of their results.
- If corrections are approved later, keep the upstream baseline available and
  give the corrected protocol a distinct config/mode and artifact identity.

This decision prioritizes traceability to
[`sebasmos/qml-medimage`](https://github.com/sebasmos/qml-medimage) over an
immediate rewrite. A baseline result may be useful for debugging and comparison,
but it must not be presented as a validated scientific result when a known
protocol concern applies.

## Audited sources

- Paper: [*Quantum Kernel Advantage over Classical Collapse in Medical
  Foundation Model Embeddings*](https://arxiv.org/html/2604.24597v1), arXiv
  `2604.24597v1`, 27 April 2026.
- Authors' code: commit
  [`9e80037305d683b0e70c94b8fa7dd648e1bac82b`](https://github.com/sebasmos/qml-medimage/commit/9e80037305d683b0e70c94b8fa7dd648e1bac82b).
- Catalogue target: <https://github.com/merlinquantum/reproduced_papers>.
- MerLin target: [MerLin 0.4 documentation](https://merlinquantum.ai/0.4/index.html)
  and source tag `0.4.0`.

The audited upstream repository had one public commit on `main`, no release/tag,
and no CI workflow. The paper mentions `scripts/run_all.sh`, but that file was
not present in the audited revision.

## Paper conclusions relevant to this reproduction

### Task and protocol

- The reference task is binary insurance classification on a 2,371-sample
  MIMIC-CXR subset, with Private insurance as the minority/positive class.
- The primary metric is minority-class F1; accuracy and AUC are secondary.
- Inputs are frozen MedSigLIP-448, RAD-DINO, and ViT-patch32-cls embeddings,
  reduced with `StandardScaler -> PCA(q) -> MinMaxScaler`.
- The primary comparison uses a fidelity QSVM with `C=1`, trace normalization,
  and one data-encoding repetition.
- Tier 1 compares QSVM `C=1` against linear SVM `C=1` at equal PCA dimension.
- Tier 2 compares QSVM `C=1` against an RBF SVM whose `C` is selected from
  `{0.01, 0.1, 1, 10, 100}` with `gamma="scale"`.
- The paper reports noiseless simulation evidence, not quantum-hardware or
  computational speedup.

The original dataset and precomputed embeddings are gated and unavailable in
the current environment. The paper reports substantial H100 compute for its
full sweep, so the complete reference experiment is outside the current CPU
budget.

### Artifact interpretation and priority

- Table 1 is the conceptual two-tier comparison summary.
- Tables 2 and 4 contain the detailed Tier 1 and Tier 2 evidence from which a
  local Table 1 analogue should be derived.
- Table 5 and Figure 3 (effective rank/eigenspectra) are the strongest mechanism
  artifacts.
- Figure 4 (kernel heatmap), Figure 5 (`q` sweep), and Table 7 (normalization)
  are useful secondary artifacts.
- Table 8, Table 10, projected kernels, `q=16`, and the full grid should be
  deferred until the CPU cost and basic protocol are understood.

The paper's `18/18` and `7/7` win counts belong only to its reference task and
grid. They must not be reused for a smaller surrogate experiment.

## Current local state

- The branch is `qsvm_medimage`.
- The authors' Qiskit/MPI/cuQuantum files, HPC launchers, notebooks, tests, and
  generated documentation are present.
- Local changes already add optional GPU imports, a serial MPI fallback, and an
  exact Qiskit statevector CPU path.
- Local data support prepares PneumoniaMNIST and maps it to the upstream
  `target`/`embedding` schema.
- A bounded Table 1 adaptation launcher covers `q = 2, 4, 6`, five seeds, and at
  most 200 samples, with linear and tuned-RBF baselines.
- No current MerLin implementation or curated local scientific result is
  present.
- The previous Git history contains a synthetic shared-runtime/MerLin prototype,
  but it should not be restored wholesale. Its photonic kernel was an analogue,
  its dependency declaration was incomplete, and its historical results are not
  current evidence.

PneumoniaMNIST uses 28x28 raw pixels and predicts normal versus pneumonia. It
changes both the representation and the task relative to the paper. It is a
useful pipeline smoke dataset, not an equivalent replacement for foundation
embeddings or insurance labels.

## Scientific observations to discuss

These findings are retained for later discussion. Under the preservation
decision above, they are not immediate authorization to alter the upstream
scientific behavior.

### 1. Trace normalization of cross-kernels

**Confirmed upstream behavior, not introduced by the CPU adaptation.**

In upstream
[`qve/core.py`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/core.py#L356-L372),
`normalize_kernel_trace` divides a matrix by its trace only when it is square and
returns rectangular matrices unchanged. Upstream
[`apply_hybrid_kernel`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L809-L840)
calls this function separately on the square train Gram matrix and rectangular
validation/test cross-kernel. Therefore the train matrix is scaled but the cross
block is not.

The paper text says that the test--train block is normalized using the same
training trace. The code/text difference should first be observed in the
preserved baseline, then discussed. A later corrected mode would need to carry
the training normalization state into every cross-kernel.

#### Local characterization on the surrogate dataset

A temporary, uncommitted diagnostic compared three otherwise identical
PneumoniaMNIST runs with `q=4`, seed 0, 100 samples, and `C=1`:

| Kernel scaling | Train minority F1 | Validation minority F1 | Test minority F1 |
|---|---:|---:|---:|
| upstream `trace` | 0.000 | 1.000 | 0.800 |
| `none` | 0.778 | 1.000 | 0.800 |
| training-trace applied consistently | 0.000 | 0.000 | 0.000 |

With consistent trace scaling, every hard prediction was the majority class,
although AUC remained 1.0. This shows that the nonzero held-out F1 of the
upstream path can be caused by the train/cross-kernel scale mismatch. It does
not establish behavior on the paper's controlled embeddings: this was one seed
of a raw-pixel surrogate with only two minority samples in the test split.

The temporary corrected option was removed after this diagnostic. The retained
code continues to expose only the upstream normalization modes; any corrected
path requires a separate explicit decision.

### 2. MinMax scaling uses held-out data by default

**Confirmed upstream behavior, not introduced by the CPU adaptation.**

Upstream
[`data_prepare_cv`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/process.py#L9-L24)
defaults to `fix_leakage=False`. In that mode, `MinMaxScaler` is fitted to the
concatenation of train and held-out samples. The main script calls the function
once with train/validation and again with train/test
([upstream lines 265--269](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L265-L269)),
which also creates two differently scaled versions of the training data.

The upstream CLI already exposes `--fix_leakage`, but it is opt-in and described
as changing legacy behavior. The current local Table 1 launcher does not enable
it. This gives a useful future comparison requiring no rewrite: preserved
default versus the existing upstream flag.

### 3. Partial failures and aggregation

- The main QSVM script catches broad per-run exceptions and can finish single
  mode successfully without producing the expected artifact.
- The local Table 1 aggregator uses an inner join, keeps the last duplicate, and
  does not require an explicit expected seed--`q` grid.
- A partial set of runs can therefore look like a complete summary.

This combines upstream execution behavior with a local aggregation concern.
Before scientific publication, completeness should be checked outside the
preserved pipeline or in a separately approved corrected path.

### 4. Circuit description and implementation differ

- The paper describes one `Ry` per qubit followed by CNOT ring entanglement.
- The audited upstream `make_bsp` applies Hadamards, `Rz` and `Ry` rotations,
  an open CNOT chain, then another `Rz` layer.

The repository should not silently replace the upstream circuit. A later
discussion should choose between code-faithful reproduction, paper-text-faithful
reproduction, or both as explicitly named modes.

### 5. Other protocol/numerical observations

- The variable described as full training data currently contains `X_train`,
  not `X_train + X_val`; code, comments, and intended protocol need comparison.
- Small sample caps use random rather than stratified selection.
- Some preprocessing scripts use train-only transforms while others preserve
  the combined-data MinMax behavior, so their outputs should not be mixed
  without checking the exact path.
- Kernel matrices are rounded before SVC and are not systematically checked for
  finite values, shape, diagonal, symmetry, range, or PSD tolerance.
- Some CLI combinations have incompatible PCA dimensions and circuit parameter
  counts.

## Catalogue integration observations

- `configs/defaults.json`, paper-root `cli.json`, and `lib/runner.py` are absent,
  so the shared runtime cannot discover this paper.
- These markers existed in earlier local history but belonged to a much larger
  synthetic implementation. Restore only a minimal wrapper, not the old
  framework.
- The wrapper should invoke the preserved upstream path rather than move or
  rewrite its scientific logic.
- Catalogue data belong under `data/qsvm_medimage/`; raw runs belong under
  `outdir/run_*`; only small sanitized artifacts belong under `results/`.
- Existing direct scripts may remain as upstream-compatible entry points.

## Tests and dependencies

- Several tests depend on absolute `/orcd/...` paths, skip without the original
  data/GPU, or refer to a missing `scripts/qsvm_hybrid_insurance.py`.
- Some files called tests are manual diagnostics rather than collected pytest
  tests.
- There is no focused local coverage for the CPU kernel path, dataset adapter,
  cross-kernel normalization behavior, preprocessing behavior, run failure, or
  Table 1 grid completeness.
- Requirements and `pyproject.toml` diverge; `memory-profiler` is duplicated and
  Torch is imported by the package without a clear minimal CPU dependency
  contract.
- MerLin 0.4 has a newer Python/NumPy/scikit-learn stack that may need an
  isolated or reconciled environment.

Initial tests should characterize the preserved upstream behavior on a tiny
offline fixture. Tests for a corrected scientific protocol should be added only
to a separately approved mode; they must not silently change what the baseline
means.

## Provenance, documentation, and data safety

- NOTICE/README name the upstream repository but do not yet pin the imported
  revision or enumerate all substantive local changes.
- The paper and repository use non-MIT content/licenses inside a catalogue whose
  root license differs. Attribution and scope must be clarified without making
  unsupported legal conclusions.
- The gated dataset naming appears inconsistent with the paper's DT9/DT11
  terminology. Verify it before documenting a mapping.
- README and generated docs reference missing scripts/directories and upstream
  HPC results not reproduced locally.
- Notebooks and commands contain personal/HPC paths and executed metadata.
- Upstream result serialization may include raw embeddings and absolute paths.
  Curated artifacts should instead contain identifiers, labels, checksums,
  relative paths, configs, and aggregate metrics.

## CPU feasibility

The exact CPU path is approximately quadratic in sample count and exponential
in qubit count, and currently rebuilds/simulates a circuit for every pair. The
existing cap (`N <= 200`, `q <= 6`) bounds the first adaptation but may still be
slow. The user should run one tiny timing/memory pilot before any grid. No
feasibility claim should be made for the full reference dataset or MerLin Gram
matrix without such measurements.

## MerLin 0.4 conclusions

- MerLin 0.4 supports a CPU `FeatureMap` + `FidelityKernel` path returning Gram
  matrices usable by scikit-learn's precomputed-kernel SVC.
- In the installed MerLin 0.4.0, `n_photons=` remains accepted by
  `FidelityKernel`, while `FidelityKernel.simple()` is deprecated. The local
  adaptation should use `FeatureMap.simple()` and pass an explicit
  `input_state` so that modes and photon count are inspectable.
- A native photonic feature map is an adaptation, not automatically equivalent
  to the qubit BSP circuit.
- MerLin should reuse the same surrogate inputs, splits, labels, SVM settings,
  and artifact schema, while clearly naming its resource definition.
- Start only after the upstream CPU baseline is characterized and after a tiny
  user-run MerLin timing pilot.
- Installing MerLin 0.4.0 in the historical virtual environment upgraded
  scikit-learn from the project pin 1.6.1 to 1.9.0. New artifacts must record
  that environment difference; it does not alter results already generated.

## Decisions still open

1. Which first offline fixture/sample count and `q` should characterize the
   preserved CPU path?
2. Should the first real surrogate remain raw-pixel PneumoniaMNIST, or should it
   be limited to smoke while an open frozen-embedding dataset is selected?
3. After baseline tests, should normalization/preprocessing be compared using
   the existing upstream options, a new corrected mode, or both?
4. Should the circuit target the upstream implementation, the paper text, or
   expose both explicitly?
5. What exact artifacts and seed count are affordable after the timing pilot?
6. Should MerLin dependencies share the Qiskit environment or be isolated?
7. Which legacy HPC/docs/notebook files should be kept, archived, or removed?

Answers to these questions should be recorded here or in an explicit decision
section before the corresponding implementation changes.
