# Protocol audit

## Scope

This audit keeps three scopes separate:

1. **Reference reproduction:** not run; it requires the gated MIMIC-CXR
   embeddings, insurance target, and reference BSP qubit protocol.
2. **Open-data CPU surrogate:** PneumoniaMNIST pixels evaluated with the local
   CPU QSVM and scikit-learn baselines.
3. **MerLin photonic adaptation:** a separate local photonic fidelity kernel
   evaluated on the same open-data splits. It is not the BSP qubit kernel.

The audit concerns protocol sensitivity in the local surrogate. It does not
transfer results between these scopes.

## Exact paper requirement

[Section III.C of the paper](https://arxiv.org/html/2604.24597v1) defines the
compute-uncompute fidelity kernel in Equation (2) and trace normalization in
Equation (3). The text immediately following Equation (3) requires the
test--train cross-kernel to be divided by the same trace as the square training
kernel. The normalization factor is therefore determined on training data and
reused for the associated held-out cross-kernel.

Section III.D defines the two-tier comparison: the fixed-`C` QSVM is compared
first with the untuned linear baseline and then with a validation-tuned RBF
baseline. Section III.E makes minority-class F1 the primary metric.

The paper describes common input preprocessing but does not present MinMax
scaling as depending on held-out extrema.

## Exact imported upstream behavior

The audited upstream revision is
`9e80037305d683b0e70c94b8fa7dd648e1bac82b`.

- [`data_prepare_cv`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/process.py#L6-L21)
  defaults to `fix_leakage=False`. StandardScaler and PCA are fitted on
  training data, but MinMaxScaler is fitted on concatenated training and
  held-out PCA features.
- [`normalize_kernel_trace`](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/qve/core.py#L325-L341)
  divides a square matrix by its trace and returns a rectangular matrix
  unchanged.
- The QSVM script [calls that helper separately on the square training kernel
  and rectangular cross-kernel](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L732-L761).
  The second call is consequently a no-op.
- [`apply_hybrid_kernel` is used for both validation and test
  pairs](https://github.com/sebasmos/qml-medimage/blob/9e80037305d683b0e70c94b8fa7dd648e1bac82b/scripts/qsvm_cuda_embeddings_insurance.py#L545-L555),
  so the same train/cross scale mismatch affects both held-out paths.

| Issue | Upstream behavior | Paper behavior | Local impact |
|---|---|---|---|
| MinMax fitting | training + held-out | not presented as held-out-dependent | negligible here, still leakage |
| Cross-kernel trace | rectangular cross-kernel unchanged | same training trace used | decisive for local QSVM F1 |
| MerLin | absent | absent | separate local adaptation |

Fitting MinMaxScaler on held-out data is leakage even though its measured
effect is small here. Leaving the cross-kernel unscaled contradicts the written
trace-normalization protocol. Neither observation establishes why the upstream
code was written this way or what the authors intended.

## Exact local protocol matrix

The local run is a 2×2 factorial comparison of preprocessing and quantum-kernel
scaling:

- `legacy_train_plus_heldout`: MinMaxScaler sees training and held-out PCA
  features;
- `train_only`: MinMaxScaler is fitted only on training PCA features, without
  clipping transformed held-out values;
- QSVM `legacy_square_only`: each square training kernel is divided by its own
  trace and its rectangular cross-kernel is unchanged;
- QSVM `train_trace`: a square training kernel and its associated rectangular
  cross-kernel are divided by the same training-kernel trace.

| Protocol ID | Preprocessing | QSVM trace protocol | MerLin normalization |
|---|---|---|---|
| `legacy_leak__legacy_trace` | `legacy_train_plus_heldout` | `legacy_square_only` | `none` |
| `train_only__legacy_trace` | `train_only` | `legacy_square_only` | `none` |
| `legacy_leak__train_trace` | `legacy_train_plus_heldout` | `train_trace` | `train_trace` |
| `train_only__train_trace` | `train_only` | `train_trace` | `train_trace` |

These common launcher IDs describe matched rows. For MerLin,
`legacy_trace` maps to `kernel_normalization=none`; this is only a launcher
mapping, not a mode inherited from upstream.

The completed PneumoniaMNIST CPU surrogate uses at most 500 samples, a
400/50/50 train/validation/test split, `q` in `{4, 6}`, and paired data/split
seeds 0 through 9. The MerLin circuit seed is fixed at 0. This gives four
protocol IDs, two dimensions, two quantum models, two baselines, and 32
aggregated rows.

Classical linear and RBF results are computed once per preprocessing mode and
reused across the two trace rows because trace scaling applies only to the
precomputed quantum kernels. The QSVM and linear SVM use `C=1`; the RBF `C` is
selected by validation minority F1 from the unchanged configured grid.

For hybrid QSVM kernels under `train_trace`, the classical and quantum
train/cross pairs are each divided by their corresponding training trace before
mixing. Training/validation and training-for-test/test pairs are normalized
independently; no trace is taken on a rectangular matrix.

The local factorial comparison isolates the two implemented code-path effects
on fixed data/split seeds. It does not isolate every possible interaction with
dataset choice, feature representation, `C`, or qubit dimension.

## Main findings

1. **MinMax leakage is empirically negligible in this surrogate.** Across
   matching leaky and train-only aggregate rows, the largest minority-F1 change
   is below 0.005. It remains methodologically invalid leakage.
2. **Cross-kernel trace scaling is decisive in this surrogate.** With QSVM
   `train_trace`, mean minority F1 and its standard deviation are zero at both
   `q=4` and `q=6`, under both preprocessing variants, on every paired seed.
3. **The controlled QSVM difference is held-out kernel scale.** The square
   training kernel is trace-normalized in both compared trace modes. The modes
   differ in whether the matching validation/test cross-kernel is divided by
   that training trace. The favorable held-out scores of the preserved
   upstream path therefore depend on this train/inference scale mismatch in
   the local surrogate.
4. **The paper's linear-collapse-avoidance mechanism is not reproduced.** The
   local linear SVM has non-zero minority F1.

Detailed seed-level and aggregate values are curated under
[`results/protocol_matrix_n500_q4_q6/`](results/protocol_matrix_n500_q4_q6/).
The historical upstream-protocol-only records remain in
[`results/q4_q6_n500_per_seed.csv`](results/q4_q6_n500_per_seed.csv).

## MerLin framing

MerLin is absent from both the paper and the imported upstream repository. The
local implementation is a native photonic fidelity-kernel adaptation, not a
translation or resource-matched implementation of the BSP circuit.

The evaluated names are:

- **MerLin unnormalized variant:** `kernel_normalization=none`;
- **MerLin train-trace variant:** the matching train and cross matrices are
  divided by the same training trace.

With no trace normalization, MerLin is in the same broad performance range as
the local QSVM and classical baselines. With train-trace normalization and
fixed `C=1`, its minority F1 is zero on every evaluated seed, matching the
qualitative QSVM sensitivity. This supports consistency of the local scale
comparison. It does not establish quantum advantage or an optimized MerLin
result.

Tuning `C`, varying `circuit_seed`, and exploring the photonic feature map,
modes, photon count, normalization, and kernel scale remain future work.

## Remaining uncertainties

- The gated MIMIC-CXR embeddings and insurance task were not evaluated, so the
  reference result is unknown locally.
- The audit establishes neither author intent nor whether the original
  MIMIC-CXR claims are valid or invalid.
- Dataset, target, raw-pixel representation, sample count, and evaluated `q`
  range all differ from the paper.
- Only `q=4` and `q=6` and 50 test samples per seed were evaluated.
- Local seeds control subset selection and splitting, not the paper's
  seed-specific embeddings.
- No local paired significance test was performed.
- Fixed-`C` behavior is sensitive to absolute kernel scale; joint normalization
  and `C` sensitivity was not evaluated here.
- The paper/upstream BSP circuit differences documented elsewhere in the
  repository are not resolved by this factorial comparison.

These limitations prevent the local protocol audit from validating or
invalidating the paper's reference claims.
