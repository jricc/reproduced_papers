# Protocol audit

## Scope

This directory keeps three distinct scopes:

1. **Reference reproduction:** not run. It requires the gated MIMIC-CXR
   embeddings, original insurance target and reference qubit protocol.
2. **Open-data CPU surrogate:** raw PneumoniaMNIST pixels evaluated with the
   local CPU QSVM and scikit-learn baselines.
3. **MerLin photonic adaptation:** a native photonic fidelity kernel evaluated
   on the open-data surrogate. It is not the paper's BSP qubit kernel.

Results from one scope must not be presented as results from another.

## Audited implementation choices

Two implementation choices are exposed without deleting the historical
behavior.

### MinMax preprocessing

- Historical mode: `legacy_train_plus_heldout`. MinMaxScaler is fitted on the
  concatenation of training and held-out PCA features.
- Corrected mode: `train_only`. MinMaxScaler is fitted on training PCA features
  only. Held-out values are not clipped and may fall outside the training
  feature range.

Classical linear and RBF baselines have these two preprocessing variants only.
They do not receive trace normalization.

### Trace scaling

- Historical QSVM mode: `legacy_square_only`. Each square training kernel is
  divided by its own trace; rectangular validation and test cross-kernels are
  unchanged.
- Corrected QSVM mode: `train_trace`. Each square training kernel and its
  matching rectangular cross-kernel are divided by the same training trace.
- Historical MerLin mode: `none`. MerLin did not have a historical
  QSVM-style square-only trace mode.
- Corrected MerLin mode: `train_trace`, applied independently to the
  train/validation-cross and train-for-test/test-cross pairs.

For hybrid QSVM kernels in `train_trace` mode, the classical and quantum
train/cross pairs are each scaled by their corresponding training trace before
mixing.

## Protocol IDs

| Protocol ID | Preprocessing | QSVM trace protocol | MerLin normalization |
|---|---|---|---|
| `legacy_leak__legacy_trace` | `legacy_train_plus_heldout` | `legacy_square_only` | `none` |
| `train_only__legacy_trace` | `train_only` | `legacy_square_only` | `none` |
| `legacy_leak__train_trace` | `legacy_train_plus_heldout` | `train_trace` | `train_trace` |
| `train_only__train_trace` | `train_only` | `train_trace` | `train_trace` |

`legacy_leak__legacy_trace` retains the historical upstream behavior.
`train_only__train_trace` is the corrected local protocol.

This matrix is a sensitivity analysis. It does not establish the original
authors' intent and does not convert the open-data surrogate into a reference
reproduction.

## Paired aggregation

[`scripts/aggregate_protocol_matrix.py`](scripts/aggregate_protocol_matrix.py)
matches results by protocol ID, seed and PCA/qubit dimension. Classical
baseline results are computed once per preprocessing mode and reused for both
trace modes.

For every row, the perspective is:

```text
delta_f1 = first-named model F1 - named baseline F1
```

W/T/L = Wins / Ties / Losses, counted seed by seed from the first-named model's
perspective.

Ties use `np.isclose` with an explicit absolute tolerance and `rtol=0`. The
aggregator writes:

- `protocol_results_per_seed.csv`;
- `protocol_summary.csv`;
- `protocol_summary.md`.

## Execution status

The historical N=500 surrogate reported in the README predates this matrix and
corresponds to `legacy_leak__legacy_trace`.

The four-protocol full run has not yet been executed. A small CPU smoke test
verified orchestration and artifact generation only; it is not a scientific
result and no smoke-test values are reported here. No corrected numerical
claim should be made until the planned full run is complete.
