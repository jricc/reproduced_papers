# PneumoniaMNIST CPU surrogate protocol matrix

This directory contains curated results from the open-data PneumoniaMNIST CPU
surrogate. It is not a reproduction on the reference MIMIC-CXR dataset.

- Maximum samples: 500
- Split: 400 training, 50 validation, 50 test
- Qubit/PCA dimensions: q=4 and q=6
- Data/split seeds: 0 through 9
- MerLin circuit seed: fixed at 0
- Protocol IDs:
  - `legacy_leak__legacy_trace`
  - `train_only__legacy_trace`
  - `legacy_leak__train_trace`
  - `train_only__train_trace`

Curated artifacts:

- `protocol_results_per_seed.csv`
- `protocol_summary.csv`
- `protocol_summary.md`

The user executed these experiments. Neither Codex nor the assistant executed
them.

The existing `../q4_q6_n500_per_seed.csv` is retained unchanged as the
historical upstream-protocol-only result.
