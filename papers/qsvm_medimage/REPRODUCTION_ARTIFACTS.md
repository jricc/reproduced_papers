# Reproduction artifacts — Tables 1–10 and Figures 2–5

This document is the coverage map and regeneration guide for every table and
figure of **arXiv:2604.24597** that this repository reproduces. Each paper item
has one dedicated script under `utils/` (`synthetic_surrogate_<item>.py`) that
writes `.md` + `.json` + `.csv`/`.png` outputs into `results/`.

> **Scope note.** With `--source synthetic` / `--source synthetic_file` these
> scripts reproduce the **paper's protocol shape** on a clearly-labelled
> foundation-embedding *substitute*, because the real MIMIC-CXR embeddings are
> gated (PhysioNet DUA, ~80 GB) and were never accessed. They do **not**
> reproduce the paper's absolute numbers on synthetic data. Each script keeps a
> `--source real --data-root <dir>` path that runs the identical pipeline on the
> gated embeddings for credentialed users; only the data source changes.

## One command to regenerate everything

```bash
cd papers/qsvm_medimage

# 1. Materialize the fixed synthetic dataset once (deterministic).
python utils/generate_synthetic_dataset.py \
  --output-root data/synthetic_qml_mimic_cxr_embeddings \
  --models synthetic_medsiglip,synthetic_raddino,synthetic_vit \
  --seeds 0,1,2,3,4,5,6,7,8,9

# 2. Regenerate all 10 tables + 4 figures from that fixed dataset.
python utils/generate_artifacts_from_dataset.py \
  --source synthetic_file \
  --data-root data/synthetic_qml_mimic_cxr_embeddings \
  --results-dir results \
  --only all
```

`--only` also accepts a subset, e.g. `--only figure4,table5,table10`.
For the credentialed paper-accurate path, swap the two flags:
`--source real --data-root $QML_DATA_ROOT`.

Each script can also be run standalone, e.g.
`python utils/synthetic_surrogate_figure4.py --source synthetic --results-dir results`.

## Coverage map (paper item → script → protocol)

| Paper item | What the paper shows | Script (`utils/…`) | Protocol reproduced | Output prefix (`results/…`) |
|---|---|---|---|---|
| **Table 1** | Two-tier framework: Tier-1 (C=1 untuned) 18/18 F1 wins; Tier-2 (QSVM C=1 vs C-tuned RBF) 7/7 F1 wins | `synthetic_surrogate_table1.py` | QSVM C=1 vs linear C=1 (Tier 1, 18 configs) and vs best-C RBF (Tier 2, 7 configs); reports wins/total + mean F1 gain | `*_table1` |
| **Table 2** | Extended Tier-1 across 3 models, q∈{4,6,8,9,10,11,12,16}, acc + minority-F1 over seeds | `synthetic_surrogate_table2.py` | Per-config mean/std accuracy and F1, QSVM C=1 vs linear C=1, over the 18 Tier-1 configs | `*_table2` |
| **Table 3** | Confusion matrix, MedSigLIP-448 QSVM q=11, seed 0 (paper N=238 test) | `synthetic_surrogate_table3.py` | One MedSigLIP q=11 seed-0 QSVM C=1 test confusion matrix + precision/recall | `*_table3` |
| **Table 4** | Tier-2 F1 advantage, QSVM C=1 vs C-tuned RBF, MedSigLIP & RAD-DINO q∈{4,6,8} | `synthetic_surrogate_table4.py` | QSVM C=1 vs best-C RBF over the 7 Tier-2 configs | `*_table4` |
| **Table 5** | Kernel effective rank at PCA-q, N=1,896; linear≈q, quantum 6.86–92.13 | `synthetic_surrogate_table5.py` | PCA variance, linear-kernel rank, linear eff-rank, quantum eff-rank per q | `*_table5` |
| **Table 6** | Linear-kernel variance on 200 subsampled training samples sorted by class | `synthetic_surrogate_table6.py` | Linear-kernel mean/std/variance on a 200-sample class-sorted subset | `*_table6` |
| **Table 7** | Four kernel normalizations at q=8: trace best, Frobenius collapses to F1=0 | `synthetic_surrogate_table7.py` | QSVM q=8 under none/trace/frobenius/cosine normalization | `*_table7` |
| **Table 8** | 1-DOF vs 3-DOF circuit at q=8 (1-DOF wins) | `synthetic_surrogate_table8.py` | QSVM q=8, reps=1, trace-norm, 1-DOF vs 3-DOF feature map | `*_table8` |
| **Table 9** | q=16 C-tuning: MedSigLIP collapses, RAD-DINO/ViT improve | `synthetic_surrogate_table9.py` | q=16 QSVM, best-C by validation F1, per model | `*_table9` |
| **Table 10** | Rank-matched RBF vs QSVM, q∈{4,6,11,16}, 10 seeds | `synthetic_surrogate_table10.py` | QSVM vs eff-rank-matched RBF, MedSigLIP, collapse = F1<0.05 | `*_table10` |
| **Figure 2** | Linear-kernel eigenspectrum, MedSigLIP q=6 (eff rank 5.53, exactly 6 positive eigenvalues) | `synthetic_surrogate_figure2.py` | Linear PCA-q kernel eigenspectrum at q=6, seed 0 | `*_figure2` |
| **Figure 3** | Quantum vs linear eigenspectrum, MedSigLIP q∈{4,6} | `synthetic_surrogate_figure3.py` | Quantum fidelity-kernel vs linear-kernel eigenvalue decay at q=4,6 | `*_figure3` |
| **Figure 4** | Quantum kernel heatmap (trace-normalized), MedSigLIP q=6, 200 samples sorted by class | `synthetic_surrogate_figure4.py` | Trace-normalized fidelity kernel on 200 class-sorted samples at q=6; class boundary drawn | `*_figure4` |
| **Figure 5** | Partial qubit sweep q∈{2,3,4,5,6,8}, seed 0, acc + minority-F1 for all three models | `synthetic_surrogate_figure5.py` | QSVM C=1 trace-norm sweep over q for MedSigLIP, RAD-DINO, ViT-p32 | `*_figure5` |

**Figure 1** (not in the 2–5 range requested) is the preprocessing schematic
(StandardScaler → PCA-q → MinMaxScaler[−1,1]); it is realized in code as
`lib/svm_pipeline.py::preprocess`, exercised by every script above.

## Photonic (MerLin) variants

`utils/photonic_artifacts.py` recomputes the three central claims with the
**photonic** (MerLin two-photon linear-optical) fidelity kernel
(`lib/photonic_kernel.py`) instead of the qubit gate-based kernel, so the two
backends can be compared on the same surrogate data. Train/test are
stratified-subsampled (`--train-cap`/`--test-cap`, default 220/100) because the
SLOS Gram matrix is O(N²); the caps are recorded in every output.

| Photonic artifact | Paper counterpart | Prefix |
|---|---|---|
| Tier-1 F1: linear vs qubit-QSVM vs **photonic-QSVM** per (model, q) | Table 1 | `photonic_table1` |
| Class-sorted photonic Gram-matrix heatmap | Figure 4 | `photonic_figure4` |
| Photonic-kernel effective rank vs q | Figure 2 / Table 5 | `photonic_effrank` |

Regenerate: `python utils/photonic_artifacts.py --source synthetic_file
--data-root data/synthetic_qml_mimic_cxr_embeddings --seeds 0,1,2`. Result on the
v5 surrogate: photonic QSVM beats the collapsing linear SVM 10/10 configs (mean
+0.28 F1), sits ~0.09 F1 below the qubit QSVM, and its effective rank grows with
q (15→112 for q=4→10).

## Output-prefix convention

The prefix depends on `--source`: `synthetic` → `synthetic_surrogate_*`,
`synthetic_file` → `synthetic_file_*`, `real` → `real_*`. So the same Figure 4
script writes `results/synthetic_surrogate_figure4.png`,
`results/synthetic_file_figure4.png`, or `results/real_figure4.png`.

## Tests

Each artifact has a unit test in `tests/test_synthetic_surrogate_<item>.py`
covering its helper logic. Run `pytest -q` from `papers/qsvm_medimage`.
