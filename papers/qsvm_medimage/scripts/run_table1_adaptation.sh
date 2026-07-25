#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-../../data/qsvm_medimage/pneumoniamnist_train.pkl}"
RESULT_ROOT="${RESULT_ROOT:-outdir/table1_adaptation}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
Q_VALUES_CSV="${Q_VALUES:-2,4,6}"
SEEDS_CSV="${SEEDS:-0,1,2,3,4}"
C_VALUES="${C_VALUES:-0.01,0.1,1,10,100}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qsvm-medimage-matplotlib}"

export MPLCONFIGDIR
export PYTHONDONTWRITEBYTECODE=1

IFS=',' read -r -a Q_VALUES_ARRAY <<< "$Q_VALUES_CSV"
IFS=',' read -r -a SEEDS_ARRAY <<< "$SEEDS_CSV"

for q_value in "${Q_VALUES_ARRAY[@]}"; do
  for seed_value in "${SEEDS_ARRAY[@]}"; do
    echo "QSVM: q=$q_value seed=$seed_value"
    "$PYTHON_BIN" scripts/qsvm_cuda_embeddings_insurance.py \
      --data_path "$DATA_PATH" \
      --output_dir "$RESULT_ROOT/qsvm/q_${q_value}/seed_${seed_value}" \
      --backend cpu \
      --qubits "$q_value" \
      --max_samples "$MAX_SAMPLES" \
      --single_mode \
      --num_seeds 1 \
      --seed "$seed_value"
  done
done

"$PYTHON_BIN" scripts/classical_svm_c1_pca.py \
  --data_path "$DATA_PATH" \
  --output_dir "$RESULT_ROOT/classical_c1_multi" \
  --pca_dims "$Q_VALUES_CSV" \
  --kernels linear \
  --c_values 1.0 \
  --seeds "$SEEDS_CSV" \
  --max_samples "$MAX_SAMPLES"

"$PYTHON_BIN" scripts/classical_svm_c1_pca.py \
  --data_path "$DATA_PATH" \
  --output_dir "$RESULT_ROOT/classical_tuned_multi" \
  --pca_dims "$Q_VALUES_CSV" \
  --kernels rbf \
  --c_values "$C_VALUES" \
  --seeds "$SEEDS_CSV" \
  --max_samples "$MAX_SAMPLES"

"$PYTHON_BIN" scripts/create_table1_adaptation.py \
  --qsvm_dirs "$RESULT_ROOT/qsvm" \
  --tier1_csv "$RESULT_ROOT/classical_c1_multi/metrics_summary.csv" \
  --tier2_csv "$RESULT_ROOT/classical_tuned_multi/metrics_summary.csv" \
  --output "$RESULT_ROOT/table1_cpu_pneumoniamnist.png"

echo "Table I adaptation: $RESULT_ROOT/table1_cpu_pneumoniamnist.png"
