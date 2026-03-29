#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="out/full_delta_k_matrix_parallel_${TIMESTAMP}.txt"

launch_worker() {
  local worker_name="$1"
  local cuda_visible="$2"
  local tasks="$3"
  local log_path="out/${worker_name}_${TIMESTAMP}.log"
  local pid_path="out/${worker_name}.pid"

  nohup env \
    CUDA_VISIBLE_DEVICES="$cuda_visible" \
    DEVICE="cuda:0" \
    DRAFT_DEVICE="cuda:0" \
    ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    TASKS="$tasks" \
    stdbuf -oL -eL \
    ./scripts/run_full_delta_k_matrix_best_models.sh > "$log_path" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_path"
  {
    echo "worker=${worker_name}"
    echo "pid=${pid}"
    echo "log=${log_path}"
    echo "tasks=${tasks}"
    echo
  } >> "$MANIFEST"
}

rm -f out/full_delta_k_matrix_worker{1,2,3,4}.pid
touch "$MANIFEST"

launch_worker "full_delta_k_matrix_worker1" "0" "qwen0p6b:2 qwen0p6b:6 llama3p2_1b:5 qwen3_1p7b:4"
launch_worker "full_delta_k_matrix_worker2" "0" "qwen0p6b:3 llama3p2_1b:2 llama3p2_1b:6 qwen3_1p7b:5"
launch_worker "full_delta_k_matrix_worker3" "1" "qwen0p6b:4 llama3p2_1b:3 qwen3_1p7b:2 qwen3_1p7b:6"
launch_worker "full_delta_k_matrix_worker4" "1" "qwen0p6b:5 llama3p2_1b:4 qwen3_1p7b:3"

echo "manifest=${MANIFEST}"
