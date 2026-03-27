#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="out/qwen3_1p7b_delta_dac_parallel_${TIMESTAMP}.txt"

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
    DAC_BITS_TASKS="$tasks" \
    stdbuf -oL -eL \
    ./scripts/run_qwen3_1p7b_delta_dac_sweep.sh > "$log_path" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_path"
  {
    echo "worker=${worker_name}"
    echo "pid=${pid}"
    echo "log=${log_path}"
    echo "dac_bits=${tasks}"
    echo
  } >> "$MANIFEST"
}

rm -f out/qwen3_1p7b_delta_dac_worker{1,2,3,4}.pid
touch "$MANIFEST"

launch_worker "qwen3_1p7b_delta_dac_worker1" "0" "5"
launch_worker "qwen3_1p7b_delta_dac_worker2" "0" "6"
launch_worker "qwen3_1p7b_delta_dac_worker3" "1" "7"
launch_worker "qwen3_1p7b_delta_dac_worker4" "1" "8"

echo "manifest=${MANIFEST}"
