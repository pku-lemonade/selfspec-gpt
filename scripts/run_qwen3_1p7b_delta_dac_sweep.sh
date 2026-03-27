#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-$DEVICE}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_write_cons_step400.pth}"
PROMPTS="${PROMPTS:-out/wikitext103_test_prompts_300.txt}"
LIMIT="${LIMIT:-100}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-1234}"
K_VALUES="${K_VALUES:-5}"
DRAFT_ADC_BITS="${DRAFT_ADC_BITS:-4}"
VERIFY_ADC_BITS="${VERIFY_ADC_BITS:-8}"
DAC_BITS_TASKS="${DAC_BITS_TASKS:-}"

if [[ ! -f "$PROMPTS" ]]; then
  echo "Missing prompts file: $PROMPTS" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Missing checkpoint: $CHECKPOINT_PATH" >&2
  exit 1
fi

run_sweep() {
  local dac_bits="$1"
  local out_json="out/compare_qwen3_1p7b_adc${DRAFT_ADC_BITS}_k${K_VALUES// /_}_verify${VERIFY_ADC_BITS}_delta_dac${dac_bits}.json"

  if [[ -f "$out_json" ]]; then
    echo "[skip] ${out_json} already exists"
    return 0
  fi

  echo "[run] model=qwen3_1p7b draft_adc_bits=${DRAFT_ADC_BITS} verify_adc_bits=${VERIFY_ADC_BITS} verify_delta_dac_bits=${dac_bits} k_values=${K_VALUES}"
  python scripts/sweep_speculate_k.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --draft_checkpoint_path "$CHECKPOINT_PATH" \
    --device "$DEVICE" \
    --draft_device "$DRAFT_DEVICE" \
    --attention_backend "$ATTENTION_BACKEND" \
    --prompts "$PROMPTS" \
    --limit "$LIMIT" \
    --prompt_length "$PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 0 \
    --top_k 200 \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --k_values $K_VALUES \
    --draft_noise_std 0.1 0.1 0.1 \
    --verify_adc_bits "$VERIFY_ADC_BITS" \
    --verify_delta_readout \
    --verify_delta_dac_bits "$dac_bits" \
    --draft_adc_bits "$DRAFT_ADC_BITS" \
    --out_json "$out_json"
}

if [[ -n "$DAC_BITS_TASKS" ]]; then
  read -r -a TASK_LIST <<< "$DAC_BITS_TASKS"
else
  TASK_LIST=(5 6 7 8)
fi

for dac_bits in "${TASK_LIST[@]}"; do
  run_sweep "$dac_bits"
done

echo "[done] qwen3_1p7b delta-DAC sweep queue finished"
