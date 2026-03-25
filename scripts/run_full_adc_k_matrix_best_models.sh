#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-$DEVICE}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex}"
PROMPTS="${PROMPTS:-out/wikitext103_test_prompts_300.txt}"
LIMIT="${LIMIT:-100}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-1234}"
VERIFY_ADC_BITS="${VERIFY_ADC_BITS:-12}"
K_VALUES="${K_VALUES:-2 3 4 5 6 7 8 9 10 11 12}"
TASKS="${TASKS:-}"

if [[ ! -f "$PROMPTS" ]]; then
  echo "Missing prompts file: $PROMPTS" >&2
  exit 1
fi

run_sweep() {
  local model_tag="$1"
  local checkpoint_path="$2"
  local draft_adc_bits="$3"
  local out_json="out/k_sweep_${model_tag}_best_adc${draft_adc_bits}_12_s100_cuda1.json"

  if [[ -f "$out_json" ]]; then
    echo "[skip] ${out_json} already exists"
    return 0
  fi

  echo "[run] model=${model_tag} draft_adc_bits=${draft_adc_bits} verify_adc_bits=${VERIFY_ADC_BITS}"
  python scripts/sweep_speculate_k.py \
    --checkpoint_path "$checkpoint_path" \
    --draft_checkpoint_path "$checkpoint_path" \
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
    --draft_adc_bits "$draft_adc_bits" \
    --out_json "$out_json"
}

resolve_checkpoint() {
  local model_tag="$1"
  case "$model_tag" in
    qwen0p6b)
      echo "checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft_rel10_write_teacher_klmix_awp_qkvout_step50.pth"
      ;;
    llama3p2_1b)
      echo "checkpoints/meta-llama/Llama-3.2-1B/model_wikitext_noise_ft_rel10_adc8_12_selftarget_qkvout_step50.pth"
      ;;
    qwen3_1p7b)
      echo "checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_write_cons_step400.pth"
      ;;
    *)
      echo "Unknown model_tag: $model_tag" >&2
      return 1
      ;;
  esac
}

ALL_TASKS=(
  "qwen0p6b:4"
  "qwen0p6b:5"
  "qwen0p6b:6"
  "qwen0p6b:7"
  "qwen0p6b:8"
  "llama3p2_1b:4"
  "llama3p2_1b:5"
  "llama3p2_1b:6"
  "llama3p2_1b:7"
  "llama3p2_1b:8"
  "qwen3_1p7b:4"
  "qwen3_1p7b:5"
  "qwen3_1p7b:6"
  "qwen3_1p7b:7"
  "qwen3_1p7b:8"
)

if [[ -n "$TASKS" ]]; then
  read -r -a TASK_LIST <<< "$TASKS"
else
  TASK_LIST=("${ALL_TASKS[@]}")
fi

for task in "${TASK_LIST[@]}"; do
  model_tag="${task%%:*}"
  draft_adc_bits="${task##*:}"
  checkpoint_path="$(resolve_checkpoint "$model_tag")"
  run_sweep "$model_tag" "$checkpoint_path" "$draft_adc_bits"
done

echo "[done] full ADC-by-k matrix queue finished"
