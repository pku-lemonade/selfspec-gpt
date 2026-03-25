#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_write_cons_step400.pth}"
DEVICE="${DEVICE:-cuda:1}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:0}"
TRAIN_ATTENTION_BACKEND="${TRAIN_ATTENTION_BACKEND:-sdpa}"
EVAL_ATTENTION_BACKEND="${EVAL_ATTENTION_BACKEND:-flex}"
SEED="${SEED:-1234}"

MAX_STEPS="${MAX_STEPS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_STEPS="${EVAL_STEPS:-20}"

LR="${LR:-2e-5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
DISTILL_KL_WEIGHT="${DISTILL_KL_WEIGHT:-1.0}"
NOISY_CE_WEIGHT="${NOISY_CE_WEIGHT:-0.05}"

WRITE_NOISE_FFN="${WRITE_NOISE_FFN:-0.1}"
WRITE_NOISE_QKV="${WRITE_NOISE_QKV:-0.1}"
WRITE_NOISE_OUT="${WRITE_NOISE_OUT:-0.1}"

PROB_CLEAN="${PROB_CLEAN:-0.2}"
PROB_WRITE="${PROB_WRITE:-0.8}"
PROB_READ="${PROB_READ:-0.0}"
PROB_BOTH="${PROB_BOTH:-0.0}"

PROMPTS="${PROMPTS:-out/wikitext103_test_prompts_300.txt}"
PILOT_LIMIT="${PILOT_LIMIT:-20}"
PILOT_PROMPT_LENGTH="${PILOT_PROMPT_LENGTH:-64}"
PILOT_MAX_NEW_TOKENS="${PILOT_MAX_NEW_TOKENS:-32}"
SPECULATE_K="${SPECULATE_K:-5}"
VERIFY_ADC_BITS="${VERIFY_ADC_BITS:-12}"
DRAFT_ADC_BITS="${DRAFT_ADC_BITS:-6}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_METHOD1="${RUN_METHOD1:-1}"
RUN_METHOD2="${RUN_METHOD2:-1}"
RUN_METHOD3="${RUN_METHOD3:-1}"
RUN_METHOD4="${RUN_METHOD4:-1}"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing base checkpoint: $BASE_CKPT" >&2
  exit 1
fi

run_eval() {
  local ckpt_path="$1"
  local out_dir="$2"
  shift 2
  python scripts/dataset_selfspec_stats.py \
    --checkpoint_path "$ckpt_path" \
    --draft_checkpoint_path "$ckpt_path" \
    --device "$DEVICE" \
    --draft_device "$DEVICE" \
    --attention_backend "$EVAL_ATTENTION_BACKEND" \
    --prompts "$PROMPTS" \
    --limit "$PILOT_LIMIT" \
    --prompt_lengths "$PILOT_PROMPT_LENGTH" \
    --max_new_tokens "$PILOT_MAX_NEW_TOKENS" \
    --speculate_k "$SPECULATE_K" \
    --temperature 0 \
    --top_k 200 \
    --num_samples 1 \
    --seed "$SEED" \
    --draft_noise_std "$WRITE_NOISE_FFN" "$WRITE_NOISE_QKV" "$WRITE_NOISE_OUT" \
    --verify_adc_bits "$VERIFY_ADC_BITS" \
    --draft_adc_bits "$DRAFT_ADC_BITS" \
    --out_dir "$out_dir" \
    "$@"
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  run_eval \
    "$BASE_CKPT" \
    "out/corrected_qwen3_1p7b_oldtuned_targetself_adc6_s20_flex_cuda1_rerun"
fi

if [[ "$RUN_METHOD1" == "1" ]]; then
  METHOD1_CKPT="checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adc6_cleanteacher_step50.pth"
  METHOD1_SAVE="out/ft_qwen3_1p7b_rel10_uniform_adc6_cleanteacher_step50"
  python scripts/finetune_noise_wikitext.py \
    --checkpoint_path "$BASE_CKPT" \
    --teacher_checkpoint_path "$BASE_CKPT" \
    --device "$DEVICE" \
    --teacher_device "$TEACHER_DEVICE" \
    --attention_backend "$TRAIN_ATTENTION_BACKEND" \
    --seed "$SEED" \
    --lr "$LR" \
    --output_path "$METHOD1_CKPT" \
    --save_dir "$METHOD1_SAVE" \
    --max_steps "$MAX_STEPS" \
    --save_interval "$SAVE_INTERVAL" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_steps "$EVAL_STEPS" \
    --write_noise_std "$WRITE_NOISE_FFN" "$WRITE_NOISE_QKV" "$WRITE_NOISE_OUT" \
    --read_noise_std 0.0 \
    --prob_clean "$PROB_CLEAN" \
    --prob_write "$PROB_WRITE" \
    --prob_read "$PROB_READ" \
    --prob_both "$PROB_BOTH" \
    --student_adc_bits "$DRAFT_ADC_BITS" \
    --teacher_adc_bits 0 \
    --distill_kl_weight "$DISTILL_KL_WEIGHT" \
    --distill_temperature "$DISTILL_TEMPERATURE" \
    --noisy_ce_weight "$NOISY_CE_WEIGHT"
  run_eval \
    "$METHOD1_CKPT" \
    "out/corrected_qwen3_1p7b_uniform_adc6_cleanteacher_step50_s20_flex_cuda1"
fi

if [[ "$RUN_METHOD2" == "1" ]]; then
  stage_bits=(8 7 6)
  stage_input="$BASE_CKPT"
  for idx in "${!stage_bits[@]}"; do
    stage_num=$((idx + 1))
    bits="${stage_bits[$idx]}"
    stage_tag="stage${stage_num}_adc${bits}_step${MAX_STEPS}"
    stage_ckpt="checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adccurriculum_${stage_tag}.pth"
    stage_save="out/ft_qwen3_1p7b_rel10_uniform_adccurriculum_${stage_tag}"
    python scripts/finetune_noise_wikitext.py \
      --checkpoint_path "$stage_input" \
      --teacher_checkpoint_path "$stage_input" \
      --device "$DEVICE" \
      --teacher_device "$TEACHER_DEVICE" \
      --attention_backend "$TRAIN_ATTENTION_BACKEND" \
      --seed "$SEED" \
      --lr "$LR" \
      --output_path "$stage_ckpt" \
      --save_dir "$stage_save" \
      --max_steps "$MAX_STEPS" \
      --save_interval "$SAVE_INTERVAL" \
      --eval_interval "$EVAL_INTERVAL" \
      --eval_steps "$EVAL_STEPS" \
      --write_noise_std "$WRITE_NOISE_FFN" "$WRITE_NOISE_QKV" "$WRITE_NOISE_OUT" \
      --read_noise_std 0.0 \
      --prob_clean "$PROB_CLEAN" \
      --prob_write "$PROB_WRITE" \
      --prob_read "$PROB_READ" \
      --prob_both "$PROB_BOTH" \
      --student_adc_bits "$bits" \
      --teacher_adc_bits 0 \
      --distill_kl_weight "$DISTILL_KL_WEIGHT" \
      --distill_temperature "$DISTILL_TEMPERATURE" \
      --noisy_ce_weight "$NOISY_CE_WEIGHT"
    stage_input="$stage_ckpt"
  done
  run_eval \
    "$stage_input" \
    "out/corrected_qwen3_1p7b_uniform_adccurriculum_stage3_adc6_step50_s20_flex_cuda1"
fi

if [[ "$RUN_METHOD4" == "1" ]]; then
  CHECKPOINT_PATH="$BASE_CKPT" \
  TEACHER_CHECKPOINT_PATH="$BASE_CKPT" \
  OUTPUT_PATH="checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adc6_stochastic_step50.pth" \
  SAVE_DIR="out/ft_qwen3_1p7b_rel10_uniform_adc6_stochastic_step50" \
  DEVICE="$DEVICE" \
  TEACHER_DEVICE="$TEACHER_DEVICE" \
  ATTENTION_BACKEND="$TRAIN_ATTENTION_BACKEND" \
  SEED="$SEED" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_INTERVAL="$SAVE_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  EVAL_STEPS="$EVAL_STEPS" \
  LR="$LR" \
  DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
  DISTILL_KL_WEIGHT="$DISTILL_KL_WEIGHT" \
  NOISY_CE_WEIGHT="$NOISY_CE_WEIGHT" \
  WRITE_NOISE_FFN="$WRITE_NOISE_FFN" \
  WRITE_NOISE_QKV="$WRITE_NOISE_QKV" \
  WRITE_NOISE_OUT="$WRITE_NOISE_OUT" \
  PROB_CLEAN="$PROB_CLEAN" \
  PROB_WRITE="$PROB_WRITE" \
  PROB_READ="$PROB_READ" \
  PROB_BOTH="$PROB_BOTH" \
  STUDENT_ADC_BITS="$DRAFT_ADC_BITS" \
  ./scripts/run_qwen0p6b_uniform_adc_stochastic.sh
  run_eval \
    "checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adc6_stochastic_step50.pth" \
    "out/corrected_qwen3_1p7b_uniform_adc6_stochastic_step50_s20_flex_cuda1"
fi

if [[ "$RUN_METHOD3" == "1" ]]; then
  CHECKPOINT_PATH="$BASE_CKPT" \
  TEACHER_CHECKPOINT_PATH="$BASE_CKPT" \
  OUTPUT_PATH="checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adc6_learnableclip_step50.pth" \
  SAVE_DIR="out/ft_qwen3_1p7b_rel10_uniform_adc6_learnableclip_step50" \
  DEVICE="$DEVICE" \
  TEACHER_DEVICE="$TEACHER_DEVICE" \
  ATTENTION_BACKEND="$TRAIN_ATTENTION_BACKEND" \
  SEED="$SEED" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_INTERVAL="$SAVE_INTERVAL" \
  EVAL_INTERVAL="$EVAL_INTERVAL" \
  EVAL_STEPS="$EVAL_STEPS" \
  LR="$LR" \
  DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
  DISTILL_KL_WEIGHT="$DISTILL_KL_WEIGHT" \
  NOISY_CE_WEIGHT="$NOISY_CE_WEIGHT" \
  WRITE_NOISE_FFN="$WRITE_NOISE_FFN" \
  WRITE_NOISE_QKV="$WRITE_NOISE_QKV" \
  WRITE_NOISE_OUT="$WRITE_NOISE_OUT" \
  PROB_CLEAN="$PROB_CLEAN" \
  PROB_WRITE="$PROB_WRITE" \
  PROB_READ="$PROB_READ" \
  PROB_BOTH="$PROB_BOTH" \
  STUDENT_ADC_BITS="$DRAFT_ADC_BITS" \
  ./scripts/run_qwen0p6b_uniform_adc_learnable_clip.sh
  METHOD3_CLIP="$(
    python - <<'PY'
import json
with open('out/ft_qwen3_1p7b_rel10_uniform_adc6_learnableclip_step50/summary.json', 'r', encoding='utf-8') as f:
    print(json.load(f)['student_adc_final_clip_scale'])
PY
  )"
  run_eval \
    "checkpoints/Qwen/Qwen3-1.7B/model_wikitext_noise_ft_rel10_uniform_adc6_learnableclip_step50.pth" \
    "out/corrected_qwen3_1p7b_uniform_adc6_learnableclip_step50_s20_flex_cuda1" \
    --draft_adc_clip_scale "$METHOD3_CLIP"
fi
