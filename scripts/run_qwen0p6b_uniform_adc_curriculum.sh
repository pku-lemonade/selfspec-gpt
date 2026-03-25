#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_CKPT="${BASE_CKPT:-checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft_rel10_write_teacher_klmix_awp_qkvout_step50.pth}"
DEVICE="${DEVICE:-cuda:1}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sdpa}"
MAX_STEPS="${MAX_STEPS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_STEPS="${EVAL_STEPS:-20}"
SEED="${SEED:-1234}"
TEACHER_MODE="${TEACHER_MODE:-stage_input}"

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

stage_bits=(10 9 8)
stage_input="$BASE_CKPT"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing base checkpoint: $BASE_CKPT" >&2
  exit 1
fi

for idx in "${!stage_bits[@]}"; do
  stage_num=$((idx + 1))
  bits="${stage_bits[$idx]}"
  stage_tag="stage${stage_num}_adc${bits}_step${MAX_STEPS}"
  stage_out_dir="out/ft_qwen0p6b_rel10_uniform_adccurriculum_${stage_tag}"
  stage_output_path="checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft_rel10_uniform_adccurriculum_${stage_tag}.pth"

  case "$TEACHER_MODE" in
    stage_input)
      teacher_ckpt="$stage_input"
      ;;
    fixed_base)
      teacher_ckpt="$BASE_CKPT"
      ;;
    *)
      echo "Unsupported TEACHER_MODE: $TEACHER_MODE (expected stage_input or fixed_base)" >&2
      exit 1
      ;;
  esac

  echo "== Stage ${stage_num}/${#stage_bits[@]}: student_adc_bits=${bits} =="
  echo "checkpoint_path=$stage_input"
  echo "teacher_checkpoint_path=$teacher_ckpt"
  echo "output_path=$stage_output_path"

  python scripts/finetune_noise_wikitext.py \
    --checkpoint_path "$stage_input" \
    --teacher_checkpoint_path "$teacher_ckpt" \
    --device "$DEVICE" \
    --teacher_device "$TEACHER_DEVICE" \
    --attention_backend "$ATTENTION_BACKEND" \
    --seed "$SEED" \
    --lr "$LR" \
    --output_path "$stage_output_path" \
    --save_dir "$stage_out_dir" \
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

  stage_input="$stage_output_path"
done

echo "Final curriculum checkpoint: $stage_input"
