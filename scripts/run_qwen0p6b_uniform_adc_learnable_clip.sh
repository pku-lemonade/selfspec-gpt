#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft_rel10_write_teacher_klmix_awp_qkvout_step50.pth}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-$CHECKPOINT_PATH}"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft_rel10_uniform_adc8_learnableclip_step50.pth}"
SAVE_DIR="${SAVE_DIR:-out/ft_qwen0p6b_rel10_uniform_adc8_learnableclip_step50}"

DEVICE="${DEVICE:-cuda:1}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda:0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sdpa}"
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

STUDENT_ADC_BITS="${STUDENT_ADC_BITS:-8}"
STUDENT_ADC_INIT_CLIP_SCALE="${STUDENT_ADC_INIT_CLIP_SCALE:-1.0}"

python scripts/finetune_noise_wikitext.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --teacher_checkpoint_path "$TEACHER_CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --teacher_device "$TEACHER_DEVICE" \
  --attention_backend "$ATTENTION_BACKEND" \
  --seed "$SEED" \
  --lr "$LR" \
  --output_path "$OUTPUT_PATH" \
  --save_dir "$SAVE_DIR" \
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
  --student_adc_bits "$STUDENT_ADC_BITS" \
  --teacher_adc_bits 0 \
  --student_adc_learnable_clip \
  --student_adc_init_clip_scale "$STUDENT_ADC_INIT_CLIP_SCALE" \
  --distill_kl_weight "$DISTILL_KL_WEIGHT" \
  --distill_temperature "$DISTILL_TEMPERATURE" \
  --noisy_ce_weight "$NOISY_CE_WEIGHT"
