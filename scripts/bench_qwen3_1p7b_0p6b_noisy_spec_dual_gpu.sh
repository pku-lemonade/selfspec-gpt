#!/usr/bin/env bash
set -euo pipefail

# Sweep draft noise for speculative decoding with:
# - target: Qwen3-1.7B
# - draft:  Qwen3-0.6B

export TARGET_MODEL_DIR=${TARGET_MODEL_DIR:-checkpoints/Qwen/Qwen3-1.7B}
export DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-checkpoints/Qwen/Qwen3-0.6B}

if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [[ "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')" -ge 2 ]]; then
    export DEVICE=cuda:1
  else
    export DEVICE=cuda:0
  fi
fi
export DRAFT_DEVICE=${DRAFT_DEVICE:-$DEVICE}

export TARGET_CKPT=${TARGET_CKPT:-$TARGET_MODEL_DIR/model.pth}
export DRAFT_CKPT=${DRAFT_CKPT:-$DRAFT_MODEL_DIR/model.pth}

if [ ! -f "$TARGET_CKPT" ]; then
  echo "Missing $TARGET_CKPT"
  echo "Create it with: python3 scripts/convert_hf_checkpoint.py --checkpoint_dir $TARGET_MODEL_DIR"
  exit 1
fi

if [ ! -f "$DRAFT_CKPT" ]; then
  echo "Missing $DRAFT_CKPT"
  echo "Create it with: python3 scripts/convert_hf_checkpoint.py --checkpoint_dir $DRAFT_MODEL_DIR"
  exit 1
fi

PYTHON=${PYTHON:-python3}
export SPECULATE_K=${SPECULATE_K:-5}
export PROMPT=${PROMPT:-"Hi my name is"}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}
export NUM_SAMPLES=${NUM_SAMPLES:-10}
export TEMPERATURE=${TEMPERATURE:-0}
export NOISE_SWEEP=${NOISE_SWEEP:-"0,1e-4,3e-4,1e-3,3e-3,5e-3,7e-3,1e-2,2e-2,3e-2"}
compile_args=()
if [[ "${COMPILE:-1}" == "1" ]]; then
  compile_args+=(--compile --compile_prefill)
fi

time "$PYTHON" scripts/bench_noisy_spec_decode.py \
  --checkpoint_path "$TARGET_CKPT" \
  --draft_checkpoint_path "$DRAFT_CKPT" \
  --device "$DEVICE" --draft_device "$DRAFT_DEVICE" \
  --speculate_k "$SPECULATE_K" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" --num_samples "$NUM_SAMPLES" --temperature "$TEMPERATURE" \
  --noise_sweep "$NOISE_SWEEP" \
  --noise_bucket all \
  "${compile_args[@]}"
