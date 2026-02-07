#!/usr/bin/env bash
set -euo pipefail

# Sweep draft noise for speculative decoding with:
# - target: Qwen3-1.7B
# - draft:  Qwen3-0.6B

export TARGET_MODEL_DIR=${TARGET_MODEL_DIR:-checkpoints/Qwen/Qwen3-1.7B}
export DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-checkpoints/Qwen/Qwen3-0.6B}

export DEVICE=${DEVICE:-cuda:0}
export DRAFT_DEVICE=${DRAFT_DEVICE:-cuda:1}

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

time "$PYTHON" scripts/bench_noisy_spec_decode.py \
  --checkpoint_path "$TARGET_CKPT" \
  --draft_checkpoint_path "$DRAFT_CKPT" \
  --device "$DEVICE" --draft_device "$DRAFT_DEVICE" \
  --speculate_k 5 \
  --prompt "Hi my name is" \
  --max_new_tokens 200 --num_samples 10 --temperature 0 \
  --noise_sweep "0,1e-4,3e-4,1e-3,3e-3,5e-3,7e-3,1e-2,2e-2,3e-2" \
  --noise_bucket all \
  --compile --compile_prefill
