#!/usr/bin/env bash
set -euo pipefail

# Qwen3 target+draf setup matching speculate_7B_fp_noisy_dual_gpu.sh semantics.
# Target model on GPU0, noisy draft model on GPU1.

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

# Additive Gaussian weight noise applied to the draft model after load.
# Provide 3 stds: FFN QKV OUT.
export DRAFT_NOISE_STD=${DRAFT_NOISE_STD:-"1e-3 1e-3 1e-3"}
export DRAFT_NOISE_SEED=${DRAFT_NOISE_SEED:-1234}

PYTHON=${PYTHON:-python3}

time "$PYTHON" generate.py \
  --compile --compile_prefill \
  --checkpoint_path "$TARGET_CKPT" \
  --draft_checkpoint_path "$DRAFT_CKPT" \
  --device "$DEVICE" --draft_device "$DRAFT_DEVICE" \
  --draft_noise_std $DRAFT_NOISE_STD --draft_noise_seed "$DRAFT_NOISE_SEED" \
  --speculate_k 5 \
  --prompt "Hi my name is" \
  --max_new_tokens 200 --num_samples 50 --temperature 0
