#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-checkpoints/Qwen/Qwen3-1.7B}

# Target model on GPU0, noisy draft model on GPU1.
export DEVICE=${DEVICE:-cuda:0}
export DRAFT_DEVICE=${DRAFT_DEVICE:-cuda:1}

# Additive Gaussian weight noise applied to the draft model after load.
# Provide 3 stds: FFN QKV OUT.
export DRAFT_NOISE_STD=${DRAFT_NOISE_STD:-"1e-3 1e-3 1e-3"}
export DRAFT_NOISE_SEED=${DRAFT_NOISE_SEED:-1234}

export PROMPT=${PROMPT:-"Hi my name is"}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}
export NUM_SAMPLES=${NUM_SAMPLES:-50}
export TEMPERATURE=${TEMPERATURE:-0}
export SPECULATE_K=${SPECULATE_K:-5}

PYTHON=${PYTHON:-python3}
compile_args=()
if [[ "${COMPILE:-1}" == "1" ]]; then
  compile_args+=(--compile --compile_prefill)
fi

if [[ ! -f "$MODEL_DIR/model.pth" ]]; then
  echo "Missing $MODEL_DIR/model.pth"
  echo "Create it with: $PYTHON scripts/convert_hf_checkpoint.py --checkpoint_dir $MODEL_DIR"
  exit 1
fi

time "$PYTHON" generate.py \
  "${compile_args[@]}" \
  --checkpoint_path "$MODEL_DIR/model.pth" \
  --draft_checkpoint_path "$MODEL_DIR/model.pth" \
  --device "$DEVICE" --draft_device "$DRAFT_DEVICE" \
  --draft_noise_std $DRAFT_NOISE_STD --draft_noise_seed "$DRAFT_NOISE_SEED" \
  --speculate_k "$SPECULATE_K" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" --num_samples "$NUM_SAMPLES" --temperature "$TEMPERATURE"
