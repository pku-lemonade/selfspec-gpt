#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-checkpoints/meta-llama/Llama-3.2-1B}

# Target and noisy draft default to the same GPU. Override DRAFT_DEVICE to split them.
export DEVICE=${DEVICE:-cuda:0}
export DRAFT_DEVICE=${DRAFT_DEVICE:-$DEVICE}

# Additive Gaussian weight noise applied to the draft model after load.
# Level-based config: map levels 1/2/3 to the per-bucket stds (level 0 = 0 disables noise).
export DRAFT_NOISE_STD=${DRAFT_NOISE_STD:-"1e-3 1e-3 1e-3"}
export DRAFT_NOISE_LEVEL_STDS=${DRAFT_NOISE_LEVEL_STDS:-"0 $DRAFT_NOISE_STD"}
export DRAFT_NOISE_LEVELS=${DRAFT_NOISE_LEVELS:-"1 2 3"}
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
  --draft_noise_level_stds $DRAFT_NOISE_LEVEL_STDS --draft_noise_levels $DRAFT_NOISE_LEVELS --draft_noise_seed "$DRAFT_NOISE_SEED" \
  --speculate_k "$SPECULATE_K" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" --num_samples "$NUM_SAMPLES" --temperature "$TEMPERATURE"
