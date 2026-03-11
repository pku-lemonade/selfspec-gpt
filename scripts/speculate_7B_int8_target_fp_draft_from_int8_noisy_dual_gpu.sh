export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms

# Target model on GPU0 (INT8 weight-only), draft model on GPU1 (fp weights dequantized from the INT8 checkpoint).
export DEVICE=cuda:0
export DRAFT_DEVICE=${DRAFT_DEVICE:-$DEVICE}

export TARGET_CKPT=$MODEL_DIR/model_int8.pth
export DRAFT_CKPT=$MODEL_DIR/model_int8.pth

if [ ! -f "$TARGET_CKPT" ]; then
  echo "Missing $TARGET_CKPT"
  echo "Create it with: python quantize.py --checkpoint_path $MODEL_DIR/model.pth --mode int8"
  exit 1
fi

# Additive Gaussian weight noise applied to the *draft fp weights* after dequantization.
# Level-based config: map levels 1/2/3 to the per-bucket stds (level 0 = 0 disables noise).
export DRAFT_NOISE_STD="1e-3 0 0"
export DRAFT_NOISE_LEVEL_STDS="0 $DRAFT_NOISE_STD"
export DRAFT_NOISE_LEVELS="1 2 3"
export DRAFT_NOISE_SEED=1234
export PROMPT=${PROMPT:-"Hi my name is"}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}
export NUM_SAMPLES=${NUM_SAMPLES:-50}
export TEMPERATURE=${TEMPERATURE:-0}
export SPECULATE_K=${SPECULATE_K:-5}

compile_args=()
if [[ "${COMPILE:-1}" == "1" ]]; then
  compile_args+=(--compile --compile_prefill)
fi
block_mask_args=()
if [[ "${NO_COMPILE_BLOCK_MASK:-1}" == "1" ]]; then
  block_mask_args+=(--no_compile_block_mask)
fi

time python generate.py \
  "${compile_args[@]}" \
  "${block_mask_args[@]}" \
  --checkpoint_path $TARGET_CKPT \
  --draft_checkpoint_path $DRAFT_CKPT \
  --draft_dequantize_int8 \
  --device $DEVICE --draft_device $DRAFT_DEVICE \
  --draft_noise_level_stds $DRAFT_NOISE_LEVEL_STDS --draft_noise_levels $DRAFT_NOISE_LEVELS --draft_noise_seed $DRAFT_NOISE_SEED \
  --speculate_k "$SPECULATE_K" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" --num_samples "$NUM_SAMPLES" --temperature "$TEMPERATURE"
