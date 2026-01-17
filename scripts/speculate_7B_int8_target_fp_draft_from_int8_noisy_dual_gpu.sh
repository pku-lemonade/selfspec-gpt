export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms

# Target model on GPU0 (INT8 weight-only), draft model on GPU1 (fp weights dequantized from the INT8 checkpoint).
export DEVICE=cuda:0
export DRAFT_DEVICE=cuda:1

export TARGET_CKPT=$MODEL_DIR/model_int8.pth
export DRAFT_CKPT=$MODEL_DIR/model_int8.pth

if [ ! -f "$TARGET_CKPT" ]; then
  echo "Missing $TARGET_CKPT"
  echo "Create it with: python quantize.py --checkpoint_path $MODEL_DIR/model.pth --mode int8"
  exit 1
fi

# Additive Gaussian weight noise applied to the *draft fp weights* after dequantization.
# Provide 3 stds: FFN QKV OUT.
export DRAFT_NOISE_STD="1e-3 0 0"
export DRAFT_NOISE_SEED=1234

time python generate.py \
  --compile --compile_prefill \
  --no_compile_block_mask \
  --checkpoint_path $TARGET_CKPT \
  --draft_checkpoint_path $DRAFT_CKPT \
  --draft_dequantize_int8 \
  --device $DEVICE --draft_device $DRAFT_DEVICE \
  --draft_noise_std $DRAFT_NOISE_STD --draft_noise_seed $DRAFT_NOISE_SEED \
  --speculate_k 5 \
  --prompt "Hi my name is" \
  --max_new_tokens 200 --num_samples 50 --temperature 0
