export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms

# Target model on GPU0, noisy draft model on GPU1.
export DEVICE=cuda:0
export DRAFT_DEVICE=cuda:1

# Additive Gaussian weight noise applied to the draft model after load.
# Level-based config: define a level->std table and assign (FFN QKV OUT) levels.
# Here we map levels 1/2/3 to the per-bucket stds (level 0 = 0 disables noise).
export DRAFT_NOISE_STD="1e-3 1e-3 1e-3"
export DRAFT_NOISE_LEVEL_STDS="0 $DRAFT_NOISE_STD"
export DRAFT_NOISE_LEVELS="1 2 3"
export DRAFT_NOISE_SEED=1234

time python generate.py \
  --compile --compile_prefill \
  --checkpoint_path $MODEL_DIR/model.pth \
  --draft_checkpoint_path $MODEL_DIR/model.pth \
  --device $DEVICE --draft_device $DRAFT_DEVICE \
  --draft_noise_level_stds $DRAFT_NOISE_LEVEL_STDS --draft_noise_levels $DRAFT_NOISE_LEVELS --draft_noise_seed $DRAFT_NOISE_SEED \
  --speculate_k 5 \
  --prompt "Hi my name is" \
  --max_new_tokens 200 --num_samples 50 --temperature 0
