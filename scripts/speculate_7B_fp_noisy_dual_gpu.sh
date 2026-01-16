export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms

# Target model on GPU0, noisy draft model on GPU1.
export DEVICE=cuda:0
export DRAFT_DEVICE=cuda:1

# Additive Gaussian weight noise applied to the draft model after load.
# Suggested sweep: 0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
export DRAFT_NOISE_STD=1e-3
export DRAFT_NOISE_SEED=1234

time python generate.py \
  --compile --compile_prefill \
  --checkpoint_path $MODEL_DIR/model.pth \
  --draft_checkpoint_path $MODEL_DIR/model.pth \
  --device $DEVICE --draft_device $DRAFT_DEVICE \
  --draft_noise_std $DRAFT_NOISE_STD --draft_noise_seed $DRAFT_NOISE_SEED \
  --speculate_k 5 \
  --prompt "Hi my name is" \
  --max_new_tokens 200 --num_samples 50 --temperature 0
