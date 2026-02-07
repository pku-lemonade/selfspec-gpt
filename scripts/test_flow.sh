export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
rm -rf "checkpoints/$MODEL_REPO"
PYTHON=${PYTHON:-python3}
$PYTHON scripts/download.py --repo_id "$MODEL_REPO"
$PYTHON scripts/convert_hf_checkpoint.py --checkpoint_dir "checkpoints/$MODEL_REPO"
$PYTHON quantize.py --checkpoint_path "checkpoints/$MODEL_REPO/model.pth"
compile_args=()
if [[ "${COMPILE:-}" == "1" ]]; then
  compile_args=(--compile)
fi
$PYTHON generate.py "${compile_args[@]}" --checkpoint_path "checkpoints/$MODEL_REPO/model_int8.pth" --max_new_tokens 100
