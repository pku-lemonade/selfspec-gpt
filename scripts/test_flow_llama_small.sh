#!/usr/bin/env bash
set -euo pipefail

# Note: Meta Llama 4 models on HF are currently 17B MoE (not ~1B).
# For a ~1B Llama-family model, this script defaults to Llama-3.2-1B-Instruct.

PYTHON=${PYTHON:-python3}
MODEL_REPO=${1:-meta-llama/Llama-3.2-1B-Instruct}

token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  token_args=(--hf_token "$HF_TOKEN")
elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  token_args=(--hf_token "$HUGGINGFACE_HUB_TOKEN")
fi

rm -rf "checkpoints/$MODEL_REPO"

$PYTHON scripts/download.py --repo_id "$MODEL_REPO" "${token_args[@]}"
$PYTHON scripts/convert_hf_checkpoint.py --checkpoint_dir "checkpoints/$MODEL_REPO"
$PYTHON quantize.py --checkpoint_path "checkpoints/$MODEL_REPO/model.pth" --mode int8
compile_args=()
if [[ "${COMPILE:-}" == "1" ]]; then
  compile_args=(--compile)
fi
$PYTHON generate.py "${compile_args[@]}" --checkpoint_path "checkpoints/$MODEL_REPO/model_int8.pth" --max_new_tokens 100 --prompt "Hello, my name is"
