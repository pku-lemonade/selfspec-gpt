PYTHON=${PYTHON:-python3}
$PYTHON scripts/download.py --repo_id "$1" && $PYTHON scripts/convert_hf_checkpoint.py --checkpoint_dir "checkpoints/$1" && $PYTHON quantize.py --checkpoint_path "checkpoints/$1/model.pth" --mode int8
