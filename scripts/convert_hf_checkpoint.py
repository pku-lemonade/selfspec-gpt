# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        config = ModelArgs.from_hf_config_path(config_path)
    else:
        config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / 'model.safetensors.index.json'
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None
   
    try:
      assert model_map_json_safetensors.is_file()
      model_map_json = model_map_json_safetensors
      print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
      print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
      try:
        assert model_map_json_pytorch.is_file()
        model_map_json = model_map_json_pytorch
        print(f"Found pytorch index at {model_map_json_pytorch}")
      except AssertionError:
        print(f"{model_map_json_pytorch} not found")
   
    bin_files = None
    if model_map_json is not None:
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        # Handle unsharded checkpoints (e.g. Llama-3.2-1B stores a single model.safetensors).
        safetensors_file = checkpoint_dir / "model.safetensors"
        pytorch_file = checkpoint_dir / "pytorch_model.bin"
        if safetensors_file.is_file():
            bin_files = {safetensors_file}
        elif pytorch_file.is_file():
            bin_files = {pytorch_file}
        else:
            raise Exception("No model weights found (expected an index json or model.safetensors / pytorch_model.bin)")

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
        "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
        'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    for file in sorted(bin_files):
       if "safetensors" in str(file):
           state_dict = load_safetensors_file(str(file), device="cpu")
           merged_result.update(state_dict)
       else:
           state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
           merged_result.update(state_dict)
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map.get(abstract_key)
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map.get(key)
            if new_key is None:
                continue

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            # LLaMA-2 style checkpoints need RoPE permutation, while newer families
            # (e.g. Qwen / Llama-3) already store packed projection matrices.
            if q.shape[0] == config.dim:
                q = permute(q, config.n_head)
            if k.shape[0] == config.dim and config.n_local_heads == config.n_head:
                k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]
    if "output.weight" not in final_result and "tok_embeddings.weight" in final_result:
        final_result["output.weight"] = final_result["tok_embeddings.weight"]

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    tokenizer_dest = checkpoint_dir / "tokenizer.model"
    if not tokenizer_dest.is_file():
        for src in (
            checkpoint_dir / "original" / "mp16" / "tokenizer.model",
            checkpoint_dir / "original" / "tokenizer.model",
        ):
            if src.is_file():
                print(f"Copying {src} to {tokenizer_dest}")
                shutil.copy(src, tokenizer_dest)
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
