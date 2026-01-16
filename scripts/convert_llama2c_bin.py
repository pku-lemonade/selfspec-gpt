#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import mmap
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch


def _read_int32s(mm: mmap.mmap, *, offset_bytes: int, count: int) -> Tuple[int, ...]:
    fmt = "<" + ("i" * count)
    return struct.unpack_from(fmt, mm, offset_bytes)


def _read_f32(
    mm: mmap.mmap, *, offset_bytes: int, numel: int
) -> Tuple[torch.Tensor, int]:
    tensor = torch.frombuffer(mm, dtype=torch.float32, count=numel, offset=offset_bytes).clone()
    return tensor, offset_bytes + (numel * 4)


def convert_llama2c_bin(*, bin_path: Path, output_path: Path) -> None:
    with bin_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size_signed, seq_len = _read_int32s(
                mm, offset_bytes=0, count=7
            )
            if dim <= 0 or hidden_dim <= 0 or n_layers <= 0 or n_heads <= 0 or n_kv_heads <= 0:
                raise ValueError(f"Invalid header values: {(dim, hidden_dim, n_layers, n_heads, n_kv_heads)}")
            if dim % n_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")

            vocab_size = abs(vocab_size_signed)
            head_dim = dim // n_heads
            kv_dim = head_dim * n_kv_heads

            print(
                "Parsed llama2.c checkpoint header:",
                {
                    "dim": dim,
                    "hidden_dim": hidden_dim,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "vocab_size": vocab_size,
                    "seq_len": seq_len,
                },
            )

            offset = 7 * 4
            state_dict: Dict[str, torch.Tensor] = {}

            tok_embeddings, offset = _read_f32(mm, offset_bytes=offset, numel=vocab_size * dim)
            tok_embeddings = tok_embeddings.view(vocab_size, dim)
            state_dict["tok_embeddings.weight"] = tok_embeddings

            for layer in range(n_layers):
                attn_norm, offset = _read_f32(mm, offset_bytes=offset, numel=dim)
                state_dict[f"layers.{layer}.attention_norm.weight"] = attn_norm

                wq, offset = _read_f32(mm, offset_bytes=offset, numel=dim * dim)
                wk, offset = _read_f32(mm, offset_bytes=offset, numel=kv_dim * dim)
                wv, offset = _read_f32(mm, offset_bytes=offset, numel=kv_dim * dim)
                wo, offset = _read_f32(mm, offset_bytes=offset, numel=dim * dim)

                wq = wq.view(dim, dim)
                wk = wk.view(kv_dim, dim)
                wv = wv.view(kv_dim, dim)
                state_dict[f"layers.{layer}.attention.wqkv.weight"] = torch.cat([wq, wk, wv], dim=0)
                state_dict[f"layers.{layer}.attention.wo.weight"] = wo.view(dim, dim)

                ffn_norm, offset = _read_f32(mm, offset_bytes=offset, numel=dim)
                state_dict[f"layers.{layer}.ffn_norm.weight"] = ffn_norm

                w1, offset = _read_f32(mm, offset_bytes=offset, numel=hidden_dim * dim)
                w2, offset = _read_f32(mm, offset_bytes=offset, numel=dim * hidden_dim)
                w3, offset = _read_f32(mm, offset_bytes=offset, numel=hidden_dim * dim)
                state_dict[f"layers.{layer}.feed_forward.w1.weight"] = w1.view(hidden_dim, dim)
                state_dict[f"layers.{layer}.feed_forward.w2.weight"] = w2.view(dim, hidden_dim)
                state_dict[f"layers.{layer}.feed_forward.w3.weight"] = w3.view(hidden_dim, dim)

            final_norm, offset = _read_f32(mm, offset_bytes=offset, numel=dim)
            state_dict["norm.weight"] = final_norm

            remaining_bytes = mm.size() - offset
            expected_wcls_bytes = vocab_size * dim * 4

            if remaining_bytes >= expected_wcls_bytes:
                wcls, offset = _read_f32(mm, offset_bytes=offset, numel=vocab_size * dim)
                state_dict["output.weight"] = wcls.view(vocab_size, dim)
            else:
                state_dict["output.weight"] = tok_embeddings

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, output_path)
            print(f"Wrote {output_path} (ignored {mm.size() - offset} trailing bytes)")
        finally:
            mm.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Karpathy llama2.c-style .bin checkpoint into a gpt-fast model.pth state_dict."
    )
    parser.add_argument(
        "--bin_path",
        type=Path,
        required=True,
        help="Path to the llama2.c .bin checkpoint (e.g. stories15M.bin).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Where to write the converted checkpoint (typically .../model.pth).",
    )
    args = parser.parse_args()
    convert_llama2c_bin(bin_path=args.bin_path, output_path=args.output_path)


if __name__ == "__main__":
    main()

