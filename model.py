# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)

ATTENTION_BACKEND = os.environ.get("GPT_FAST_ATTENTION_BACKEND", "flex").lower()
READ_NOISE_STD = float(os.environ.get("GPT_FAST_READ_NOISE_STD", "0") or "0")


def set_attention_backend(backend: str) -> None:
    global ATTENTION_BACKEND
    ATTENTION_BACKEND = (backend or "flex").lower()


def set_read_noise_std(std: float) -> None:
    global READ_NOISE_STD
    READ_NOISE_STD = max(0.0, float(std))


def get_read_noise_std() -> float:
    return float(READ_NOISE_STD)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


def linear_with_read_noise(linear: nn.Module, x: Tensor) -> Tensor:
    if READ_NOISE_STD <= 0:
        return linear(x)
    # Only apply read noise to standard floating-point stationary weights.
    if not isinstance(linear, nn.Linear):
        return linear(x)
    weight = linear.weight
    if not weight.is_floating_point():
        return linear(x)
    # Relative read noise: multiplicative perturbation (e.g., std=0.1 means ~10%).
    noisy_weight = weight * (1.0 + torch.randn_like(weight) * READ_NOISE_STD)
    return F.linear(x, noisy_weight, linear.bias)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = -1
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    use_qk_norm: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim == -1:
            self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        if not config:
            raise ValueError(f"Unknown model name: {name}")
        return cls(**transformer_configs[config[0]])

    @classmethod
    def from_hf_config(cls, hf_config: Mapping[str, Any]) -> "ModelArgs":
        dim = hf_config.get("hidden_size") or hf_config.get("dim")
        n_layer = hf_config.get("num_hidden_layers") or hf_config.get("n_layer") or hf_config.get("n_layers")
        n_head = hf_config.get("num_attention_heads") or hf_config.get("n_head") or hf_config.get("n_heads")
        vocab_size = hf_config.get("vocab_size") or hf_config.get("n_vocab") or hf_config.get("n_words")
        block_size = (
            hf_config.get("max_position_embeddings")
            or hf_config.get("max_seq_len")
            or hf_config.get("seq_length")
            or hf_config.get("block_size")
        )
        intermediate_size = hf_config.get("intermediate_size") or hf_config.get("ffn_dim") or hf_config.get("n_inner")
        n_local_heads = hf_config.get("num_key_value_heads") or hf_config.get("n_local_heads") or hf_config.get("num_kv_heads")
        head_dim = hf_config.get("head_dim")
        rope_base = hf_config.get("rope_theta") or hf_config.get("rope_base") or hf_config.get("rotary_emb_base")
        norm_eps = hf_config.get("rms_norm_eps") or hf_config.get("norm_eps") or hf_config.get("layer_norm_eps") or cls.norm_eps
        rope_scaling = hf_config.get("rope_scaling")
        model_type = str(hf_config.get("model_type") or "").lower()
        use_qk_norm = bool(hf_config.get("use_qk_norm") or hf_config.get("qk_norm"))
        if not use_qk_norm and model_type == "qwen3":
            use_qk_norm = True

        if dim is None or n_layer is None or n_head is None:
            raise ValueError(
                "Unsupported HF config: expected hidden_size/num_hidden_layers/num_attention_heads "
                f"(got hidden_size={dim}, num_hidden_layers={n_layer}, num_attention_heads={n_head})"
            )

        kwargs: dict[str, Any] = dict(
            dim=int(dim),
            n_layer=int(n_layer),
            n_head=int(n_head),
        )
        if vocab_size is not None:
            kwargs["vocab_size"] = int(vocab_size)
        if block_size is not None:
            kwargs["block_size"] = int(block_size)
        if intermediate_size is not None:
            kwargs["intermediate_size"] = int(intermediate_size)
        if n_local_heads is not None:
            kwargs["n_local_heads"] = int(n_local_heads)
        if head_dim is not None:
            kwargs["head_dim"] = int(head_dim)
        if rope_base is not None:
            kwargs["rope_base"] = float(rope_base)
        if norm_eps is not None:
            kwargs["norm_eps"] = float(norm_eps)
        if rope_scaling:
            kwargs["rope_scaling"] = dict(rope_scaling)
        kwargs["use_qk_norm"] = use_qk_norm
        return cls(**kwargs)

    @classmethod
    def from_hf_config_path(cls, config_path: Path) -> "ModelArgs":
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
        return cls.from_hf_config(hf_config)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: Path, *, model_name: Optional[str] = None) -> "ModelArgs":
        config_path = checkpoint_dir / "config.json"
        if config_path.is_file():
            return cls.from_hf_config_path(config_path)
        return cls.from_name(model_name or checkpoint_dir.name)


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.head_dim
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.head_dim, self.config.rope_base, dtype, self.config.rope_scaling)

    def forward(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = linear_with_read_noise(self.output, x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        q_dim = config.n_head * config.head_dim
        kv_dim = config.n_local_heads * config.head_dim
        total_head_dim = q_dim + 2 * kv_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(q_dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps) if config.use_qk_norm else nn.Identity()
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = linear_with_read_noise(self.wqkv, x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if ATTENTION_BACKEND == "flex":
            y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))
        elif ATTENTION_BACKEND in {"sdpa", "sdp"}:
            if input_pos is None:
                raise ValueError("input_pos is required for SDPA attention backend")

            # Restrict to the populated prefix to avoid attending to stale KVCache entries.
            kv_len = int(input_pos[-1].item()) + 1
            k = k[:, :, :kv_len]
            v = v[:, :, :kv_len]

            if self.n_head != self.n_local_heads:
                if self.n_head % self.n_local_heads != 0:
                    raise ValueError("n_head must be a multiple of n_local_heads for GQA")
                repeat = self.n_head // self.n_local_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

            kv_pos = torch.arange(kv_len, device=input_pos.device).view(1, -1)
            q_pos = input_pos.view(-1, 1)
            # NOTE: For torch.nn.functional.scaled_dot_product_attention, boolean masks use
            # True = "attend/keep", False = "mask out".
            attn_mask = kv_pos <= q_pos
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        else:
            raise ValueError(f"Unknown attention backend: {ATTENTION_BACKEND}")

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.q_dim)

        y = linear_with_read_noise(self.wo, y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return linear_with_read_noise(self.w2, F.silu(linear_with_read_noise(self.w1, x)) * linear_with_read_noise(self.w3, x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    if rope_scaling is None:
        return freqs

    rope_type = (rope_scaling.get("rope_type") or rope_scaling.get("type") or "").lower()
    factor = rope_scaling.get("factor")
    if factor is None:
        return freqs
    factor = float(factor)

    if rope_type in {"linear", "dynamic"} and "low_freq_factor" not in rope_scaling:
        return freqs / factor

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
