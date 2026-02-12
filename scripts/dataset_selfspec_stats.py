from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate as g
from draft_noise import resolve_level_based_draft_noise_stds
from model import set_attention_backend, set_read_noise_std
from selfspec_stats import accept_counts_to_stats, build_stats_meta, write_json
from tokenizer import get_tokenizer, resolve_tokenizer_path


def _load_prompts_txt(path: Path) -> List[str]:
    prompts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        prompts.append(s)
    return prompts


def _load_prompts_jsonl(path: Path, *, prompt_field: str) -> List[str]:
    prompts: List[str] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSONL at {path}:{idx}") from exc
        if prompt_field not in obj:
            raise ValueError(f"Missing field {prompt_field!r} at {path}:{idx}")
        val = obj[prompt_field]
        if not isinstance(val, str):
            raise ValueError(f"Field {prompt_field!r} must be a string at {path}:{idx}; got {type(val)}")
        s = val.strip()
        if not s:
            continue
        prompts.append(s)
    return prompts


def load_prompts(path: Path, *, prompt_field: str = "prompt") -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_prompts_txt(path)
    if suffix in {".jsonl", ".jsonlines"}:
        return _load_prompts_jsonl(path, prompt_field=prompt_field)
    raise ValueError(f"Unsupported prompts format: {path.suffix} (expected .txt/.jsonl)")


def _coerce_draft_noise_stds(draft_noise_std: Sequence[float]) -> Tuple[float, float, float]:
    if draft_noise_std is None or len(draft_noise_std) == 0:
        return 0.0, 0.0, 0.0
    vals = [float(x) for x in draft_noise_std]
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) == 3:
        return vals[0], vals[1], vals[2]
    raise ValueError("--draft_noise_std must have 1 or 3 values (FFN QKV OUT)")


def _resolve_out_dir(out_dir: Optional[Path], run_id: Optional[str]) -> Path:
    if out_dir is not None:
        return out_dir
    if run_id is not None:
        rid = run_id
    else:
        rid = datetime.now(timezone.utc).strftime("selfspec_stats_%Y%m%d_%H%M%SZ")
    return Path("out") / rid


def _seed_for(prompt_idx: int, sample_idx: int, *, base_seed: int, num_samples: int) -> int:
    return int(base_seed) + prompt_idx * int(num_samples) + sample_idx


@torch.no_grad()
def _run_prompt(
    *,
    model: g.Transformer,
    draft_model: g.Transformer,
    encoded_prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    speculate_k: int,
    seed: int,
) -> List[int]:
    torch.manual_seed(int(seed))
    _, stats = g.generate(
        model,
        encoded_prompt,
        max_new_tokens,
        batch_size=1,
        draft_model=draft_model,
        speculate_k=speculate_k,
        interactive=False,
        callback=lambda _: None,
        temperature=temperature,
        top_k=top_k,
    )
    accept_counts = stats.get("accept_counts") or [0] * (speculate_k + 1)
    return [int(x) for x in accept_counts]


def _truncate_or_skip_to_length(encoded: torch.Tensor, *, length: int) -> torch.Tensor | None:
    if int(encoded.size(-1)) < int(length):
        return None
    if int(encoded.size(-1)) == int(length):
        return encoded
    return encoded[: int(length)]


def _aggregate_for_length(
    *,
    prompts: Sequence[str],
    prompt_length: int | None,
    tokenizer,
    device: str,
    model: g.Transformer,
    draft_model: g.Transformer,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    speculate_k: int,
    num_samples: int,
    seed: int,
) -> Tuple[List[int], dict[str, Any]]:
    counts = [0] * (int(speculate_k) + 1)
    processed_prompts = 0
    skipped_short = 0
    truncated = 0

    for prompt_idx, prompt in enumerate(prompts):
        encoded = g.encode_tokens(tokenizer, prompt, bos=True, device=device)
        if prompt_length is not None:
            orig_len = int(encoded.size(-1))
            maybe = _truncate_or_skip_to_length(encoded, length=int(prompt_length))
            if maybe is None:
                skipped_short += 1
                continue
            if int(maybe.size(-1)) != orig_len:
                truncated += 1
            encoded = maybe

        for sample_idx in range(int(num_samples)):
            s = _seed_for(prompt_idx, sample_idx, base_seed=int(seed), num_samples=int(num_samples))
            accept_counts = _run_prompt(
                model=model,
                draft_model=draft_model,
                encoded_prompt=encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                speculate_k=speculate_k,
                seed=s,
            )
            if len(accept_counts) != len(counts):
                raise ValueError(
                    f"Inconsistent accept_counts length: got {len(accept_counts)} expected {len(counts)}"
                )
            for a, v in enumerate(accept_counts):
                counts[a] += int(v)

        processed_prompts += 1

    meta = {
        "processed_prompts": processed_prompts,
        "skipped_short_prompts": skipped_short,
        "truncated_prompts": truncated,
        "num_samples_per_prompt": int(num_samples),
    }
    return counts, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate speculative decoding acceptance stats over a prompt dataset.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Target model checkpoint (.pth).")
    parser.add_argument("--draft_checkpoint_path", type=Path, required=True, help="Draft model checkpoint (.pth).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device.")
    parser.add_argument("--draft_device", type=str, default=None, help="Draft device (defaults to --device).")
    parser.add_argument("--speculate_k", type=int, default=5, help="Speculative depth (k).")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling.")
    parser.add_argument("--num_samples", type=int, default=1, help="Runs per prompt (aggregated).")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed for sampling/acceptance.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts processed.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts deterministically using --seed.")

    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts (.txt or .jsonl).")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="JSONL field name for prompt text.")
    parser.add_argument(
        "--prompt_lengths",
        nargs="+",
        type=int,
        default=None,
        help="Optional prompt-length sweep. Writes one stats file per length.",
    )

    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: out/<run_id>/).")
    parser.add_argument("--run_id", type=str, default=None, help="Run id used when --out_dir is not provided.")

    # Keep key knobs aligned with generate.py so results are reproducible.
    parser.add_argument(
        "--draft_noise_std",
        type=float,
        nargs="+",
        default=[0.0],
        help="Gaussian noise std(s) to add to draft weights after load. Provide 1 value (all) or 3 values: FFN QKV OUT.",
    )
    parser.add_argument(
        "--draft_noise_level_stds",
        type=float,
        nargs="+",
        default=None,
        help="Draft noise level std table. Index i is noise level i. Requires --draft_noise_levels. Overrides --draft_noise_std.",
    )
    parser.add_argument(
        "--draft_noise_levels",
        type=int,
        nargs="+",
        default=None,
        help="Draft noise level assignments. Provide 1 value (all), 3 values (FFN QKV OUT), or 3*n_layer values (per-layer triplets in FFN QKV OUT order). Requires --draft_noise_level_stds.",
    )
    parser.add_argument("--draft_noise_seed", type=int, default=1234, help="RNG seed for draft weight noise.")
    parser.add_argument(
        "--draft_dequantize_int8",
        action="store_true",
        help="If set, treat the draft checkpoint as an int8 weight-only checkpoint and dequantize it to fp weights for draft inference.",
    )
    parser.add_argument(
        "--draft_fake_act_quant_int8",
        action="store_true",
        help="If set, apply per-token int8 fake activation quantization to the draft model linears (still runs fp matmuls).",
    )
    parser.add_argument(
        "--int8_act_quant",
        action="store_true",
        help="If set (and checkpoint is int8), quantize activations per-token and run int8xint8 matmuls for linear layers (target model).",
    )
    parser.add_argument(
        "--post_matmul_quant_bits",
        type=int,
        default=0,
        help="If non-zero, fake-quantize the output of each linear matmul per token to this many bits (supported: 8, 16).",
    )
    parser.add_argument(
        "--draft_post_matmul_quant_bits",
        type=int,
        default=0,
        help="Same as --post_matmul_quant_bits but applied to the draft model.",
    )
    parser.add_argument(
        "--read_noise_std",
        type=float,
        default=0.0,
        help="Per-matmul Gaussian read-noise std for stationary fp weights. 0 disables runtime read noise.",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        choices=["flex", "sdpa"],
        default="flex",
        help="Attention backend to use. Use sdpa as a stability fallback if flex_attention crashes.",
    )

    args = parser.parse_args()

    out_dir = _resolve_out_dir(args.out_dir, args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts, prompt_field=args.prompt_field)
    if args.shuffle:
        rng = torch.Generator()
        rng.manual_seed(int(args.seed))
        order = torch.randperm(len(prompts), generator=rng).tolist()
        prompts = [prompts[i] for i in order]
    if args.limit is not None:
        prompts = prompts[: int(args.limit)]

    if len(prompts) == 0:
        raise ValueError("No prompts to process (after filtering/limit).")

    assert args.checkpoint_path.is_file(), str(args.checkpoint_path)
    assert args.draft_checkpoint_path.is_file(), str(args.draft_checkpoint_path)

    tokenizer_path = resolve_tokenizer_path(args.checkpoint_path.parent)
    assert tokenizer_path.is_file(), str(tokenizer_path)

    draft_device = args.draft_device or args.device
    set_attention_backend(args.attention_backend)
    set_read_noise_std(float(args.read_noise_std))

    precision = torch.bfloat16
    model = g._load_model(args.checkpoint_path, args.device, precision, use_tp=False, int8_act_quant=bool(args.int8_act_quant))
    if args.post_matmul_quant_bits:
        from quantize import set_post_matmul_output_quant_bits

        set_post_matmul_output_quant_bits(model, int(args.post_matmul_quant_bits))

    if args.draft_dequantize_int8:
        draft_model = g._load_int8_weight_only_as_fp_model(args.draft_checkpoint_path, draft_device, precision, use_tp=False)
    else:
        draft_model = g._load_model(args.draft_checkpoint_path, draft_device, precision, use_tp=False)

    if args.draft_fake_act_quant_int8:
        from quantize import replace_linear_fake_act_quant

        replace_linear_fake_act_quant(draft_model)
    if args.draft_post_matmul_quant_bits:
        from quantize import set_post_matmul_output_quant_bits

        set_post_matmul_output_quant_bits(draft_model, int(args.draft_post_matmul_quant_bits))

    # Optional draft noise after load.
    n_layer = len(draft_model.layers)
    use_levels = (args.draft_noise_levels is not None) or (args.draft_noise_level_stds is not None)
    per_layer_stds: Sequence[Tuple[float, float, float]]
    output_std: float
    if use_levels:
        if args.draft_noise_levels is None or args.draft_noise_level_stds is None:
            raise ValueError("Level-based draft noise requires both --draft_noise_level_stds and --draft_noise_levels.")
        per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
            draft_noise_level_stds=args.draft_noise_level_stds,
            draft_noise_levels=args.draft_noise_levels,
            n_layer=n_layer,
        )
    else:
        ffn_std, qkv_std, out_std = _coerce_draft_noise_stds(args.draft_noise_std)
        per_layer_stds = [(ffn_std, qkv_std, out_std) for _ in range(n_layer)]
        output_std = float(out_std)

    if output_std > 0 or any((ffn > 0 or qkv > 0 or out > 0) for ffn, qkv, out in per_layer_stds):
        g.add_gaussian_noise_to_draft_weights_(
            draft_model,
            per_layer_stds=per_layer_stds,
            output_std=output_std,
            seed=int(args.draft_noise_seed),
        )

    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)

    prompt_lengths = args.prompt_lengths
    if prompt_lengths is None:
        counts, agg_meta = _aggregate_for_length(
            prompts=prompts,
            prompt_length=None,
            tokenizer=tokenizer,
            device=args.device,
            model=model,
            draft_model=draft_model,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            speculate_k=int(args.speculate_k),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
        )

        stats_payload = accept_counts_to_stats(counts, k=int(args.speculate_k))
        stats_path = out_dir / "stats.json"
        write_json(stats_path, stats_payload)

        meta_path = out_dir / "stats_meta.json"
        meta_payload = build_stats_meta(
            stats=stats_payload,
            run_id=out_dir.name,
            repo_root=REPO_ROOT,
            paths={
                "out_dir": str(out_dir),
                "stats": str(stats_path),
                "stats_meta": str(meta_path),
                "prompts": str(args.prompts),
            },
            model={
                "checkpoint_path": str(args.checkpoint_path),
                "draft_checkpoint_path": str(args.draft_checkpoint_path),
                "tokenizer_path": str(tokenizer_path),
            },
            generation={
                "max_new_tokens": int(args.max_new_tokens),
                "num_samples_per_prompt": int(args.num_samples),
                "top_k": int(args.top_k),
                "temperature": float(args.temperature),
                "speculate_k": int(args.speculate_k),
                "bos_included": True,
            },
            knobs={
                "attention_backend": str(args.attention_backend),
                "read_noise_std": float(args.read_noise_std),
                "draft_noise_std": [float(x) for x in args.draft_noise_std] if args.draft_noise_std is not None else None,
                "draft_noise_level_stds": None if args.draft_noise_level_stds is None else [float(x) for x in args.draft_noise_level_stds],
                "draft_noise_levels": None if args.draft_noise_levels is None else [int(x) for x in args.draft_noise_levels],
                "draft_noise_seed": int(args.draft_noise_seed),
                "draft_dequantize_int8": bool(args.draft_dequantize_int8),
                "draft_fake_act_quant_int8": bool(args.draft_fake_act_quant_int8),
                "int8_act_quant": bool(args.int8_act_quant),
                "post_matmul_quant_bits": int(args.post_matmul_quant_bits),
                "draft_post_matmul_quant_bits": int(args.draft_post_matmul_quant_bits),
                "device": str(args.device),
                "draft_device": str(draft_device),
            },
            seeds={
                "base_seed": int(args.seed),
                "scheme": "seed = base_seed + prompt_idx * num_samples + sample_idx",
            },
            dataset={
                "prompts_path": str(args.prompts),
                "format": args.prompts.suffix.lower(),
                "prompt_field": str(args.prompt_field),
                "limit": None if args.limit is None else int(args.limit),
                "shuffle": bool(args.shuffle),
            },
            aggregation={
                **agg_meta,
                "total_prompts_loaded": int(len(prompts)),
            },
        )
        write_json(meta_path, meta_payload)
        return

    # Prompt-length sweep mode.
    for L in [int(x) for x in prompt_lengths]:
        counts, agg_meta = _aggregate_for_length(
            prompts=prompts,
            prompt_length=L,
            tokenizer=tokenizer,
            device=args.device,
            model=model,
            draft_model=draft_model,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            speculate_k=int(args.speculate_k),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
        )
        if int(agg_meta["processed_prompts"]) == 0:
            raise ValueError(f"No prompts processed for prompt_length={L} (all prompts shorter than L).")

        stats_payload = accept_counts_to_stats(counts, k=int(args.speculate_k))
        stats_path = out_dir / f"stats_Lprompt_{L}.json"
        write_json(stats_path, stats_payload)

        meta_path = out_dir / f"stats_meta_Lprompt_{L}.json"
        meta_payload = build_stats_meta(
            stats=stats_payload,
            run_id=out_dir.name,
            repo_root=REPO_ROOT,
            paths={
                "out_dir": str(out_dir),
                "stats": str(stats_path),
                "stats_meta": str(meta_path),
                "prompts": str(args.prompts),
            },
            model={
                "checkpoint_path": str(args.checkpoint_path),
                "draft_checkpoint_path": str(args.draft_checkpoint_path),
                "tokenizer_path": str(tokenizer_path),
            },
            generation={
                "prompt_length": int(L),
                "max_new_tokens": int(args.max_new_tokens),
                "num_samples_per_prompt": int(args.num_samples),
                "top_k": int(args.top_k),
                "temperature": float(args.temperature),
                "speculate_k": int(args.speculate_k),
                "bos_included": True,
                "prompt_length_policy": "truncate_or_skip_short",
            },
            knobs={
                "attention_backend": str(args.attention_backend),
                "read_noise_std": float(args.read_noise_std),
                "draft_noise_std": [float(x) for x in args.draft_noise_std] if args.draft_noise_std is not None else None,
                "draft_noise_level_stds": None if args.draft_noise_level_stds is None else [float(x) for x in args.draft_noise_level_stds],
                "draft_noise_levels": None if args.draft_noise_levels is None else [int(x) for x in args.draft_noise_levels],
                "draft_noise_seed": int(args.draft_noise_seed),
                "draft_dequantize_int8": bool(args.draft_dequantize_int8),
                "draft_fake_act_quant_int8": bool(args.draft_fake_act_quant_int8),
                "int8_act_quant": bool(args.int8_act_quant),
                "post_matmul_quant_bits": int(args.post_matmul_quant_bits),
                "draft_post_matmul_quant_bits": int(args.draft_post_matmul_quant_bits),
                "device": str(args.device),
                "draft_device": str(draft_device),
            },
            seeds={
                "base_seed": int(args.seed),
                "scheme": "seed = base_seed + prompt_idx * num_samples + sample_idx",
            },
            dataset={
                "prompts_path": str(args.prompts),
                "format": args.prompts.suffix.lower(),
                "prompt_field": str(args.prompt_field),
                "limit": None if args.limit is None else int(args.limit),
                "shuffle": bool(args.shuffle),
            },
            aggregation={
                **agg_meta,
                "total_prompts_loaded": int(len(prompts)),
            },
        )
        write_json(meta_path, meta_payload)


if __name__ == "__main__":
    main()
