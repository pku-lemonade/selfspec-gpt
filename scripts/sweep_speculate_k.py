#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate as g
from draft_noise import resolve_level_based_draft_noise_stds
from model import set_attention_backend, set_read_noise_std
from scripts.dataset_selfspec_stats import _aggregate_for_length, load_prompts
from tokenizer import get_tokenizer, resolve_tokenizer_path

DEFAULT_CUDA_DEVICE = (
    "cuda:1"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)


def _coerce_draft_noise_stds(draft_noise_std: Sequence[float]) -> Tuple[float, float, float]:
    if draft_noise_std is None or len(draft_noise_std) == 0:
        return 0.0, 0.0, 0.0
    vals = [float(x) for x in draft_noise_std]
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    if len(vals) == 3:
        return vals[0], vals[1], vals[2]
    raise ValueError("--draft_noise_std must have 1 or 3 values (FFN QKV OUT)")


def _default_out_path(run_id: Optional[str]) -> Path:
    if run_id:
        return Path("out") / f"{run_id}.json"
    rid = datetime.now(timezone.utc).strftime("sweep_speculate_k_%Y%m%d_%H%M%SZ")
    return Path("out") / f"{rid}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep speculative depth k over a prompt dataset and report acceptance stats.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Target model checkpoint (.pth).")
    parser.add_argument("--draft_checkpoint_path", type=Path, default=None, help="Draft model checkpoint (.pth). Defaults to --checkpoint_path.")
    parser.add_argument("--device", type=str, default=DEFAULT_CUDA_DEVICE, help="Target device.")
    parser.add_argument("--draft_device", type=str, default=None, help="Draft device (defaults to --device).")

    parser.add_argument("--prompts", type=Path, required=True, help="Prompt dataset (.txt or .jsonl).")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="JSONL field name for prompt text.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of prompts loaded.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle prompts deterministically using --seed.")
    parser.add_argument("--prompt_length", type=int, default=64, help="Prompt length to truncate/skip to.")

    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling.")
    parser.add_argument("--num_samples", type=int, default=1, help="Runs per prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8], help="Speculative depths to test.")

    parser.add_argument(
        "--draft_noise_std",
        type=float,
        nargs="+",
        default=[0.0],
        help="Relative Gaussian noise std(s) to apply to draft weights after load (multiplicative). Provide 1 or 3 values.",
    )
    parser.add_argument(
        "--draft_noise_level_stds",
        type=float,
        nargs="+",
        default=None,
        help="Draft noise level std table. Requires --draft_noise_levels.",
    )
    parser.add_argument(
        "--draft_noise_levels",
        type=int,
        nargs="+",
        default=None,
        help="Draft noise level assignments. Requires --draft_noise_level_stds.",
    )
    parser.add_argument("--draft_noise_seed", type=int, default=1234, help="RNG seed for draft weight noise.")
    parser.add_argument("--read_noise_std", type=float, default=0.0, help="Relative read-noise std.")
    parser.add_argument("--attention_backend", type=str, choices=["flex", "sdpa"], default="flex")
    parser.add_argument("--draft_dequantize_int8", action="store_true")
    parser.add_argument("--draft_fake_act_quant_int8", action="store_true")
    parser.add_argument("--int8_act_quant", action="store_true")
    parser.add_argument("--verify_adc_bits", type=int, default=0, help="ADC-style interface quantization bits for target/verify analog linear outputs. 0 disables.")
    parser.add_argument(
        "--verify_delta_readout",
        action="store_true",
        help="If set, quantize verify ADC outputs in predictive-delta mode against the previous reconstructed token output.",
    )
    parser.add_argument(
        "--verify_delta_dac_bits",
        type=int,
        default=0,
        help="Optional DAC bitwidth for the stored verify delta-readout feedback baseline. 0 keeps DAC feedback ideal/unmodeled.",
    )
    parser.add_argument("--draft_adc_bits", type=int, default=0, help="ADC-style interface quantization bits for draft analog linear outputs. 0 disables.")
    parser.add_argument(
        "--draft_delta_readout",
        action="store_true",
        help="If set, quantize draft ADC outputs in predictive-delta mode against the previous reconstructed token output.",
    )
    parser.add_argument(
        "--draft_delta_dac_bits",
        type=int,
        default=0,
        help="Optional DAC bitwidth for the stored draft delta-readout feedback baseline. 0 keeps DAC feedback ideal/unmodeled.",
    )
    parser.add_argument("--post_matmul_quant_bits", type=int, default=0, help="Legacy alias for verify ADC/interface quantization bits.")
    parser.add_argument("--draft_post_matmul_quant_bits", type=int, default=0, help="Legacy alias for draft ADC/interface quantization bits.")

    parser.add_argument("--run_id", type=str, default=None, help="Optional run id used in the output path.")
    parser.add_argument("--out_json", type=Path, default=None, help="Optional explicit output JSON path.")
    args = parser.parse_args()

    if args.draft_device is None:
        args.draft_device = args.device
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(str(args.checkpoint_path))
    draft_checkpoint_path = args.draft_checkpoint_path or args.checkpoint_path
    if not draft_checkpoint_path.is_file():
        raise FileNotFoundError(str(draft_checkpoint_path))

    prompts = load_prompts(args.prompts, prompt_field=args.prompt_field)
    if args.shuffle:
        rng = torch.Generator()
        rng.manual_seed(int(args.seed))
        order = torch.randperm(len(prompts), generator=rng).tolist()
        prompts = [prompts[i] for i in order]
    if args.limit is not None:
        prompts = prompts[: int(args.limit)]
    if len(prompts) == 0:
        raise ValueError("No prompts to process.")

    tokenizer_path = resolve_tokenizer_path(args.checkpoint_path.parent)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(str(tokenizer_path))

    set_attention_backend(args.attention_backend)
    set_read_noise_std(float(args.read_noise_std))

    precision = torch.bfloat16
    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)
    verify_quant_bits = g._resolve_interface_quant_bits(
        explicit_bits=int(args.verify_adc_bits),
        legacy_bits=int(args.post_matmul_quant_bits),
        label="verify ADC/interface",
    )
    verify_delta_dac_quant_bits = g._resolve_optional_quant_bits(
        bits=int(args.verify_delta_dac_bits),
        label="verify delta-readout DAC",
    )
    draft_quant_bits = g._resolve_interface_quant_bits(
        explicit_bits=int(args.draft_adc_bits),
        legacy_bits=int(args.draft_post_matmul_quant_bits),
        label="draft ADC/interface",
    )
    draft_delta_dac_quant_bits = g._resolve_optional_quant_bits(
        bits=int(args.draft_delta_dac_bits),
        label="draft delta-readout DAC",
    )

    if verify_delta_dac_quant_bits and (not args.verify_delta_readout):
        raise ValueError("--verify_delta_dac_bits requires --verify_delta_readout.")
    if draft_delta_dac_quant_bits and (not args.draft_delta_readout):
        raise ValueError("--draft_delta_dac_bits requires --draft_delta_readout.")
    if args.draft_delta_readout and (not draft_quant_bits):
        raise ValueError("--draft_delta_readout requires --draft_adc_bits > 0 (or legacy --draft_post_matmul_quant_bits).")

    target = g._load_model(args.checkpoint_path, args.device, precision, use_tp=False, int8_act_quant=bool(args.int8_act_quant))
    if verify_quant_bits:
        from quantize import (
            set_post_matmul_output_quant_bits,
            set_post_matmul_output_quant_delta_dac_bits,
            set_post_matmul_output_quant_delta_mode,
        )

        set_post_matmul_output_quant_bits(target, int(verify_quant_bits))
        set_post_matmul_output_quant_delta_mode(target, delta_readout=bool(args.verify_delta_readout))
        set_post_matmul_output_quant_delta_dac_bits(target, bits=int(verify_delta_dac_quant_bits))
    elif args.verify_delta_readout:
        raise ValueError("--verify_delta_readout requires --verify_adc_bits > 0 (or legacy --post_matmul_quant_bits).")

    if args.draft_dequantize_int8:
        draft = g._load_int8_weight_only_as_fp_model(draft_checkpoint_path, args.draft_device, precision, use_tp=False)
    else:
        draft = g._load_model(draft_checkpoint_path, args.draft_device, precision, use_tp=False)
    if args.draft_fake_act_quant_int8:
        from quantize import replace_linear_fake_act_quant

        replace_linear_fake_act_quant(draft)
    if draft_quant_bits:
        from quantize import (
            set_post_matmul_output_quant_bits,
            set_post_matmul_output_quant_delta_dac_bits,
            set_post_matmul_output_quant_delta_mode,
        )

        set_post_matmul_output_quant_bits(draft, int(draft_quant_bits))
        set_post_matmul_output_quant_delta_mode(draft, delta_readout=bool(args.draft_delta_readout))
        set_post_matmul_output_quant_delta_dac_bits(draft, bits=int(draft_delta_dac_quant_bits))

    use_levels = (args.draft_noise_levels is not None) or (args.draft_noise_level_stds is not None)
    if use_levels:
        if args.draft_noise_levels is None or args.draft_noise_level_stds is None:
            raise ValueError("Level-based draft noise requires both --draft_noise_level_stds and --draft_noise_levels.")
        per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
            draft_noise_level_stds=args.draft_noise_level_stds,
            draft_noise_levels=args.draft_noise_levels,
            n_layer=len(draft.layers),
        )
    else:
        ffn_std, qkv_std, out_std = _coerce_draft_noise_stds(args.draft_noise_std)
        per_layer_stds = [(ffn_std, qkv_std, out_std) for _ in range(len(draft.layers))]
        output_std = float(out_std)
    if output_std > 0 or any((ffn > 0 or qkv > 0 or out > 0) for ffn, qkv, out in per_layer_stds):
        g.add_gaussian_noise_to_draft_weights_(
            draft,
            per_layer_stds=per_layer_stds,
            output_std=output_std,
            seed=int(args.draft_noise_seed),
        )

    results: Dict[str, Any] = {
        "run_id": args.run_id,
        "model": {
            "checkpoint_path": str(args.checkpoint_path),
            "draft_checkpoint_path": str(draft_checkpoint_path),
            "tokenizer_path": str(tokenizer_path),
        },
        "dataset": {
            "prompts_path": str(args.prompts),
            "prompt_field": str(args.prompt_field),
            "limit": None if args.limit is None else int(args.limit),
            "shuffle": bool(args.shuffle),
            "loaded_prompts": len(prompts),
            "prompt_length": int(args.prompt_length),
        },
        "knobs": {
            "device": str(args.device),
            "draft_device": str(args.draft_device),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "attention_backend": str(args.attention_backend),
            "read_noise_std": float(args.read_noise_std),
            "draft_noise_std": [float(x) for x in args.draft_noise_std] if args.draft_noise_std is not None else None,
            "draft_noise_level_stds": None if args.draft_noise_level_stds is None else [float(x) for x in args.draft_noise_level_stds],
            "draft_noise_levels": None if args.draft_noise_levels is None else [int(x) for x in args.draft_noise_levels],
            "draft_noise_seed": int(args.draft_noise_seed),
            "verify_adc_bits": int(verify_quant_bits),
            "verify_delta_readout": bool(args.verify_delta_readout),
            "verify_delta_dac_bits": int(verify_delta_dac_quant_bits),
            "verify_adc_quant_domain": "delta" if bool(args.verify_delta_readout) else "absolute",
            "draft_adc_bits": int(draft_quant_bits),
            "draft_delta_readout": bool(args.draft_delta_readout),
            "draft_delta_dac_bits": int(draft_delta_dac_quant_bits),
            "draft_adc_quant_domain": "delta" if bool(args.draft_delta_readout) else "absolute",
            "post_matmul_quant_bits": int(args.post_matmul_quant_bits),
            "draft_post_matmul_quant_bits": int(args.draft_post_matmul_quant_bits),
        },
        "results_by_k": {},
    }

    for k in [int(x) for x in args.k_values]:
        if k < 1:
            raise ValueError("--k_values must be >= 1")
        counts, meta = _aggregate_for_length(
            prompts=prompts,
            prompt_length=int(args.prompt_length),
            tokenizer=tokenizer,
            device=args.device,
            model=target,
            draft_model=draft,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            speculate_k=int(k),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
        )
        total_bursts = sum(counts)
        mean_accepted = sum(i * c for i, c in enumerate(counts)) / max(total_bursts, 1)
        acceptance_ratio = mean_accepted / float(max(k, 1))
        committed_tokens_per_burst = mean_accepted + 1.0
        results["results_by_k"][str(k)] = {
            "counts": counts,
            "total_bursts": total_bursts,
            "mean_accepted": mean_accepted,
            "acceptance_ratio": acceptance_ratio,
            "expected_committed_tokens_per_burst": committed_tokens_per_burst,
            "meta": meta,
        }

    ranked_by_mean = sorted(
        ((int(k), rec["mean_accepted"]) for k, rec in results["results_by_k"].items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    ranked_by_ratio = sorted(
        ((int(k), rec["acceptance_ratio"]) for k, rec in results["results_by_k"].items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    results["best_by_mean_accepted"] = {"k": ranked_by_mean[0][0], "mean_accepted": ranked_by_mean[0][1]}
    results["best_by_acceptance_ratio"] = {"k": ranked_by_ratio[0][0], "acceptance_ratio": ranked_by_ratio[0][1]}

    out_json = args.out_json or _default_out_path(args.run_id)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("k\tmean_accepted\tacceptance_ratio\tcommitted_per_burst\ttotal_bursts")
    for k, _ in sorted((int(k), None) for k in results["results_by_k"].keys()):
        rec = results["results_by_k"][str(k)]
        print(
            f"{k}\t"
            f"{rec['mean_accepted']:.6f}\t"
            f"{rec['acceptance_ratio']:.6f}\t"
            f"{rec['expected_committed_tokens_per_burst']:.6f}\t"
            f"{rec['total_bursts']}"
        )
    print()
    print(f"Best by mean_accepted: k={results['best_by_mean_accepted']['k']} mean_accepted={results['best_by_mean_accepted']['mean_accepted']:.6f}")
    print(f"Best by acceptance_ratio: k={results['best_by_acceptance_ratio']['k']} acceptance_ratio={results['best_by_acceptance_ratio']['acceptance_ratio']:.6f}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
