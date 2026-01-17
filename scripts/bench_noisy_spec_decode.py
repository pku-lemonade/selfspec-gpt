import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate as g
import model as model_lib
from quantize import replace_linear_fake_act_quant
from tokenizer import get_tokenizer


def _parse_noise_sweep(value: str) -> List[float]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    values = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if any(v < 0 for v in values):
        raise ValueError("noise_std values must be >= 0")
    return values


def _sync_devices(target_device: str, draft_device: Optional[str]) -> List[str]:
    if not draft_device or draft_device == target_device:
        return [target_device]
    return [target_device, draft_device]


def _run_one(
    model: g.Transformer,
    encoded_prompt: torch.Tensor,
    prompt_length: int,
    *,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    draft_model: Optional[g.Transformer],
    speculate_k: int,
    sync_devices: List[str],
    batch_size: int = 1,
) -> Tuple[float, List[int]]:
    g.device_sync(sync_devices)
    t0 = time.perf_counter()
    seq, stats = g.generate(
        model,
        encoded_prompt,
        max_new_tokens,
        batch_size=batch_size,
        draft_model=draft_model,
        speculate_k=speculate_k,
        interactive=False,
        callback=lambda _: None,
        temperature=temperature,
        top_k=top_k,
    )
    g.device_sync(sync_devices)
    elapsed = time.perf_counter() - t0
    tokens_generated = int(seq.size(-1) - prompt_length)
    tokens_per_sec = tokens_generated / max(elapsed, 1e-9)
    accept_counts = stats.get("accept_counts") or [0] * (speculate_k + 1)
    return tokens_per_sec, [int(x) for x in accept_counts]


def _aggregate_accept(accept_counts_runs: List[List[int]]) -> Tuple[List[int], float, List[float]]:
    if not accept_counts_runs:
        return [], 0.0, []
    k_plus_1 = len(accept_counts_runs[0])
    counts = [0] * k_plus_1
    for run in accept_counts_runs:
        if len(run) != k_plus_1:
            raise ValueError("inconsistent accept_counts lengths")
        for i, v in enumerate(run):
            counts[i] += int(v)
    total = sum(counts)
    if total == 0:
        return counts, 0.0, [0.0] * k_plus_1
    mean_accepted = sum(i * c for i, c in enumerate(counts)) / total
    probs = [c / total for c in counts]
    return counts, mean_accepted, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding with noisy draft weights (dual GPU).")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/modelscope/Llama-2-7b-chat-ms/model.pth"),
        help="Path to model.pth for both target and draft.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device.")
    parser.add_argument("--draft_device", type=str, default="cuda:1", help="Draft device.")
    parser.add_argument(
        "--draft_dequantize_int8",
        action="store_true",
        help="If set, load the draft from an int8 weight-only checkpoint but dequantize to fp weights for draft inference.",
    )
    parser.add_argument(
        "--int8_act_quant",
        action="store_true",
        help="If set (and checkpoint is int8), quantize activations per-token and run int8xint8 matmuls for target linear layers.",
    )
    parser.add_argument(
        "--draft_fake_act_quant_int8",
        action="store_true",
        help="If set, apply per-token int8 fake activation quantization to the draft model linears (still runs fp matmuls).",
    )
    parser.add_argument("--speculate_k", type=int, default=5, help="Speculative depth (k).")
    parser.add_argument("--prompt", type=str, default="Hi my name is", help="Prompt text.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Tokens to generate per run.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of runs to average per noise_std.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--noise_sweep",
        type=str,
        default="0,1e-4,3e-4,1e-3,3e-3,5e-3,7e-3,1e-2,2e-2,3e-2",
        help="Comma-separated draft noise std values to test (ascending).",
    )
    parser.add_argument(
        "--noise_bucket",
        type=str,
        choices=["all", "ffn", "qkv", "out"],
        default="all",
        help="Which weights to perturb on the draft model: all, ffn, qkv, or out (output projections).",
    )
    parser.add_argument("--draft_noise_seed", type=int, default=1234, help="Base seed for draft weight noise.")
    parser.add_argument("--sample_seed", type=int, default=2026, help="Base seed for sampling/acceptance RNG.")
    parser.add_argument(
        "--min_mean_accepted",
        type=float,
        default=3.0,
        help="Minimum mean accepted draft tokens (out of k) considered 'not too bad'.",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        choices=["flex", "sdpa"],
        default="flex",
        help="Attention backend to use. Use sdpa as a stability fallback if flex_attention crashes.",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended).")
    parser.add_argument("--compile_prefill", action="store_true", help="Also compile prefill (slower compile, faster prefill).")
    args = parser.parse_args()

    model_lib.set_attention_backend(args.attention_backend)

    checkpoint_path: Path = args.checkpoint_path
    assert checkpoint_path.is_file(), str(checkpoint_path)
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    noise_sweep = _parse_noise_sweep(args.noise_sweep)
    if not noise_sweep:
        raise ValueError("--noise_sweep must contain at least one value")
    if noise_sweep != sorted(noise_sweep):
        raise ValueError("--noise_sweep must be in ascending order")

    precision = torch.bfloat16
    print(f"Loading target model on {args.device} ...")
    model = g._load_model(checkpoint_path, args.device, precision, use_tp=False, int8_act_quant=bool(args.int8_act_quant))
    print(f"Loading draft model on {args.draft_device} ...")
    if args.draft_dequantize_int8:
        draft_model = g._load_int8_weight_only_as_fp_model(checkpoint_path, args.draft_device, precision, use_tp=False)
    else:
        draft_model = g._load_model(checkpoint_path, args.draft_device, precision, use_tp=False)
    if args.draft_fake_act_quant_int8:
        replace_linear_fake_act_quant(draft_model)

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    encoded = g.encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)
    prompt_length = int(encoded.size(-1))

    if args.compile:
        print("Compiling kernels (torch.compile) ...")
        g.decode_one_token = torch.compile(g.decode_one_token, mode="reduce-overhead", fullgraph=True)
        g.model_forward = torch.compile(g.model_forward, mode="reduce-overhead", fullgraph=True)
        if args.compile_prefill:
            g.prefill = torch.compile(g.prefill, fullgraph=True, dynamic=True)

    sync_devices_base = _sync_devices(args.device, None)
    sync_devices_spec = _sync_devices(args.device, args.draft_device)

    # Warmups so compile time isn't counted.
    torch.manual_seed(args.sample_seed)
    _run_one(
        model,
        encoded,
        prompt_length,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        draft_model=None,
        speculate_k=args.speculate_k,
        sync_devices=sync_devices_base,
    )
    torch.manual_seed(args.sample_seed)
    _run_one(
        model,
        encoded,
        prompt_length,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        draft_model=draft_model,
        speculate_k=args.speculate_k,
        sync_devices=sync_devices_spec,
    )

    # Baseline: target-only.
    base_tok_s: List[float] = []
    for i in range(args.num_samples):
        torch.manual_seed(args.sample_seed + i)
        tok_s, _ = _run_one(
            model,
            encoded,
            prompt_length,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            draft_model=None,
            speculate_k=args.speculate_k,
            sync_devices=sync_devices_base,
        )
        base_tok_s.append(tok_s)
    baseline = sum(base_tok_s) / len(base_tok_s)
    print(f"Baseline target-only: {baseline:.2f} tok/s (n={args.num_samples})")
    print()

    applied_noise_std = 0.0
    best_noise_std = None
    results = []

    for idx, noise_std in enumerate(noise_sweep):
        if noise_std < applied_noise_std:
            raise ValueError("noise_sweep must be ascending when using incremental noise application")
        inc = math.sqrt(max(noise_std * noise_std - applied_noise_std * applied_noise_std, 0.0))
        if inc > 0:
            seed = args.draft_noise_seed + idx
            if args.noise_bucket == "all":
                g.add_gaussian_noise_to_model_weights_(draft_model, inc, seed)
            elif args.noise_bucket == "ffn":
                g.add_gaussian_noise_to_draft_weights_(draft_model, ffn_std=inc, qkv_std=0.0, out_std=0.0, seed=seed)
            elif args.noise_bucket == "qkv":
                g.add_gaussian_noise_to_draft_weights_(draft_model, ffn_std=0.0, qkv_std=inc, out_std=0.0, seed=seed)
            elif args.noise_bucket == "out":
                g.add_gaussian_noise_to_draft_weights_(draft_model, ffn_std=0.0, qkv_std=0.0, out_std=inc, seed=seed)
            else:
                raise ValueError(f"Unknown noise_bucket: {args.noise_bucket}")
            applied_noise_std = noise_std

        tok_s_runs: List[float] = []
        accept_runs: List[List[int]] = []
        for i in range(args.num_samples):
            torch.manual_seed(args.sample_seed + i)
            tok_s, accept_counts = _run_one(
                model,
                encoded,
                prompt_length,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
                draft_model=draft_model,
                speculate_k=args.speculate_k,
                sync_devices=sync_devices_spec,
            )
            tok_s_runs.append(tok_s)
            accept_runs.append(accept_counts)

        tok_s_mean = sum(tok_s_runs) / len(tok_s_runs)
        _, mean_accepted, _ = _aggregate_accept(accept_runs)
        speedup = tok_s_mean / max(baseline, 1e-9)
        ok = mean_accepted >= float(args.min_mean_accepted)
        if ok:
            best_noise_std = noise_std
        results.append((noise_std, tok_s_mean, speedup, mean_accepted))

        status = "OK" if ok else "LOW_ACCEPT"
        print(
            f"noise_std={noise_std:.6g}  tok/s={tok_s_mean:7.2f}  speedup={speedup:5.2f}  mean_accepted={mean_accepted:4.2f}  {status}"
        )

    print()
    if best_noise_std is None:
        print(f"Max acceptable noise_std (mean_accepted>={args.min_mean_accepted:.2f}): NONE")
    else:
        print(f"Max acceptable noise_std (mean_accepted>={args.min_mean_accepted:.2f}): {best_noise_std:.6g}")


if __name__ == "__main__":
    main()
