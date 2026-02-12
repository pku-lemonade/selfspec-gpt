# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask as _create_block_mask

create_block_mask = _create_block_mask

def device_sync(device):
    if device is None:
        return
    if isinstance(device, (list, tuple)):
        for d in device:
            device_sync(d)
        return

    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        # torch.cuda.synchronize accepts either a device index or torch.device.
        if dev.index is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(dev)
    elif dev.type in ("cpu", "mps"):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from model import ModelArgs, set_attention_backend, set_read_noise_std
from tokenizer import get_tokenizer, resolve_tokenizer_path
from draft_noise import resolve_level_based_draft_noise_stds
from selfspec_stats import accept_counts_to_stats, build_stats_meta, resolve_stats_out, write_json

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    return model(mask, x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    target_device = cur_token.device
    draft_device = draft_model.output.weight.device
    cur_token_draft = cur_token.to(device=draft_device)
    orig_input_pos_draft = torch.tensor([input_pos], dtype=torch.int64, device=draft_device)
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model,
        cur_token_draft.view(1, -1),
        orig_input_pos_draft.clone(),
        speculate_k,
        **sampling_kwargs,
    )

    draft_tokens_draft = torch.cat(draft_tokens)
    draft_tokens = draft_tokens_draft.to(device=target_device)
    draft_token_ids = draft_tokens.view(-1).to(dtype=torch.long)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1, 1), draft_tokens], dim=0).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=target_device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.cat(draft_probs, dim=0).to(device=target_device)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    positions = torch.arange(0, speculate_k, device=target_device)
    p = draft_probs[positions, draft_token_ids]
    q = target_probs[positions, draft_token_ids]
    accept_draft_prob = torch.minimum(torch.ones_like(q), q / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        last_token = multinomial_sample_one_no_sync(target_probs[-1].unsqueeze(0))
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens_draft[-1].view(1, -1),
            orig_input_pos_draft + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new.unsqueeze(0))
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    if is_speculative:
        assert batch_size == 1, "Speculative decoding currently supports batch_size=1"
        draft_device = draft_model.output.weight.device
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            with torch.device(draft_device):
                draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        draft_prompt = prompt.to(device=draft_device)
        draft_input_pos = input_pos.to(device=draft_device)
        prefill(draft_model, draft_prompt.view(batch_size, -1), draft_input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[:, input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added].view(1, -1)
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[num_added - 1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp, *, int8_act_quant: bool = False):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer(ModelArgs.from_checkpoint_dir(checkpoint_path.parent))

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model, act_quant=bool(int8_act_quant))
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _dequantize_int8_weight_only_state_dict(
    state_dict: Mapping[str, Any],
    *,
    dtype: torch.dtype,
) -> dict:
    # gpt-fast int8 checkpoints store per-channel symmetric int8 weights and per-row scales:
    #   <fqn>.weight  (int8) and <fqn>.scales (bf16)
    # Runtime computes: y = linear(x, int8_weight) * scales
    # Equivalent float weight: weight_fp = int8_weight * scales[:, None]
    out = dict(state_dict)
    scale_keys = [k for k in out.keys() if k.endswith(".scales")]
    for scales_key in scale_keys:
        weight_key = scales_key[: -len(".scales")] + ".weight"
        if weight_key not in out:
            continue
        weight = out[weight_key]
        scales = out[scales_key]
        if not isinstance(weight, torch.Tensor) or not isinstance(scales, torch.Tensor):
            continue
        if weight.dtype is not torch.int8:
            continue
        if scales.dim() != 1:
            continue
        # Dequantize to float weights expected by the fp model.
        weight_fp = weight.to(dtype=dtype) * scales.to(dtype=dtype).unsqueeze(1)
        out[weight_key] = weight_fp
        # Remove int8-only aux tensor for fp model.
        out.pop(scales_key, None)
    return out


def _load_int8_weight_only_as_fp_model(checkpoint_path: Path, device, precision, use_tp):
    with torch.device("meta"):
        model = Transformer(ModelArgs.from_checkpoint_dir(checkpoint_path.parent))

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    checkpoint = _dequantize_int8_weight_only_state_dict(checkpoint, dtype=precision)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


@torch.no_grad()
def add_gaussian_noise_to_model_weights_(model: Transformer, std: float, seed: int) -> None:
    if std <= 0:
        return
    device = model.output.weight.device
    devices = [device.index] if device.type == "cuda" else None
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for _, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            param.add_(torch.randn_like(param) * std)


def _coerce_draft_noise_stds(draft_noise_std: Union[float, Sequence[float]]) -> Tuple[float, float, float]:
    if isinstance(draft_noise_std, (int, float)):
        v = float(draft_noise_std)
        return v, v, v
    if isinstance(draft_noise_std, (list, tuple)):
        if len(draft_noise_std) == 1:
            v = float(draft_noise_std[0])
            return v, v, v
        if len(draft_noise_std) == 3:
            return float(draft_noise_std[0]), float(draft_noise_std[1]), float(draft_noise_std[2])
    raise ValueError("draft_noise_std must be 1 value or 3 values (FFN QKV OUT)")


@torch.no_grad()
def add_gaussian_noise_to_draft_weights_(
    model: Transformer,
    *,
    per_layer_stds: Sequence[Tuple[float, float, float]],
    output_std: float,
    seed: int,
) -> dict:
    counts = {"ffn": 0, "qkv": 0, "out": 0}
    if len(per_layer_stds) == 0:
        return counts
    if (output_std <= 0) and all((ffn <= 0 and qkv <= 0 and out <= 0) for ffn, qkv, out in per_layer_stds):
        return counts

    device = model.output.weight.device
    devices = [device.index] if device.type == "cuda" else None
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            std = 0.0
            bucket = None

            if name == "output.weight":
                std = float(output_std)
                bucket = "out"
            elif name.startswith("layers."):
                rest = name[len("layers.") :]
                idx_str, _, suffix = rest.partition(".")
                if not idx_str.isdigit():
                    continue
                layer_idx = int(idx_str)
                if layer_idx < 0 or layer_idx >= len(per_layer_stds):
                    continue

                ffn_std, qkv_std, out_std = per_layer_stds[layer_idx]
                if suffix.endswith("feed_forward.w1.weight") or suffix.endswith("feed_forward.w2.weight") or suffix.endswith("feed_forward.w3.weight"):
                    std = float(ffn_std)
                    bucket = "ffn"
                elif suffix.endswith("attention.wqkv.weight"):
                    std = float(qkv_std)
                    bucket = "qkv"
                elif suffix.endswith("attention.wo.weight"):
                    std = float(out_std)
                    bucket = "out"

            if std <= 0:
                continue

            param.add_(torch.randn_like(param) * std)
            counts[bucket] += param.numel()
    return counts

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    compile_block_mask: bool = True,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    draft_device: Optional[str] = None,
    draft_noise_std: Union[float, Sequence[float]] = 0.0,
    draft_noise_level_stds: Optional[Sequence[float]] = None,
    draft_noise_levels: Optional[Sequence[int]] = None,
    draft_noise_seed: int = 1234,
    draft_dequantize_int8: bool = False,
    draft_fake_act_quant_int8: bool = False,
    int8_act_quant: bool = False,
    post_matmul_quant_bits: int = 0,
    draft_post_matmul_quant_bits: int = 0,
    speculate_k: int = 5,
    read_noise_std: float = 0.0,
    attention_backend: str = "flex",
    stats_out: Optional[Path] = None,
    no_stats_meta: bool = False,
    device=default_device,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = resolve_tokenizer_path(checkpoint_path.parent)
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)
    if draft_device is None:
        draft_device = device
    sync_devices = [device] if (not is_speculative or draft_device == device) else [device, draft_device]

    print(f"Using device={device}")
    if is_speculative and draft_device != device:
        print(f"Using draft_device={draft_device}")

    if stats_out is not None and not is_speculative:
        raise ValueError("--stats_out requires speculative decoding. Provide --draft_checkpoint_path.")

    set_attention_backend(attention_backend)
    set_read_noise_std(read_noise_std)
    if read_noise_std > 0:
        print(f"Enabling per-matmul read noise: std={read_noise_std}")

    global create_block_mask
    if compile_block_mask:
        create_block_mask = torch.compile(_create_block_mask)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp, int8_act_quant=int8_act_quant)

    if post_matmul_quant_bits:
        from quantize import set_post_matmul_output_quant_bits

        set_post_matmul_output_quant_bits(model, post_matmul_quant_bits)

    if is_speculative:
        if draft_dequantize_int8:
            draft_model = _load_int8_weight_only_as_fp_model(draft_checkpoint_path, draft_device, precision, use_tp)
        else:
            draft_model = _load_model(draft_checkpoint_path, draft_device, precision, use_tp)
        if draft_fake_act_quant_int8:
            from quantize import replace_linear_fake_act_quant

            replace_linear_fake_act_quant(draft_model)
        n_layer = len(draft_model.layers)
        use_levels = (draft_noise_levels is not None) or (draft_noise_level_stds is not None)

        per_layer_stds: Sequence[Tuple[float, float, float]]
        output_std: float

        if use_levels:
            if draft_noise_levels is None or draft_noise_level_stds is None:
                raise ValueError("Level-based draft noise requires both --draft_noise_level_stds and --draft_noise_levels.")
            if isinstance(draft_noise_std, (list, tuple)) and list(draft_noise_std) != [0.0]:
                print("WARNING: ignoring --draft_noise_std because level-based draft noise flags were provided")

            per_layer_stds, output_std = resolve_level_based_draft_noise_stds(
                draft_noise_level_stds=draft_noise_level_stds,
                draft_noise_levels=draft_noise_levels,
                n_layer=n_layer,
            )
            ffn0, qkv0, out0 = per_layer_stds[0]
            ffn_last, qkv_last, out_last = per_layer_stds[-1]
            print(
                "Adding Gaussian noise to draft weights (levels): "
                f"n_layer={n_layer}, seed={draft_noise_seed}, "
                f"layer0(ffn,qkv,out)=({ffn0},{qkv0},{out0}), "
                f"layerLast(ffn,qkv,out)=({ffn_last},{qkv_last},{out_last}), "
                f"output_std={output_std}"
            )
        else:
            ffn_std, qkv_std, out_std = _coerce_draft_noise_stds(draft_noise_std)
            per_layer_stds = [(ffn_std, qkv_std, out_std) for _ in range(n_layer)]
            output_std = float(out_std)
            if ffn_std > 0 or qkv_std > 0 or out_std > 0:
                print(
                    "Adding Gaussian noise to draft weights: "
                    f"ffn_std={ffn_std}, qkv_std={qkv_std}, out_std={out_std}, seed={draft_noise_seed}"
                )

        if output_std > 0 or any((ffn > 0 or qkv > 0 or out > 0) for ffn, qkv, out in per_layer_stds):
            counts = add_gaussian_noise_to_draft_weights_(
                draft_model,
                per_layer_stds=per_layer_stds,
                output_std=output_std,
                seed=draft_noise_seed,
            )
            print(f"Noised params (numel): ffn={counts['ffn']}, qkv={counts['qkv']}, out={counts['out']}")

        if draft_post_matmul_quant_bits:
            from quantize import set_post_matmul_output_quant_bits

            set_post_matmul_output_quant_bits(draft_model, draft_post_matmul_quant_bits)
    else:
        draft_model = None

    device_sync(sync_devices) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    if isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # generate a fully synthetic prompt
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(sync_devices) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        if is_speculative:
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(sync_devices) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            # Just displaying the first generation
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
    print()
    print("==========")
    counts_aggregated = None
    if is_speculative:
        if aggregate_metrics["accept_counts"]:
            counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
            total_bursts = sum(counts_aggregated)
            if total_bursts > 0:
                acceptance_probs = [c / total_bursts for c in counts_aggregated]
                mean_accepted = sum(idx * c for idx, c in enumerate(counts_aggregated)) / total_bursts
                print(f"Acceptance probs: {acceptance_probs}")
                print(f"Mean Accepted: {mean_accepted}")
            else:
                print("Acceptance probs: N/A (no speculative bursts recorded)")
        else:
            print("Acceptance probs: N/A (no samples recorded)")

    if stats_out is not None and (not use_tp or rank == 0):
        if counts_aggregated is None:
            raise ValueError("No speculative acceptance histogram available for --stats_out export.")

        stats_path, meta_path = resolve_stats_out(stats_out)
        stats_payload = accept_counts_to_stats(counts_aggregated, k=speculate_k)
        write_json(stats_path, stats_payload)

        if not no_stats_meta:
            repo_root = Path(__file__).resolve().parent
            run_id = stats_path.parent.name or stats_path.stem
            meta_payload = build_stats_meta(
                stats=stats_payload,
                run_id=run_id,
                repo_root=repo_root,
                paths={
                    "stats": str(stats_path),
                    "stats_meta": str(meta_path),
                },
                model={
                    "checkpoint_path": str(checkpoint_path),
                    "draft_checkpoint_path": str(draft_checkpoint_path) if draft_checkpoint_path is not None else None,
                    "tokenizer_path": str(tokenizer_path),
                },
                generation={
                    "prompt_length": int(prompt_length),
                    "max_new_tokens": int(max_new_tokens),
                    "num_samples": int(num_samples),
                    "batch_size": int(batch_size),
                    "top_k": int(top_k),
                    "temperature": float(temperature),
                    "speculate_k": int(speculate_k),
                    "is_chat": bool(is_chat),
                },
                knobs={
                    "attention_backend": str(attention_backend),
                    "read_noise_std": float(read_noise_std),
                    "draft_noise_std": list(draft_noise_std) if isinstance(draft_noise_std, (list, tuple)) else float(draft_noise_std),
                    "draft_noise_level_stds": None if draft_noise_level_stds is None else [float(x) for x in draft_noise_level_stds],
                    "draft_noise_levels": None if draft_noise_levels is None else [int(x) for x in draft_noise_levels],
                    "draft_noise_seed": int(draft_noise_seed),
                    "draft_dequantize_int8": bool(draft_dequantize_int8),
                    "draft_fake_act_quant_int8": bool(draft_fake_act_quant_int8),
                    "int8_act_quant": bool(int8_act_quant),
                    "post_matmul_quant_bits": int(post_matmul_quant_bits),
                    "draft_post_matmul_quant_bits": int(draft_post_matmul_quant_bits),
                    "compile": bool(compile),
                    "compile_prefill": bool(compile_prefill),
                    "compile_block_mask": bool(compile_block_mask),
                    "device": str(device),
                    "draft_device": str(draft_device),
                },
                seeds={
                    "sample_seed": 1234,
                },
                aggregation={
                    "compile_warmup_excluded": bool(compile),
                    "total_bursts": int(sum(counts_aggregated)),
                },
            )
            write_json(meta_path, meta_payload)

        print(f"Wrote stats to: {stats_path}")
        if not no_stats_meta:
            print(f"Wrote stats meta to: {meta_path}")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    if "cuda" in device:
        target_mem = torch.cuda.max_memory_reserved(torch.device(device)) / 1e9
        if is_speculative and draft_device != device and "cuda" in draft_device:
            draft_mem = torch.cuda.max_memory_reserved(torch.device(draft_device)) / 1e9
            print(f"Memory used (target): {target_mem:.02f} GB")
            print(f"Memory used (draft): {draft_mem:.02f} GB")
        else:
            print(f"Memory used: {target_mem:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", help="Input prompt. If it's an integer, will instead generate a synthetic prompt.")
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument(
        '--no_compile_block_mask',
        action='store_true',
        help='Disable torch.compile for create_block_mask (stability fallback).',
    )
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--draft_device', type=str, default=None, help='Device for the draft model (defaults to --device).')
    parser.add_argument(
        '--draft_dequantize_int8',
        action='store_true',
        help='If set, treat the draft checkpoint as an int8 weight-only checkpoint and dequantize it to fp weights for draft inference.',
    )
    parser.add_argument(
        '--draft_fake_act_quant_int8',
        action='store_true',
        help='If set, apply per-token int8 fake activation quantization to the draft model linears (still runs fp matmuls).',
    )
    parser.add_argument(
        '--int8_act_quant',
        action='store_true',
        help='If set (and checkpoint is int8), quantize activations per-token and run int8xint8 matmuls for linear layers (target model).',
    )
    parser.add_argument(
        '--post_matmul_quant_bits',
        type=int,
        default=0,
        help='If non-zero, fake-quantize the output of each linear matmul per token to this many bits (supported: 8, 16).',
    )
    parser.add_argument(
        '--draft_post_matmul_quant_bits',
        type=int,
        default=0,
        help='Same as --post_matmul_quant_bits but applied to the draft model.',
    )
    parser.add_argument(
        '--draft_noise_std',
        type=float,
        nargs='+',
        default=[0.0],
        help='Gaussian noise std(s) to add to draft model weights after load. Provide 1 value (all) or 3 values: FFN QKV OUT.',
    )
    parser.add_argument(
        '--draft_noise_level_stds',
        type=float,
        nargs='+',
        default=None,
        help='Draft noise level std table. Index i is noise level i. Requires --draft_noise_levels. Overrides --draft_noise_std.',
    )
    parser.add_argument(
        '--draft_noise_levels',
        type=int,
        nargs='+',
        default=None,
        help='Draft noise level assignments. Provide 1 value (all), 3 values (FFN QKV OUT), or 3*n_layer values (per-layer triplets in FFN QKV OUT order). Requires --draft_noise_level_stds.',
    )
    parser.add_argument('--draft_noise_seed', type=int, default=1234, help='RNG seed for draft weight noise.')
    parser.add_argument(
        '--read_noise_std',
        type=float,
        default=0.0,
        help='Per-matmul Gaussian read-noise std for stationary fp weights. 0 disables runtime read noise.',
    )
    parser.add_argument(
        '--attention_backend',
        type=str,
        choices=['flex', 'sdpa'],
        default='flex',
        help='Attention backend to use. Use sdpa as a stability fallback if flex_attention crashes.',
    )
    parser.add_argument(
        '--stats_out',
        type=Path,
        default=None,
        help='Write calculator-compatible speculation stats JSON. If path is a directory (or has no .json suffix), writes <path>/stats.json.',
    )
    parser.add_argument(
        '--no_stats_meta',
        action='store_true',
        help='Disable writing stats_meta.json sidecar when exporting stats.',
    )
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, (args.compile and (not args.no_compile_block_mask)), args.profile, args.draft_checkpoint_path,
        args.draft_device,
        args.draft_noise_std,
        args.draft_noise_level_stds,
        args.draft_noise_levels,
        args.draft_noise_seed,
        args.draft_dequantize_int8,
        args.draft_fake_act_quant_int8,
        args.int8_act_quant,
        args.post_matmul_quant_bits,
        args.draft_post_matmul_quant_bits,
        args.speculate_k,
        args.read_noise_std,
        args.attention_backend,
        args.stats_out,
        args.no_stats_meta,
        args.device,
    )
