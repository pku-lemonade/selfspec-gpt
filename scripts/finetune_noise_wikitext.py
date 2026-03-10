#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate as g
from model import set_attention_backend, set_read_noise_std
from tokenizer import get_tokenizer, resolve_tokenizer_path


def _coerce_write_noise_stds(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if len(vals) == 1:
        ffn = qkv = out = vals[0]
    elif len(vals) == 3:
        ffn, qkv, out = vals
    else:
        raise ValueError("--write_noise_std must have 1 or 3 values (FFN QKV OUT)")
    if ffn < 0 or qkv < 0 or out < 0:
        raise ValueError("--write_noise_std values must be >= 0")
    return {"ffn": float(ffn), "qkv": float(qkv), "out": float(out)}


def _iter_parquet_text(paths: Sequence[Path], *, text_key: str, batch_rows: int = 2048) -> Iterator[str]:
    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(columns=[text_key], batch_size=batch_rows):
            texts = batch.column(0).to_pylist()
            for text in texts:
                if not isinstance(text, str):
                    continue
                s = text.strip()
                if s:
                    yield s


class ParquetTokenStream:
    def __init__(
        self,
        *,
        parquet_paths: Sequence[Path],
        tokenizer,
        text_key: str = "text",
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> None:
        if len(parquet_paths) == 0:
            raise ValueError("No parquet files provided")
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.bos_id = int(tokenizer.bos_id())
        self.eos_id = int(tokenizer.eos_id())
        self._buffer: deque[int] = deque()
        self._reset_iter()

    def _reset_iter(self) -> None:
        self._iter = _iter_parquet_text(self.parquet_paths, text_key=self.text_key)

    def reset(self) -> None:
        self._buffer.clear()
        self._reset_iter()

    def _append_encoded(self, text: str) -> None:
        ids = self.tokenizer.encode(text)
        if len(ids) == 0:
            return
        if self.add_bos:
            self._buffer.append(self.bos_id)
        self._buffer.extend(int(t) for t in ids)
        if self.add_eos and self.eos_id >= 0:
            self._buffer.append(self.eos_id)

    def _fill(self, min_tokens: int) -> None:
        while len(self._buffer) < int(min_tokens):
            try:
                text = next(self._iter)
            except StopIteration:
                self._reset_iter()
                text = next(self._iter)
            self._append_encoded(text)

    def next_tokens(self, n_tokens: int) -> List[int]:
        n_tokens = int(n_tokens)
        self._fill(n_tokens)
        out = [self._buffer.popleft() for _ in range(n_tokens)]
        return out

    def next_batch(self, *, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(batch_size) * (int(seq_len) + 1)
        toks = self.next_tokens(n)
        x = torch.tensor(toks, dtype=torch.long, device=device).view(batch_size, seq_len + 1)
        return x[:, :seq_len], x[:, 1:]


def _bucket_for_param(name: str) -> Optional[str]:
    if name == "output.weight":
        return "out"
    if name.endswith("feed_forward.w1.weight") or name.endswith("feed_forward.w2.weight") or name.endswith("feed_forward.w3.weight"):
        return "ffn"
    if name.endswith("attention.wqkv.weight"):
        return "qkv"
    if name.endswith("attention.wo.weight"):
        return "out"
    return None


def _collect_write_noise_params(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    groups: Dict[str, List[torch.nn.Parameter]] = {"ffn": [], "qkv": [], "out": []}
    for name, param in model.named_parameters():
        if (not param.requires_grad) or (not param.is_floating_point()):
            continue
        bucket = _bucket_for_param(name)
        if bucket is None:
            continue
        groups[bucket].append(param)
    return groups


def _collect_cuda_indices(param_groups: Dict[str, Sequence[torch.nn.Parameter]]) -> List[int]:
    idx: set[int] = set()
    for params in param_groups.values():
        for p in params:
            if p.device.type == "cuda" and p.device.index is not None:
                idx.add(int(p.device.index))
    return sorted(idx)


@torch.no_grad()
def _apply_write_noise_(
    *,
    param_groups: Dict[str, Sequence[torch.nn.Parameter]],
    stds: Dict[str, float],
    seed: int,
    add: bool,
) -> None:
    if all(float(stds[k]) <= 0 for k in ("ffn", "qkv", "out")):
        return

    cuda_indices = _collect_cuda_indices(param_groups)
    with torch.random.fork_rng(devices=cuda_indices):
        torch.manual_seed(int(seed))
        for bucket in ("ffn", "qkv", "out"):
            std = float(stds[bucket])
            if std <= 0:
                continue
            for p in param_groups[bucket]:
                # Relative write noise with exact undo:
                # add=True:  p <- p * (1 + eps),  eps ~ N(0, std)
                # add=False: p <- p / (1 + eps)  (same seed -> same eps)
                factor = 1.0 + torch.randn_like(p) * std
                if add:
                    p.mul_(factor)
                else:
                    p.div_(factor)


@dataclass
class ModeDistribution:
    names: Tuple[str, ...]
    probs: Tuple[float, ...]

    @classmethod
    def from_args(cls, *, clean: float, write: float, read: float, both: float) -> "ModeDistribution":
        raw = {
            "clean": max(0.0, float(clean)),
            "write": max(0.0, float(write)),
            "read": max(0.0, float(read)),
            "both": max(0.0, float(both)),
        }
        total = sum(raw.values())
        if total <= 0:
            raw = {"clean": 1.0, "write": 0.0, "read": 0.0, "both": 0.0}
            total = 1.0
        names = tuple(raw.keys())
        probs = tuple(raw[k] / total for k in names)
        return cls(names=names, probs=probs)

    def sample(self, rng: random.Random) -> str:
        return rng.choices(self.names, weights=self.probs, k=1)[0]


def _default_wikitext_paths() -> Tuple[List[Path], List[Path]]:
    base = Path("datasets/Salesforce__wikitext/wikitext-103-raw-v1")
    train = [
        base / "train-00000-of-00002.parquet",
        base / "train-00001-of-00002.parquet",
    ]
    val = [base / "validation-00000-of-00001.parquet"]
    return train, val


def _require_files(paths: Sequence[Path], *, name: str) -> None:
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {name} parquet file(s): {missing}")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _lr_for_step(step: int, *, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    if step >= warmup_steps:
        return float(base_lr)
    return float(base_lr) * (float(step) / float(max(1, warmup_steps)))


@torch.no_grad()
def _grad_norm_and_finite(model: torch.nn.Module) -> Tuple[float, bool]:
    norms: List[torch.Tensor] = []
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if not torch.isfinite(g).all():
            return float("nan"), False
        norms.append(torch.linalg.vector_norm(g.float(), ord=2))
    if len(norms) == 0:
        return 0.0, True
    total = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return float(total.item()), bool(torch.isfinite(total).item())


@torch.no_grad()
def _scale_grads_(model: torch.nn.Module, scale: float) -> None:
    s = float(scale)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(s)


@torch.no_grad()
def _evaluate_clean_loss(
    *,
    model: torch.nn.Module,
    val_stream: ParquetTokenStream,
    device: str,
    batch_size: int,
    seq_len: int,
    eval_steps: int,
) -> float:
    model.eval()
    set_read_noise_std(0.0)
    val_stream.reset()
    input_pos = torch.arange(0, seq_len, device=device, dtype=torch.int64)
    losses: List[float] = []
    for _ in range(int(eval_steps)):
        x, y = val_stream.next_batch(batch_size=batch_size, seq_len=seq_len, device=device)
        logits = g.model_forward(model, x, input_pos)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def parse_args() -> argparse.Namespace:
    train_default, val_default = _default_wikitext_paths()

    parser = argparse.ArgumentParser(description="Fine-tune a model on WikiText with mixed write/read noise.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Base checkpoint (.pth) to fine-tune.")
    parser.add_argument("--output_path", type=Path, default=Path("checkpoints/Qwen/Qwen3-0.6B/model_wikitext_noise_ft.pth"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher_checkpoint_path", type=Path, default=None, help="Optional frozen teacher checkpoint (.pth) for clean consistency targets.")
    parser.add_argument("--teacher_device", type=str, default="cuda:1", help="Teacher device when --teacher_checkpoint_path is provided.")
    parser.add_argument("--attention_backend", type=str, choices=["flex", "sdpa"], default="sdpa")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--train_parquet", type=Path, nargs="+", default=train_default, help="Train parquet file(s).")
    parser.add_argument("--val_parquet", type=Path, nargs="+", default=val_default, help="Validation parquet file(s).")
    parser.add_argument("--text_key", type=str, default="text", help="Parquet text column.")

    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--noisy_ce_weight",
        type=float,
        default=1.0,
        help="Weight on standard CE loss computed from noisy forward.",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="If > 0 and write noise is active, add CE(noisy_logits, clean_argmax_tokens) with this weight.",
    )
    parser.add_argument(
        "--distill_kl_weight",
        type=float,
        default=0.0,
        help="If > 0 and write noise is active, add KL(student || clean teacher) with this weight.",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=1.0,
        help="Temperature used for KL teacher distillation.",
    )
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--sgd_momentum", type=float, default=0.9)

    parser.add_argument(
        "--write_noise_std",
        type=float,
        nargs="+",
        default=[1e-3, 1e-3, 1e-3],
        help="Relative write noise std (multiplicative). Provide 1 value (all buckets) or 3 values: FFN QKV OUT.",
    )
    parser.add_argument(
        "--read_noise_std",
        type=float,
        default=1e-4,
        help="Relative read noise std (multiplicative) used in read/both modes.",
    )

    parser.add_argument("--prob_clean", type=float, default=0.25, help="Sampling probability weight for clean mode.")
    parser.add_argument("--prob_write", type=float, default=0.25, help="Sampling probability weight for write-only mode.")
    parser.add_argument("--prob_read", type=float, default=0.25, help="Sampling probability weight for read-only mode.")
    parser.add_argument("--prob_both", type=float, default=0.25, help="Sampling probability weight for write+read mode.")

    parser.add_argument("--save_interval", type=int, default=50, help="Save intermediate checkpoint every N steps (0 disables).")
    parser.add_argument("--save_dir", type=Path, default=Path("out/finetune_qwen3_0p6b_wikitext_noise"))
    parser.add_argument(
        "--max_attempt_steps",
        type=int,
        default=0,
        help="Max train attempts including skipped non-finite batches. 0 defaults to max(2*max_steps, max_steps+100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(str(args.checkpoint_path))
    if args.teacher_checkpoint_path is not None and not args.teacher_checkpoint_path.is_file():
        raise FileNotFoundError(str(args.teacher_checkpoint_path))
    _require_files(args.train_parquet, name="train")
    _require_files(args.val_parquet, name="val")
    if args.seq_len < 8:
        raise ValueError("--seq_len must be >= 8")
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("--batch_size and --grad_accum_steps must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be >= 1")
    if float(args.noisy_ce_weight) < 0 or float(args.consistency_weight) < 0 or float(args.distill_kl_weight) < 0:
        raise ValueError("--noisy_ce_weight, --consistency_weight, and --distill_kl_weight must be >= 0")
    if float(args.distill_temperature) <= 0:
        raise ValueError("--distill_temperature must be > 0")

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    precision = torch.bfloat16
    set_attention_backend(args.attention_backend)
    set_read_noise_std(0.0)

    tokenizer_path = resolve_tokenizer_path(args.checkpoint_path.parent)
    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)

    print(f"Loading model from {args.checkpoint_path} on {args.device}")
    model = g._load_model(args.checkpoint_path, args.device, precision, use_tp=False)
    model.train()
    with torch.device(args.device):
        model.setup_caches(max_batch_size=int(args.batch_size), max_seq_length=int(args.seq_len))
    # For full-sequence training we do not use autoregressive KV caching across steps.
    # Keeping kv_cache tensors attached to previous-step graphs can break backward.
    for layer in model.layers:
        layer.attention.kv_cache = None
    input_pos = torch.arange(0, int(args.seq_len), device=args.device, dtype=torch.int64)

    teacher = None
    teacher_device = args.device
    teacher_input_pos = input_pos
    if args.teacher_checkpoint_path is not None:
        teacher_device = args.teacher_device or args.device
        print(f"Loading frozen teacher from {args.teacher_checkpoint_path} on {teacher_device}")
        teacher = g._load_model(args.teacher_checkpoint_path, teacher_device, precision, use_tp=False)
        teacher.eval()
        with torch.device(teacher_device):
            teacher.setup_caches(max_batch_size=int(args.batch_size), max_seq_length=int(args.seq_len))
        for layer in teacher.layers:
            layer.attention.kv_cache = None
        teacher_input_pos = torch.arange(0, int(args.seq_len), device=teacher_device, dtype=torch.int64)

    write_noise_stds = _coerce_write_noise_stds(args.write_noise_std)
    read_noise_std = max(0.0, float(args.read_noise_std))
    mode_dist = ModeDistribution.from_args(
        clean=args.prob_clean,
        write=args.prob_write,
        read=args.prob_read,
        both=args.prob_both,
    )
    mode_rng = random.Random(int(args.seed) + 17)

    noise_params = _collect_write_noise_params(model)
    print(
        "Write-noise params (num tensors): "
        f"ffn={len(noise_params['ffn'])}, qkv={len(noise_params['qkv'])}, out={len(noise_params['out'])}"
    )
    print(f"Noise mix probs: {dict(zip(mode_dist.names, mode_dist.probs))}")
    print(f"Write noise stds: {write_noise_stds}, read noise std: {read_noise_std}")

    train_stream = ParquetTokenStream(
        parquet_paths=args.train_parquet,
        tokenizer=tokenizer,
        text_key=args.text_key,
        add_bos=False,
        add_eos=True,
    )
    val_stream = ParquetTokenStream(
        parquet_paths=args.val_parquet,
        tokenizer=tokenizer,
        text_key=args.text_key,
        add_bos=False,
        add_eos=True,
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            momentum=float(args.sgd_momentum),
            weight_decay=float(args.weight_decay),
            nesterov=False,
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    mode_counts = {"clean": 0, "write": 0, "read": 0, "both": 0}
    tokens_per_step = int(args.batch_size) * int(args.seq_len) * int(args.grad_accum_steps)
    train_start = time.perf_counter()

    max_steps = int(args.max_steps)
    max_attempt_steps = int(args.max_attempt_steps)
    if max_attempt_steps <= 0:
        max_attempt_steps = max(2 * max_steps, max_steps + 100)

    step = 0
    attempt = 0
    skipped_nonfinite_steps = 0

    while step < max_steps and attempt < max_attempt_steps:
        attempt += 1
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_main_loss = 0.0
        accum_cons_loss = 0.0
        accum_distill_loss = 0.0
        bad_step = False
        bad_reason = ""

        for micro in range(int(args.grad_accum_steps)):
            mode = mode_dist.sample(mode_rng)
            mode_counts[mode] += 1

            use_write = mode in ("write", "both")
            use_read = mode in ("read", "both")
            this_read_noise = read_noise_std if use_read else 0.0
            needs_clean_teacher = use_write and (
                float(args.consistency_weight) > 0 or float(args.distill_kl_weight) > 0
            )
            set_read_noise_std(0.0 if needs_clean_teacher else this_read_noise)

            noise_seed = int(args.seed) + attempt * 100003 + micro
            try:
                x, y = train_stream.next_batch(batch_size=int(args.batch_size), seq_len=int(args.seq_len), device=args.device)
                clean_targets: Optional[torch.Tensor] = None
                clean_teacher_logits: Optional[torch.Tensor] = None
                if needs_clean_teacher:
                    with torch.no_grad():
                        if teacher is not None:
                            clean_logits = g.model_forward(teacher, x.to(teacher_device), teacher_input_pos)
                            if float(args.consistency_weight) > 0:
                                clean_targets = clean_logits.argmax(dim=-1).to(args.device)
                            if float(args.distill_kl_weight) > 0:
                                clean_teacher_logits = clean_logits.to(args.device)
                        else:
                            clean_logits = g.model_forward(model, x, input_pos)
                            if float(args.consistency_weight) > 0:
                                clean_targets = clean_logits.argmax(dim=-1)
                            if float(args.distill_kl_weight) > 0:
                                clean_teacher_logits = clean_logits.detach()

                set_read_noise_std(this_read_noise)
                if use_write:
                    _apply_write_noise_(param_groups=noise_params, stds=write_noise_stds, seed=noise_seed, add=True)

                logits = g.model_forward(model, x, input_pos)
                if not torch.isfinite(logits).all():
                    bad_step = True
                    bad_reason = "nonfinite_logits"
                    break
                main_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                if not torch.isfinite(main_loss):
                    bad_step = True
                    bad_reason = "nonfinite_loss"
                    break
                total_loss = float(args.noisy_ce_weight) * main_loss

                cons_loss_val = 0.0
                if clean_targets is not None and float(args.consistency_weight) > 0:
                    cons_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), clean_targets.reshape(-1))
                    if not torch.isfinite(cons_loss):
                        bad_step = True
                        bad_reason = "nonfinite_consistency_loss"
                        break
                    total_loss = total_loss + float(args.consistency_weight) * cons_loss
                    cons_loss_val = float(cons_loss.item())

                distill_loss_val = 0.0
                if clean_teacher_logits is not None and float(args.distill_kl_weight) > 0:
                    temp = float(args.distill_temperature)
                    student_log_probs = F.log_softmax(
                        logits.float().reshape(-1, logits.size(-1)) / temp,
                        dim=-1,
                    )
                    teacher_probs = F.softmax(
                        clean_teacher_logits.float().reshape(-1, clean_teacher_logits.size(-1)) / temp,
                        dim=-1,
                    )
                    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
                    if not torch.isfinite(distill_loss):
                        bad_step = True
                        bad_reason = "nonfinite_distill_loss"
                        break
                    total_loss = total_loss + float(args.distill_kl_weight) * distill_loss
                    distill_loss_val = float(distill_loss.item())

                if not torch.isfinite(total_loss):
                    bad_step = True
                    bad_reason = "nonfinite_total_loss"
                    break

                (total_loss / float(args.grad_accum_steps)).backward()
                accum_main_loss += float(main_loss.item())
                accum_cons_loss += cons_loss_val
                accum_distill_loss += distill_loss_val
                accum_loss += float(total_loss.item())
            finally:
                if use_write:
                    _apply_write_noise_(param_groups=noise_params, stds=write_noise_stds, seed=noise_seed, add=False)

            grad_norm, grads_finite = _grad_norm_and_finite(model)
            if (not grads_finite) or (not math.isfinite(grad_norm)):
                bad_step = True
                bad_reason = "nonfinite_grad"
                break

        set_read_noise_std(0.0)
        if bad_step:
            skipped_nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"[skip] attempt={attempt:04d} reason={bad_reason}")
            continue

        step += 1
        lr_now = _lr_for_step(step, base_lr=float(args.lr), warmup_steps=int(args.warmup_steps))
        _set_optimizer_lr(optimizer, lr_now)

        if float(args.max_grad_norm) > 0:
            grad_norm, grads_finite = _grad_norm_and_finite(model)
            if (not grads_finite) or (not math.isfinite(grad_norm)):
                skipped_nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"[skip] attempt={attempt:04d} reason=nonfinite_grad_preclip")
                step -= 1
                continue
            max_grad_norm = float(args.max_grad_norm)
            if grad_norm > max_grad_norm:
                _scale_grads_(model, max_grad_norm / (grad_norm + 1e-6))
        else:
            grad_norm = 0.0
        optimizer.step()

        step_time = time.perf_counter() - t0
        train_loss = accum_loss / float(args.grad_accum_steps)
        train_main_loss = accum_main_loss / float(args.grad_accum_steps)
        train_cons_loss = accum_cons_loss / float(args.grad_accum_steps)
        train_distill_loss = accum_distill_loss / float(args.grad_accum_steps)

        if int(args.log_interval) > 0 and (step % int(args.log_interval) == 0):
            tok_s = float(tokens_per_step) / max(step_time, 1e-9)
            print(
                f"step={step:04d} attempt={attempt:04d} "
                f"loss={train_loss:.4f} main={train_main_loss:.4f} cons={train_cons_loss:.4f} distill={train_distill_loss:.4f} lr={lr_now:.3e} "
                f"grad_norm={grad_norm:.3f} tok/s={tok_s:.1f}"
            )

        if int(args.eval_interval) > 0 and (step % int(args.eval_interval) == 0):
            val_loss = _evaluate_clean_loss(
                model=model,
                val_stream=val_stream,
                device=args.device,
                batch_size=int(args.batch_size),
                seq_len=int(args.seq_len),
                eval_steps=int(args.eval_steps),
            )
            print(f"[eval] step={step:04d} clean_val_loss={val_loss:.4f} ppl={torch.exp(torch.tensor(val_loss)).item():.2f}")

        if int(args.save_interval) > 0 and (step % int(args.save_interval) == 0):
            ckpt_path = args.save_dir / f"model_step_{step:04d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved intermediate checkpoint: {ckpt_path}")

    if step < max_steps:
        raise RuntimeError(
            f"Training stopped early: completed {step}/{max_steps} steps after {attempt} attempts "
            f"(skipped_nonfinite_steps={skipped_nonfinite_steps}). "
            "Increase --max_attempt_steps or adjust training settings."
        )

    elapsed = time.perf_counter() - train_start
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    set_read_noise_std(0.0)

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "teacher_checkpoint_path": None if args.teacher_checkpoint_path is None else str(args.teacher_checkpoint_path),
        "teacher_device": None if args.teacher_checkpoint_path is None else str(teacher_device),
        "output_path": str(args.output_path),
        "noisy_ce_weight": float(args.noisy_ce_weight),
        "consistency_weight": float(args.consistency_weight),
        "distill_kl_weight": float(args.distill_kl_weight),
        "distill_temperature": float(args.distill_temperature),
        "train_parquet": [str(p) for p in args.train_parquet],
        "val_parquet": [str(p) for p in args.val_parquet],
        "steps": int(step),
        "attempt_steps": int(attempt),
        "skipped_nonfinite_steps": int(skipped_nonfinite_steps),
        "target_steps": int(max_steps),
        "tokens_per_step": int(tokens_per_step),
        "elapsed_sec": float(elapsed),
        "mode_counts": {k: int(v) for k, v in mode_counts.items()},
        "write_noise_stds": write_noise_stds,
        "read_noise_std": float(read_noise_std),
        "mode_probs": dict(zip(mode_dist.names, mode_dist.probs)),
    }
    summary_path = args.save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved final checkpoint: {args.output_path}")
    print(f"wrote run summary: {summary_path}")


if __name__ == "__main__":
    main()
