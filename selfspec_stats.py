from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git(repo_root: Path, args: Sequence[str]) -> str | None:
    try:
        out = subprocess.check_output(  # noqa: S603,S607
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def get_git_info(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    dirty = None
    status = _run_git(repo_root, ["status", "--porcelain"])
    if status is not None:
        dirty = len(status) > 0
    return {"commit": commit, "dirty": dirty}


def accept_counts_to_stats(accept_counts: Sequence[int], *, k: int | None = None) -> dict[str, Any]:
    if accept_counts is None or len(accept_counts) == 0:
        raise ValueError("accept_counts must be a non-empty sequence of length K+1")

    inferred_k = len(accept_counts) - 1
    if k is None:
        k = inferred_k
    k = int(k)
    if k != inferred_k:
        raise ValueError(f"accept_counts length {len(accept_counts)} implies k={inferred_k}, got k={k}")
    if k < 0:
        raise ValueError(f"k must be >= 0; got k={k}")

    histogram = {int(a): int(v) for a, v in enumerate(accept_counts)}
    payload: dict[str, Any] = {"k": k, "histogram": histogram}
    validate_stats_payload(payload)
    return payload


def _coerce_histogram(hist: Mapping[Any, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for raw_a, raw_v in hist.items():
        try:
            a = int(raw_a)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"histogram key must be an int; got {raw_a!r}") from exc
        try:
            v = float(raw_v)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"histogram value must be a number; got a={raw_a!r} val={raw_v!r}") from exc
        out[a] = v
    return out


def validate_stats_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"stats payload must be a mapping; got {type(payload)}")

    if "k" not in payload:
        raise ValueError("stats payload missing required field: k")
    if "histogram" not in payload:
        raise ValueError("stats payload missing required field: histogram")

    k = int(payload["k"])
    if k < 0:
        raise ValueError(f"stats.k must be >= 0; got {k}")

    hist_raw = payload["histogram"]
    if not isinstance(hist_raw, Mapping):
        raise ValueError(f"stats.histogram must be a mapping; got {type(hist_raw)}")

    hist = _coerce_histogram(hist_raw)
    if len(hist) == 0:
        raise ValueError("stats.histogram must not be empty")

    for a, v in hist.items():
        if a < 0 or a > k:
            raise ValueError(f"histogram bin out of range: a={a} for k={k}")
        if v < 0:
            raise ValueError(f"histogram value must be non-negative: a={a} val={v}")

    expected_bins = set(range(0, k + 1))
    missing = expected_bins.difference(hist.keys())
    if missing:
        missing_list = ", ".join(str(x) for x in sorted(missing))
        raise ValueError(f"histogram missing required bins for k={k}: {missing_list}")

    total = float(sum(hist.values()))
    if total <= 0:
        raise ValueError("histogram sum must be > 0")


def stats_meta_path_for(stats_path: Path) -> Path:
    name = stats_path.name
    if name.startswith("stats") and name.endswith(".json"):
        meta_name = "stats_meta" + name[len("stats") :]
        return stats_path.with_name(meta_name)
    return stats_path.with_name("stats_meta.json")


def resolve_stats_out(stats_out: str | Path) -> Tuple[Path, Path]:
    p = Path(stats_out)
    is_dir = (p.exists() and p.is_dir()) or (p.suffix.lower() != ".json")
    stats_path = p / "stats.json" if is_dir else p
    meta_path = stats_meta_path_for(stats_path)
    return stats_path, meta_path


def summarize_histogram(histogram: Mapping[int, float], *, k: int) -> dict[str, Any]:
    hist = {int(a): float(v) for a, v in histogram.items()}
    total = float(sum(hist.values()))
    mean_accepted = sum(float(a) * v for a, v in hist.items()) / max(total, 1e-12)
    probs = [0.0] * (k + 1)
    if total > 0:
        for a in range(0, k + 1):
            probs[a] = float(hist.get(a, 0.0)) / total
    return {
        "total_bursts": total,
        "mean_accepted": mean_accepted,
        "acceptance_probs": probs,
    }


def build_stats_meta(
    *,
    stats: Mapping[str, Any],
    run_id: str | None = None,
    repo_root: Path | None = None,
    paths: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    generation: Mapping[str, Any] | None = None,
    knobs: Mapping[str, Any] | None = None,
    seeds: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    aggregation: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_stats_payload(stats)

    k = int(stats["k"])
    histogram = _coerce_histogram(stats["histogram"])
    summary = summarize_histogram(histogram, k=k)

    meta: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "stats": {
            "k": k,
            "counts_kind": "counts",
            **summary,
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    if repo_root is not None:
        meta["git"] = get_git_info(repo_root)

    def _maybe_add(key: str, value: Mapping[str, Any] | None) -> None:
        if value is None:
            return
        if isinstance(value, Mapping) and len(value) == 0:
            return
        meta[key] = dict(value)

    _maybe_add("paths", paths)
    _maybe_add("model", model)
    _maybe_add("generation", generation)
    _maybe_add("knobs", knobs)
    _maybe_add("seeds", seeds)
    _maybe_add("dataset", dataset)
    _maybe_add("aggregation", aggregation)
    _maybe_add("extra", extra)

    return meta


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
