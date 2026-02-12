from __future__ import annotations

from typing import List, Sequence, Tuple

NoiseLevelsByLayer = List[Tuple[int, int, int]]  # (ffn, qkv, out) per layer
NoiseStdsByLayer = List[Tuple[float, float, float]]  # (ffn, qkv, out) per layer


def _coerce_level_stds(level_stds: Sequence[float]) -> List[float]:
    if level_stds is None:
        raise ValueError("draft_noise_level_stds is required when using draft noise levels")
    if len(level_stds) == 0:
        raise ValueError("draft_noise_level_stds must have at least 1 value")

    out: List[float] = []
    for idx, v in enumerate(level_stds):
        fv = float(v)
        if fv < 0:
            raise ValueError(f"draft_noise_level_stds[{idx}] must be >= 0; got {fv}")
        out.append(fv)
    return out


def coerce_draft_noise_levels(levels: Sequence[int], *, n_layer: int) -> NoiseLevelsByLayer:
    if n_layer <= 0:
        raise ValueError(f"n_layer must be > 0; got {n_layer}")
    if levels is None or len(levels) == 0:
        raise ValueError("draft_noise_levels must be a non-empty list of integers")

    level_list = [int(x) for x in levels]
    if len(level_list) == 1:
        v = level_list[0]
        return [(v, v, v) for _ in range(n_layer)]
    if len(level_list) == 3:
        ffn_level, qkv_level, out_level = level_list
        return [(ffn_level, qkv_level, out_level) for _ in range(n_layer)]

    expected = 3 * n_layer
    if len(level_list) == expected:
        per_layer: NoiseLevelsByLayer = []
        for layer_idx in range(n_layer):
            ffn_level, qkv_level, out_level = level_list[3 * layer_idx : 3 * layer_idx + 3]
            per_layer.append((ffn_level, qkv_level, out_level))
        return per_layer

    raise ValueError(
        "draft_noise_levels must have length 1, 3, or 3*n_layer "
        f"(expected 1, 3, or {expected} for n_layer={n_layer}); got {len(level_list)}"
    )


def resolve_level_based_draft_noise_stds(
    *,
    draft_noise_level_stds: Sequence[float],
    draft_noise_levels: Sequence[int],
    n_layer: int,
) -> Tuple[NoiseStdsByLayer, float]:
    level_stds = _coerce_level_stds(draft_noise_level_stds)
    per_layer_levels = coerce_draft_noise_levels(draft_noise_levels, n_layer=n_layer)

    num_levels = len(level_stds)
    per_layer_stds: NoiseStdsByLayer = []
    for layer_idx, (ffn_level, qkv_level, out_level) in enumerate(per_layer_levels):
        for bucket, level in (("ffn", ffn_level), ("qkv", qkv_level), ("out", out_level)):
            if level < 0 or level >= num_levels:
                raise ValueError(
                    f"Invalid draft noise level for {bucket} at layer {layer_idx}: {level}. "
                    f"Valid range is [0, {num_levels - 1}] "
                    f"(draft_noise_level_stds length {num_levels})."
                )
        per_layer_stds.append((level_stds[ffn_level], level_stds[qkv_level], level_stds[out_level]))

    # Keep current behavior: treat output projection noise as OUT bucket noise.
    # For per-layer configs, use the last layer's OUT level/std.
    output_std = per_layer_stds[-1][2]
    return per_layer_stds, output_std
