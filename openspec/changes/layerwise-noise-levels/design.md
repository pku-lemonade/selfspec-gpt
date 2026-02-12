## Context

`generate.py` currently supports adding Gaussian noise to *draft* model weights after load via `--draft_noise_std`, with either:

- 1 value: apply the same std to all supported weights, or
- 3 values: apply different stds for **FFN / QKV / OUT**.

This noise is applied uniformly across *all* transformer layers by matching parameter name suffixes in `add_gaussian_noise_to_draft_weights_` (e.g., `layers.*.attention.wqkv.weight`, `layers.*.attention.wo.weight`, `layers.*.feed_forward.w{1,2,3}.weight`). The code also includes `output.weight` in the OUT bucket.

We want two changes:

1. Make draft noise configuration **layer-wise** (each layer independently configurable for FFN/QKV/OUT).
2. Replace “direct std per bucket” with discrete **noise levels**, where components that share the same level share the same underlying std via a global `level -> std` mapping.

## Goals / Non-Goals

**Goals:**
- Support per-layer, per-bucket draft noise configuration for FFN/QKV/OUT.
- Introduce a shared `noise_level -> noise_std` mapping and apply noise by level, not by directly specifying std per bucket.
- Preserve determinism via the existing `--draft_noise_seed` behavior.
- Provide a backwards-compatible path for existing `--draft_noise_std` usage (or a clear deprecation with a migration path).

**Non-Goals:**
- Change runtime “read noise” (`--read_noise_std`) semantics.
- Add per-head/per-channel/per-weight noise configuration.
- Apply noise dynamically per forward pass (this remains “post-load, one-time” weight perturbation).
- Define a general configuration system beyond the draft-noise feature scope.

## Decisions

### 1) Configuration surface: “level stds” + “level assignments”

**Decision:** Add a new level-based configuration pair:
- A global list/table of stds indexed by integer noise level (e.g., level 0, 1, 2, …).
- A per-layer, per-bucket assignment of integer levels for (FFN, QKV, OUT).

This enables the user to say “layer 0 QKV uses level 1” and “layer 4 OUT uses level 3”, while keeping the underlying std defined once per level.

**Alternatives considered:**
- Keep std-based config but add per-layer triples of floats (meets layer-wise goal but does not meet the “level” abstraction goal, and is harder to keep consistent).
- Introduce only sparse overrides without a full per-layer representation (compact, but requires a more complex parsing format and careful precedence rules).

### 2) CLI encoding

**Decision:** Implement a “shape-coerced” encoding similar to the existing `--draft_noise_std` behavior, but for levels.

Proposed flags:
- `--draft_noise_level_stds <float...>`: list of stds where index == level (e.g., `0 1e-4 2e-4`).
- `--draft_noise_levels <int...>`: level assignments, accepted shapes:
  - 1 value: apply the same level to all buckets in all layers
  - 3 values: apply levels for (FFN, QKV, OUT) uniformly across all layers
  - `n_layer * 3` values: per-layer triplets in (FFN, QKV, OUT) order for each layer `0..n_layer-1`

**Alternatives considered:**
- JSON/YAML config file (more expressive and sparse-friendly; more moving parts and file management in launch scripts).
- Repeated “override” flags like `--draft_noise_level qkv:4:3` (sparse and readable; more parsing and potential for user mistakes).

### 3) How to treat `output.weight`

**Decision:** Continue supporting `output.weight` noise injection as part of the OUT bucket, and resolve its level/std as follows:
- If `--draft_noise_levels` is provided as `n_layer * 3`, use the **last layer’s OUT level** for `output.weight`.
- If levels are provided as 1 or 3 values, use the configured OUT level.

**Alternatives considered:**
- Introduce a dedicated `--draft_noise_output_level` (explicit but adds another knob).
- Exclude `output.weight` from noise entirely (simplifies configuration but changes current behavior).

### 4) Backwards compatibility and precedence

**Decision:** Keep `--draft_noise_std` as a supported compatibility path for now.

Precedence:
- If any level-based flags are provided (`--draft_noise_level_stds` / `--draft_noise_levels`), use the level-based path and ignore `--draft_noise_std` (emit a warning).
- Otherwise, fall back to the existing `--draft_noise_std` behavior unchanged.

This avoids breaking existing scripts while enabling the new configuration.

### 5) Implementation approach (draft noise injection)

**Decision:** Refactor the draft noise injection logic to resolve a per-layer std triple and apply noise using the existing parameter-name matching, with added layer-index extraction:
- Extract `layer_idx` from parameter names that start with `layers.<idx>.`
- Determine bucket by suffix:
  - FFN: `.feed_forward.w{1,2,3}.weight`
  - QKV: `.attention.wqkv.weight`
  - OUT: `.attention.wo.weight`
  - Output projection: `output.weight` (special-cased)
- Lookup std via `std = level_stds[level_assignment[layer_idx][bucket]]` (and the chosen std for `output.weight`)
- Apply once using the existing RNG seed semantics and keep the existing “counts” reporting.

## Risks / Trade-offs

- **User configuration errors (length mismatch):** `n_layer*3` requires the model’s `n_layer` to be known; add explicit validation errors showing expected vs provided lengths.
- **Level range issues:** Ensure levels are `>= 0` and `< len(level_stds)`; otherwise error with a clear message.
- **Compatibility complexity:** Keeping both `--draft_noise_std` and level-based flags introduces precedence rules; mitigate with clear warnings and README updates.
- **Indexing confusion:** If we later add sparse override formats, we must clearly define whether layer indices are 0-based or 1-based; avoid ambiguous formats in the first iteration.

## Migration Plan

1. Add the new CLI flags and implement level-based noise injection.
2. Update README and launch scripts to prefer the level-based configuration.
3. Keep `--draft_noise_std` functioning; optionally mark as deprecated in docs (with a removal date only if desired).

## Open Questions

- Should we also support a sparse override format (e.g., “only specify a few layers”) in the initial implementation, or defer until the list-based approach is validated?
- Do we want explicit control for `output.weight` via a dedicated flag, or is “use last layer’s OUT” sufficient?
