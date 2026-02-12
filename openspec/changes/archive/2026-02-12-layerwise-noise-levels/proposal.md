## Why

Draft weight noise is currently configured as a single std (or a single FFN/QKV/OUT triple) applied uniformly across all transformer layers, which makes it hard to run experiments where noise varies by depth and is cumbersome to manage at scale. A discrete “noise level” abstraction with shared stds also simplifies configuration and keeps noise settings consistent across repeated uses.

## What Changes

- Add **layer-wise** configuration for draft weight noise so each transformer layer can independently specify noise settings for the three existing buckets: **QKV**, **OUT**, and **FFN**.
- Replace per-bucket `noise_std` inputs with per-bucket **noise levels** (e.g., integer level IDs), where a shared mapping defines `level -> noise_std` and all components using the same level share the same std.
- Provide a configuration surface to express:
  - the shared level→std table
  - per-layer per-bucket level assignments (QKV/OUT/FFN)
- Update docs and scripts that currently use `--draft_noise_std` to use the new level-based, layer-wise configuration.
- Keep the existing global `--draft_noise_std` behavior as a backwards-compatible shortcut where possible (or clearly deprecate it if removal is required). **BREAKING** only if the existing CLI/config is removed or its meaning changes.

## Capabilities

### New Capabilities
- `draft-layerwise-noise`: Configure draft-model post-load weight noise per layer and per bucket (FFN/QKV/OUT) using discrete noise levels backed by a shared level→std mapping.

### Modified Capabilities
<!-- None (no existing OpenSpec capabilities in this repo yet). -->

## Impact

- Affects draft-weight noise configuration and application code paths (e.g., `generate.py` draft noise args and the draft-weight noise injection logic).
- Affects documentation (`README.md`) and any benchmarking/launch scripts that pass draft noise settings.
- May require a migration path for existing users/scripts that rely on `--draft_noise_std` (depending on compatibility decisions).
