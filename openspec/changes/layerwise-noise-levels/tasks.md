## 1. CLI + Configuration Parsing

- [ ] 1.1 Add `--draft_noise_level_stds` and `--draft_noise_levels` CLI flags to `generate.py` (with help text and defaults)
- [ ] 1.2 Implement validation for level-based inputs (required std table, allowed lengths 1/3/3*n_layer, non-negative levels, in-range levels)
- [ ] 1.3 Implement precedence rules between level-based flags and legacy `--draft_noise_std` (warning + ignore legacy when levels are present)

## 2. Noise Resolution + Application

- [ ] 2.1 Implement “shape coercion” to produce per-layer (ffn/qkv/out) stds from level assignments + level std table
- [ ] 2.2 Refactor `add_gaussian_noise_to_draft_weights_` to apply per-layer bucket stds (extract `layer_idx` from `layers.<i>.…` parameter names)
- [ ] 2.3 Define and implement how `output.weight` picks its std (use OUT std or “last layer OUT” for per-layer configs) and keep the existing `counts` reporting

## 3. Backwards Compatibility + Error Messages

- [ ] 3.1 Keep existing `--draft_noise_std` behavior unchanged when no level-based flags are provided
- [ ] 3.2 Ensure all user-facing errors are actionable (show expected lengths, valid level range, and which input was invalid)
- [ ] 3.3 Add a small set of unit-style checks (or a lightweight test script) covering: broadcast (1), broadcast (3), per-layer (3*n_layer), out-of-range level, and missing level stds

## 4. Docs + Script Updates

- [ ] 4.1 Update `README.md` speculative sampling examples to demonstrate level-based configuration (and note legacy `--draft_noise_std` compatibility)
- [ ] 4.2 Update any repo scripts that pass `--draft_noise_std` to use the new level-based flags (or document why they remain legacy)
- [ ] 4.3 Run a local smoke check (`python generate.py --help` and one minimal speculative run path) to confirm flags parse and noise injection code paths execute
