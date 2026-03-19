## Why

The roadmap requires the functional simulator to model finite DAC/ADC interface precision as part of the analog read path, especially low-resolution draft-mode ADC behavior. Today the simulator exposes only generic post-matmul fake quantization flags, which are not expressed in roadmap terms and are easy to misuse as generic quantization rather than analog interface / partial-sum quantization.

## What Changes

- Add explicit simulator knobs for draft and verify ADC-style interface quantization.
- Define ADC quantization as partial-sum / analog-interface quantization applied after analog linear accumulations, not as weight quantization.
- Support separate draft and target ADC bit settings so draft mode can be coarse while verify mode remains higher precision.
- Preserve the existing post-matmul quantization flags as compatibility aliases where practical.
- Record the ADC-interface quantization knobs in exported metadata and sweep tooling outputs.

## Capabilities

### New Capabilities
- `adc-interface-quantization`: Configure and apply ADC-style partial-sum quantization for analog linear outputs in draft and verify paths.

### Modified Capabilities
- `draft-layerwise-noise`: Clarify that draft noise can compose with ADC-interface quantization during speculative simulation.

## Impact

- **Code**: `generate.py`, `scripts/dataset_selfspec_stats.py`, `scripts/bench_noisy_spec_decode.py`, `scripts/sweep_speculate_k.py`, and quantization helpers in `quantize.py`.
- **Outputs**: metadata JSONs gain explicit ADC-interface quantization fields.
- **Simulation semantics**: draft/verify precision can be expressed in roadmap-aligned terms instead of generic post-matmul fake quantization.
