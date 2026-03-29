## Why

The roadmap's "Predictive Delta Readout" design (§3.2) is still missing from the functional simulator. Today the simulator only models ADC/interface precision as absolute-output fake quantization, which makes it impossible to study the roadmap claim that temporal locality can shrink the residual readout dynamic range and reduce the required physical ADC precision.

## What Changes

- Add optional predictive delta readout modes on top of the existing verify and draft ADC/interface quantization paths.
- Model draft and verify analog linear outputs relative to the previous reconstructed token output, with absolute quantization used for the first token in a stream.
- Allow optional DAC-side quantization of the stored draft/verify feedback baselines.
- Define when delta-readout state is preserved and when it is reset, including continuity from prompt prefill into decode within one generation stream.
- Export the selected draft/verify readout modes and DAC settings in stats and sweep metadata.

## Capabilities

### New Capabilities
- `predictive-delta-readout`: Simulate roadmap-style draft/verify delta-domain ADC readout using previous-token feedback.

### Modified Capabilities
- `adc-interface-quantization`: Allow draft and verify ADC/interface precision to operate in absolute or predictive-delta mode and record the selected modes and DAC settings in metadata.

## Impact

- **Code**: likely `quantize.py`, `generate.py`, `scripts/dataset_selfspec_stats.py`, `scripts/sweep_speculate_k.py`, and `scripts/bench_noisy_spec_decode.py`.
- **Outputs**: metadata gains explicit draft/verify delta-readout and DAC fields.
- **Simulation semantics**: draft and verify ADC bitwidths can be interpreted as physical delta-domain precision instead of only absolute-output precision.
