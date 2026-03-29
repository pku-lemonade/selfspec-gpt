## 1. OpenSpec Artifacts

- [x] 1.1 Write the proposal for predictive delta readout in roadmap terms
- [x] 1.2 Add the new `predictive-delta-readout` capability spec
- [x] 1.3 Add the `adc-interface-quantization` delta spec for verify delta mode
- [x] 1.4 Write the technical design for stateful delta-domain verify quantization
- [x] 1.5 Update the proposal/design/specs after draft-side delta-readout support was added

## 2. Simulator Support

- [x] 2.1 Add a stateful verify-side delta readout mode to the output-quantization helpers in `quantize.py`
- [x] 2.2 Add model-wide reset helpers for delta-readout state and wire them into `generate.py`
- [x] 2.3 Add `--verify_delta_readout` to `generate.py` and validate that it requires verify ADC bits
- [x] 2.4 Thread verify delta readout through `scripts/dataset_selfspec_stats.py`
- [x] 2.5 Thread verify delta readout through `scripts/sweep_speculate_k.py` and `scripts/bench_noisy_spec_decode.py`
- [x] 2.6 Export verify delta-readout mode in stats and sweep metadata
- [x] 2.7 Add optional DAC quantization for verify-side delta feedback
- [x] 2.8 Add draft-side delta readout mode and `--draft_delta_readout`
- [x] 2.9 Add optional DAC quantization for draft-side delta feedback
- [x] 2.10 Export draft delta-readout mode and DAC settings in stats and sweep metadata

## 3. Validation

- [x] 3.1 Run syntax / import checks for the touched scripts
- [x] 3.2 Run a small generate smoke test comparing absolute vs. delta verify ADC mode
- [x] 3.3 Run one dataset acceptance export check and confirm metadata records the delta mode
- [x] 3.4 Run verify-side DAC sensitivity checks
- [x] 3.5 Run draft-side delta-readout and DAC sensitivity checks
- [x] 3.6 Run the full draft-ADC by `k` matrix under draft+verify delta readout with `8-bit` DACs
