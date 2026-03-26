## 1. OpenSpec Artifacts

- [x] 1.1 Write the proposal for predictive delta readout in roadmap terms
- [x] 1.2 Add the new `predictive-delta-readout` capability spec
- [x] 1.3 Add the `adc-interface-quantization` delta spec for verify delta mode
- [x] 1.4 Write the technical design for stateful delta-domain verify quantization

## 2. Simulator Support

- [ ] 2.1 Add a stateful verify-side delta readout mode to the output-quantization helpers in `quantize.py`
- [ ] 2.2 Add model-wide reset helpers for delta-readout state and wire them into `generate.py`
- [ ] 2.3 Add `--verify_delta_readout` to `generate.py` and validate that it requires verify ADC bits
- [ ] 2.4 Thread verify delta readout through `scripts/dataset_selfspec_stats.py`
- [ ] 2.5 Thread verify delta readout through `scripts/sweep_speculate_k.py` and `scripts/bench_noisy_spec_decode.py`
- [ ] 2.6 Export verify delta-readout mode in stats and sweep metadata

## 3. Validation

- [ ] 3.1 Run syntax / import checks for the touched scripts
- [ ] 3.2 Run a small generate smoke test comparing absolute vs. delta verify ADC mode
- [ ] 3.3 Run one dataset acceptance export check and confirm metadata records the delta mode
