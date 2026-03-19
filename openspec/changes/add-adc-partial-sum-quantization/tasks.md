## 1. OpenSpec Artifacts

- [x] 1.1 Write the proposal for ADC-style partial-sum quantization in roadmap terms
- [x] 1.2 Add the new `adc-interface-quantization` capability spec
- [x] 1.3 Add the `draft-layerwise-noise` delta spec describing composition with ADC quantization
- [x] 1.4 Write the technical design for mapping ADC knobs onto the existing output-quantization path

## 2. Simulator Knobs

- [x] 2.1 Add explicit `--verify_adc_bits` and `--draft_adc_bits` flags to `generate.py`
- [x] 2.2 Map the new ADC flags onto the existing post-matmul output quantization implementation
- [x] 2.3 Define and enforce precedence / validation when legacy and new flags are both provided

## 3. Tooling Integration

- [x] 3.1 Thread the new ADC flags through `scripts/dataset_selfspec_stats.py`
- [x] 3.2 Thread the new ADC flags through `scripts/bench_noisy_spec_decode.py`
- [x] 3.3 Thread the new ADC flags through `scripts/sweep_speculate_k.py`
- [x] 3.4 Export the resolved ADC quantization settings in metadata and sweep outputs

## 4. Validation

- [x] 4.1 Run syntax / import checks for the touched scripts
- [x] 4.2 Run a small generate smoke test using the new ADC-style flags
- [x] 4.3 Run one small dataset acceptance check to verify the metadata and behavior
