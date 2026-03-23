## 1. OpenSpec Artifacts

- [x] 1.1 Write the proposal for ADC-aware finetuning
- [x] 1.2 Add the `adc-aware-finetuning` capability spec
- [x] 1.3 Add the `adc-interface-quantization` delta spec for training-path support
- [x] 1.4 Write the technical design for ADC-aware self-target refinement

## 2. Finetuning Support

- [ ] 2.1 Add student and teacher ADC-interface quantization flags to `scripts/finetune_noise_wikitext.py`
- [ ] 2.2 Apply student ADC quantization to the noisy student forward path
- [ ] 2.3 Apply optional teacher ADC quantization to the frozen teacher forward path

## 3. Validation

- [ ] 3.1 Run syntax/import validation after adding the training flags
- [ ] 3.2 Run a 1.7B ADC-aware self-target refinement pilot on `cuda:1`
- [ ] 3.3 Evaluate the pilot under roadmap-style ADC acceptance and compare to the current 1.7B winner
