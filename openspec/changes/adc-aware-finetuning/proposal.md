## Why

The simulator now exposes ADC-style interface quantization for draft and verify paths, but the current finetuning loop does not train checkpoints against that degradation. For roadmap-style deployment, draft accuracy should be optimized under both write noise and draft ADC quantization, especially for the strongest current model (`Qwen3-1.7B`).

## What Changes

- Add ADC-aware finetuning controls so the noisy student can train with draft-side interface quantization enabled.
- Allow the frozen clean teacher to optionally use a separate verify-side ADC precision during distillation.
- Run a 1.7B ADC-aware self-target refinement pilot from the current best `1.7B` checkpoint.
- Evaluate the pilot under roadmap-style acceptance with ADC quantization enabled.

## Capabilities

### New Capabilities
- `adc-aware-finetuning`: Train a noisy draft model while simulating draft/verify ADC-style interface quantization during distillation.

### Modified Capabilities
- `adc-interface-quantization`: Extend the simulator feature so training can consume the same ADC-style interface quantization knobs used in inference/evaluation.

## Impact

- **Code**: `scripts/finetune_noise_wikitext.py`
- **Evaluation**: roadmap-style 1.7B acceptance runs with ADC quantization enabled
- **Artifacts**: new finetune summaries and corrected acceptance result files for ADC-aware checkpoints
