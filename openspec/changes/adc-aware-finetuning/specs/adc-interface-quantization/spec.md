## MODIFIED Requirements

### Requirement: Configure draft and verify ADC interface precision
The simulator SHALL allow users to configure separate ADC-style interface quantization settings for draft-mode and verify-mode analog linear outputs, and SHALL make those settings available to both inference/evaluation and finetuning paths.

#### Scenario: Draft ADC precision is configured
- **WHEN** the user provides a draft ADC precision setting
- **THEN** the simulator applies that precision to the draft model's analog linear outputs in inference, evaluation, and finetuning

#### Scenario: Verify ADC precision is configured
- **WHEN** the user provides a verify ADC precision setting
- **THEN** the simulator applies that precision to the target / verify model's analog linear outputs in inference, evaluation, and teacher distillation paths
