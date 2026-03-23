## ADDED Requirements

### Requirement: Train the noisy student with draft ADC quantization
The finetuning pipeline SHALL allow the student model to simulate draft-side ADC-style interface quantization during training.

#### Scenario: Student draft ADC bits are configured
- **WHEN** the user provides a student draft ADC precision setting
- **THEN** the training loop applies that ADC-style interface quantization to the student model's analog linear outputs during forward passes

### Requirement: Allow an optional teacher ADC precision
The finetuning pipeline SHALL allow the frozen clean teacher to use a separate verify-side ADC precision setting during distillation.

#### Scenario: Teacher ADC bits are configured
- **WHEN** the user provides a teacher ADC precision setting
- **THEN** the training loop applies that ADC-style interface quantization to the teacher model's analog linear outputs when producing distillation targets

### Requirement: Preserve roadmap-style self-target evaluation
ADC-aware finetuning outputs SHALL remain evaluable under the roadmap objective.

#### Scenario: ADC-aware checkpoint is evaluated
- **WHEN** an ADC-aware tuned checkpoint is evaluated under roadmap-style acceptance
- **THEN** the simulator supports `target = tuned clean`, `draft = tuned + noise`, with configurable draft and verify ADC-style interface quantization
