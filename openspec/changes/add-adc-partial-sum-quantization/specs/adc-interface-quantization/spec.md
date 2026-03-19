## ADDED Requirements

### Requirement: Configure draft and verify ADC interface precision
The simulator SHALL allow users to configure separate ADC-style interface quantization settings for draft-mode and verify-mode analog linear outputs.

#### Scenario: Draft ADC precision is configured
- **WHEN** the user provides a draft ADC precision setting
- **THEN** the simulator applies that precision only to the draft model's analog linear outputs

#### Scenario: Verify ADC precision is configured
- **WHEN** the user provides a verify ADC precision setting
- **THEN** the simulator applies that precision only to the target / verify model's analog linear outputs

### Requirement: Model ADC precision as partial-sum / interface quantization
The simulator SHALL treat ADC precision as quantization of analog partial-sum outputs after linear accumulation, not as weight quantization.

#### Scenario: ADC quantization is applied after analog linear output formation
- **WHEN** ADC interface quantization is enabled for a linear layer
- **THEN** the simulator fake-quantizes the layer output after the matrix multiply / accumulation stage

#### Scenario: Non-linear digital operators remain unquantized by ADC knobs
- **WHEN** ADC interface quantization is enabled
- **THEN** the simulator does not apply those ADC knobs directly to non-linear digital operators such as softmax or elementwise activation functions

### Requirement: Preserve backwards-compatible post-matmul quantization flags
The simulator SHALL preserve the existing post-matmul fake quantization CLI for compatibility while exposing roadmap-aligned ADC interface names.

#### Scenario: Existing post-matmul flags are used
- **WHEN** the user provides existing post-matmul quantization flags
- **THEN** the simulator continues to run and maps the behavior consistently to the analog linear output quantization path

#### Scenario: Explicit ADC flags and legacy flags conflict
- **WHEN** the user provides both explicit ADC interface flags and legacy post-matmul flags for the same path
- **THEN** the simulator exits with an actionable error or applies a documented precedence rule

### Requirement: Export ADC interface precision in metadata
The simulator SHALL record the ADC-style interface quantization settings in run metadata and sweep outputs.

#### Scenario: Dataset acceptance stats are exported
- **WHEN** dataset or single-run stats metadata is written
- **THEN** the metadata includes the configured draft and verify ADC interface precision settings

#### Scenario: K-sweep results are exported
- **WHEN** the K sweep script writes its JSON summary
- **THEN** the summary includes the configured draft and verify ADC interface precision settings
