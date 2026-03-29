## ADDED Requirements

### Requirement: Configure predictive delta readout for draft and verify ADC paths
The simulator SHALL allow users to enable roadmap-style predictive delta readout for draft-side and verify-side ADC/interface quantization.

#### Scenario: Verify delta readout is enabled
- **WHEN** the user enables verify delta readout and provides a verify ADC precision setting
- **THEN** the simulator quantizes verify analog linear outputs in delta domain relative to the previous reconstructed token output

#### Scenario: Draft delta readout is enabled
- **WHEN** the user enables draft delta readout and provides a draft ADC precision setting
- **THEN** the simulator quantizes draft analog linear outputs in delta domain relative to the previous reconstructed token output

#### Scenario: Verify delta readout is enabled without verify ADC precision
- **WHEN** the user enables verify delta readout while verify ADC/interface quantization is disabled
- **THEN** the simulator exits with a clear error explaining that verify ADC bits are required

#### Scenario: Draft delta readout is enabled without draft ADC precision
- **WHEN** the user enables draft delta readout while draft ADC/interface quantization is disabled
- **THEN** the simulator exits with a clear error explaining that draft ADC bits are required

### Requirement: Fall back to absolute readout for the first token in a stream
The simulator SHALL preserve an absolute-readout fallback whenever no previous reconstructed token output exists for a draft or verify stream.

#### Scenario: First token has no delta baseline
- **WHEN** a supported draft-side or verify-side analog linear output is evaluated without a stored previous reconstructed output
- **THEN** the simulator applies the existing absolute ADC quantization path for that token
- **AND** the simulator stores the reconstructed result as the baseline for the next token

### Requirement: Reconstruct current outputs from quantized deltas
When draft or verify delta readout is enabled and a previous reconstructed output exists, the simulator SHALL reconstruct the current output from the previous reconstructed output plus the quantized delta.

#### Scenario: Later token uses previous reconstructed output
- **WHEN** draft or verify delta readout is enabled and a previous reconstructed output exists for a supported linear output
- **THEN** the simulator computes the current delta relative to that previous reconstructed output
- **AND** the simulator reconstructs the current output as `prev_reconstructed + quantized_delta`
- **AND** the simulator stores that reconstructed current output as the baseline for the next token

### Requirement: Optionally quantize the delta-readout DAC feedback baseline
When draft or verify delta readout is enabled, the simulator SHALL allow the stored previous reconstructed output to be quantized before the subtraction step to model finite DAC feedback precision.

#### Scenario: Verify DAC feedback precision is configured
- **WHEN** verify delta readout is enabled and `verify_delta_dac_bits > 0`
- **THEN** the simulator quantizes the stored previous reconstructed output before forming the analog-domain delta signal
- **AND** the simulator still reconstructs the current output by adding the ADC-quantized delta to the stored previous reconstructed output

#### Scenario: Draft DAC feedback precision is configured
- **WHEN** draft delta readout is enabled and `draft_delta_dac_bits > 0`
- **THEN** the simulator quantizes the stored previous reconstructed output before forming the analog-domain delta signal
- **AND** the simulator still reconstructs the current output by adding the ADC-quantized delta to the stored previous reconstructed output

#### Scenario: DAC feedback precision is left ideal
- **WHEN** verify delta readout is enabled and `verify_delta_dac_bits = 0`
- **AND** draft delta readout is disabled or `draft_delta_dac_bits = 0`
- **THEN** the simulator preserves the previous ideal-feedback delta-readout behavior

### Requirement: Preserve token-order continuity within one sequence
Within one prompt/generation stream, predictive delta readout SHALL follow token order rather than resetting between internal forward calls.

#### Scenario: Prompt prefill is followed by decode
- **WHEN** a prompt prefill is followed by autoregressive decode in the same generation stream
- **THEN** the first generated token uses the last reconstructed prompt-token output as its delta baseline

#### Scenario: One forward pass contains multiple token positions
- **WHEN** a single forward pass processes multiple token positions
- **THEN** delta quantization is applied in token order so token `t` uses token `t-1`'s reconstructed output as its baseline

### Requirement: Reset delta baselines at sequence boundaries
The simulator SHALL clear stored delta-readout baselines before processing a new independent sequence.

#### Scenario: A new sequence begins
- **WHEN** a new top-level generation run, dataset prompt, or other independent sequence begins
- **THEN** the simulator clears all stored delta-readout baselines before processing the first token of that sequence
