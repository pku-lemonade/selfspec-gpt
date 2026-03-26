## ADDED Requirements

### Requirement: Configure verify-side predictive delta readout
The simulator SHALL allow users to enable roadmap-style predictive delta readout for verify-side ADC/interface quantization.

#### Scenario: Verify delta readout is enabled
- **WHEN** the user enables verify delta readout and provides a verify ADC precision setting
- **THEN** the simulator quantizes verify analog linear outputs in delta domain relative to the previous reconstructed token output

#### Scenario: Verify delta readout is enabled without verify ADC precision
- **WHEN** the user enables verify delta readout while verify ADC/interface quantization is disabled
- **THEN** the simulator exits with a clear error explaining that verify ADC bits are required

### Requirement: Fall back to absolute readout for the first token in a stream
The simulator SHALL preserve an absolute-readout fallback whenever no previous reconstructed token output exists for a verify stream.

#### Scenario: First token has no delta baseline
- **WHEN** a supported verify-side analog linear output is evaluated without a stored previous reconstructed output
- **THEN** the simulator applies the existing absolute verify ADC quantization path for that token
- **AND** the simulator stores the reconstructed result as the baseline for the next token

### Requirement: Reconstruct current outputs from quantized deltas
When verify delta readout is enabled and a previous reconstructed output exists, the simulator SHALL reconstruct the current output from the previous reconstructed output plus the quantized delta.

#### Scenario: Later token uses previous reconstructed output
- **WHEN** verify delta readout is enabled and a previous reconstructed output exists for a supported linear output
- **THEN** the simulator computes the current delta relative to that previous reconstructed output
- **AND** the simulator reconstructs the current output as `prev_reconstructed + quantized_delta`
- **AND** the simulator stores that reconstructed current output as the baseline for the next token

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
