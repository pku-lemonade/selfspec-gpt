## MODIFIED Requirements

### Requirement: Configure draft and verify ADC interface precision
The simulator SHALL allow users to configure separate ADC-style interface quantization settings for draft-mode and verify-mode analog linear outputs, and SHALL allow the verify path to use either absolute or predictive-delta readout semantics.

#### Scenario: Verify delta readout is configured
- **WHEN** the user enables verify delta readout
- **THEN** the configured verify ADC precision is interpreted as the physical bitwidth used to quantize the verify-path delta signal rather than the absolute verify output

#### Scenario: Verify delta readout is disabled
- **WHEN** the user does not enable verify delta readout
- **THEN** the simulator preserves the existing absolute-output verify ADC quantization behavior

### Requirement: Export ADC interface precision in metadata
The simulator SHALL record both ADC interface precision and readout mode in run metadata and sweep outputs.

#### Scenario: Dataset acceptance stats are exported for a delta-readout run
- **WHEN** dataset or single-run stats metadata is written for a run with verify delta readout enabled
- **THEN** the metadata includes the configured verify ADC precision and an explicit indication that the verify readout mode was predictive-delta

#### Scenario: K-sweep results are exported for a delta-readout run
- **WHEN** the K sweep script writes its JSON summary for a run with verify delta readout enabled
- **THEN** the summary includes the configured verify ADC precision and the verify readout mode
