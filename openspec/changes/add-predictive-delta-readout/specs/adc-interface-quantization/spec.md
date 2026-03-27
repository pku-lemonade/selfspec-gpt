## MODIFIED Requirements

### Requirement: Configure draft and verify ADC interface precision
The simulator SHALL allow users to configure separate ADC-style interface quantization settings for draft-mode and verify-mode analog linear outputs, and SHALL allow the verify path to use either absolute or predictive-delta readout semantics.

#### Scenario: Verify delta readout is configured
- **WHEN** the user enables verify delta readout
- **THEN** the configured verify ADC precision is interpreted as the physical bitwidth used to quantize the verify-path delta signal rather than the absolute verify output

#### Scenario: Verify delta readout is disabled
- **WHEN** the user does not enable verify delta readout
- **THEN** the simulator preserves the existing absolute-output verify ADC quantization behavior

### Requirement: Configure verify delta-readout DAC feedback precision
The simulator SHALL allow users to configure optional DAC-side precision for the stored verify delta-readout feedback baseline.

#### Scenario: Verify delta-readout DAC precision is configured
- **WHEN** the user enables verify delta readout and provides a positive verify delta-readout DAC precision
- **THEN** the simulator interprets that precision as the physical bitwidth used to quantize the stored verify feedback baseline before subtraction

#### Scenario: Verify delta-readout DAC precision is provided without delta readout
- **WHEN** the user provides verify delta-readout DAC precision while verify delta readout is disabled
- **THEN** the simulator exits with a clear error explaining that verify delta readout must be enabled

### Requirement: Export ADC interface precision in metadata
The simulator SHALL record ADC/DAC interface precision and readout mode in run metadata and sweep outputs.

#### Scenario: Dataset acceptance stats are exported for a delta-readout run
- **WHEN** dataset or single-run stats metadata is written for a run with verify delta readout enabled
- **THEN** the metadata includes the configured verify ADC precision, the configured verify delta-readout DAC precision, and an explicit indication that the verify readout mode was predictive-delta

#### Scenario: K-sweep results are exported for a delta-readout run
- **WHEN** the K sweep script writes its JSON summary for a run with verify delta readout enabled
- **THEN** the summary includes the configured verify ADC precision, the configured verify delta-readout DAC precision, and the verify readout mode
