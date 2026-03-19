## MODIFIED Requirements

### Requirement: Preserve backwards-compatible std-based configuration
The system SHALL continue to support the existing std-based draft noise configuration for compatibility and SHALL allow that draft noise to compose with ADC-style interface quantization when both are enabled.

#### Scenario: Existing `--draft_noise_std` is used
- **WHEN** the user provides `--draft_noise_std` and does not provide any level-based noise flags
- **THEN** the system applies draft weight noise using the existing behavior (1 value broadcast to all, or 3 values interpreted as FFN/QKV/OUT)

#### Scenario: Level-based flags take precedence
- **WHEN** the user provides any level-based noise flags alongside `--draft_noise_std`
- **THEN** the system uses the level-based configuration and ignores `--draft_noise_std`

#### Scenario: Draft noise composes with ADC interface quantization
- **WHEN** draft noise and draft ADC interface quantization are both enabled
- **THEN** the simulator applies draft weight noise to the draft model and then applies ADC-style output quantization on the affected analog linear outputs during draft inference
