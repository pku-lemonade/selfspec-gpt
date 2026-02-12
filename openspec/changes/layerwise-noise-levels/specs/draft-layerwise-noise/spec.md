## ADDED Requirements

### Requirement: Define noise level standard deviations
The system SHALL allow users to define a mapping from discrete draft-noise levels to Gaussian noise standard deviations.

#### Scenario: Level-to-std mapping is provided
- **WHEN** the user provides `--draft_noise_level_stds` as a list of floats
- **THEN** the system uses list indices as the noise levels (index 0 is level 0, index 1 is level 1, etc.)

### Requirement: Configure draft noise levels for FFN/QKV/OUT
The system SHALL allow users to assign a noise level to each draft-noise bucket (**FFN**, **QKV**, **OUT**) and resolve each bucket’s Gaussian std via the shared level-to-std mapping.

#### Scenario: Broadcast a single level to all layers and buckets
- **WHEN** the user provides `--draft_noise_levels <level>` as a single integer
- **THEN** the system applies that level to FFN, QKV, and OUT in every transformer layer

#### Scenario: Broadcast per-bucket levels to all layers
- **WHEN** the user provides `--draft_noise_levels <ffn_level> <qkv_level> <out_level>`
- **THEN** the system applies those three levels to (FFN, QKV, OUT) respectively in every transformer layer

#### Scenario: Provide per-layer per-bucket levels
- **WHEN** the user provides `--draft_noise_levels` with exactly `3 * n_layer` integers
- **THEN** the system interprets the values as per-layer triplets in (FFN, QKV, OUT) order for layers `0..n_layer-1`

### Requirement: Apply draft weight noise per layer and bucket
When draft weight noise is enabled, the system SHALL add i.i.d. Gaussian noise to supported draft-model weight matrices after draft model load, using the resolved per-layer bucket stds.

#### Scenario: Noise is applied to the supported draft weights
- **WHEN** draft noise is enabled and the resolved std for a given bucket is greater than 0
- **THEN** the system perturbs the draft model’s corresponding weights for each layer:
  - QKV: `layers.<i>.attention.wqkv.weight`
  - OUT: `layers.<i>.attention.wo.weight`
  - FFN: `layers.<i>.feed_forward.w1.weight`, `layers.<i>.feed_forward.w2.weight`, `layers.<i>.feed_forward.w3.weight`

### Requirement: Validate noise configuration inputs
The system SHALL reject invalid draft noise configuration inputs with a clear error.

#### Scenario: Levels require a level-to-std mapping
- **WHEN** the user provides `--draft_noise_levels` without `--draft_noise_level_stds`
- **THEN** the system exits with an error indicating that level stds are required to resolve levels

#### Scenario: Level indices are out of range
- **WHEN** any configured noise level is less than 0 or greater than or equal to `len(draft_noise_level_stds)`
- **THEN** the system exits with an error indicating the invalid level and the valid range

#### Scenario: Invalid assignment shape is provided
- **WHEN** `--draft_noise_levels` is provided with a length that is not 1, 3, or `3 * n_layer`
- **THEN** the system exits with an error indicating the expected lengths

### Requirement: Preserve backwards-compatible std-based configuration
The system SHALL continue to support the existing std-based draft noise configuration for compatibility.

#### Scenario: Existing `--draft_noise_std` is used
- **WHEN** the user provides `--draft_noise_std` and does not provide any level-based noise flags
- **THEN** the system applies draft weight noise using the existing behavior (1 value broadcast to all, or 3 values interpreted as FFN/QKV/OUT)

#### Scenario: Level-based flags take precedence
- **WHEN** the user provides any level-based noise flags alongside `--draft_noise_std`
- **THEN** the system uses the level-based configuration and ignores `--draft_noise_std`
