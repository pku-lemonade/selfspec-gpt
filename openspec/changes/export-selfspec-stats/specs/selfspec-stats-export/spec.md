## ADDED Requirements

### Requirement: Export calculator-compatible speculation statistics
The system SHALL support exporting speculative decoding acceptance statistics to a `stats.json` file that is compatible with `selfspec-calculator` / `ppa-calculator`.

#### Scenario: Stats export is requested for a speculative run
- **WHEN** the user runs `generate.py` with a draft model enabled and provides `--stats_out <path>`
- **THEN** the system writes a JSON file to `<path>` (or `<path>/stats.json` if `<path>` is a directory)

#### Scenario: Output directories are created
- **WHEN** the user provides `--stats_out <path>` and the parent directory does not exist
- **THEN** the system creates the directory before writing `stats.json`

### Requirement: Define `stats.json` schema and semantics
The exported `stats.json` SHALL include a speculative depth `k` and an acceptance histogram `histogram`.

#### Scenario: `stats.json` matches the calculator schema
- **WHEN** `stats.json` is written
- **THEN** the file contains:
  - `k`: an integer speculative depth `K` (where `K >= 0`)
  - `histogram`: a mapping from accepted-prefix length `a` to a non-negative numeric value
- **AND THEN** `histogram` includes bins for every `a ∈ [0..K]`
- **AND THEN** `sum(histogram.values()) > 0`

#### Scenario: Histogram bins reflect accepted-prefix length per burst
- **WHEN** the system records one speculative “burst” (a single speculative step proposing up to `K` draft tokens)
- **THEN** it increments exactly one histogram bin `a`, where `a` is the number of accepted draft tokens in that burst

#### Scenario: Histogram aggregation spans multiple samples
- **WHEN** `generate.py` is run with `--num_samples N` and `--stats_out <path>`
- **THEN** each histogram bin value equals the sum of per-sample acceptance counts for that bin across all `N` samples

### Requirement: Export reproducibility metadata in a sidecar file
The system SHALL write a `stats_meta.json` sidecar next to `stats.json` unless explicitly disabled.

#### Scenario: Sidecar is written by default
- **WHEN** the user provides `--stats_out <path>` and does not provide `--no_stats_meta`
- **THEN** the system writes `stats_meta.json` in the same directory as `stats.json`

#### Scenario: Sidecar contains core run metadata
- **WHEN** `stats_meta.json` is written
- **THEN** it includes at least:
  - target and draft model identifiers (e.g., checkpoint paths and/or resolved IDs)
  - generation parameters (`k`, `max_new_tokens`, `temperature`, `top_k`, `batch_size`, `num_samples`)
  - noise / precision knobs actually used (draft noise settings, read-noise, quantization knobs, attention backend)
  - RNG seed(s) used for sampling and acceptance decisions

### Requirement: Reject unsupported export cases
The system SHALL reject requests to export speculation statistics when acceptance counts are not available.

#### Scenario: Export is requested without a draft model
- **WHEN** the user provides `--stats_out <path>` but does not enable speculative decoding (no draft model)
- **THEN** the system exits with an error indicating that stats export requires a draft model
