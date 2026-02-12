## ADDED Requirements

### Requirement: Aggregate acceptance statistics across a prompt dataset
The system SHALL provide a dataset runner that executes speculative decoding over many prompts and aggregates acceptance statistics into a single calculator-compatible histogram.

#### Scenario: Aggregate stats over a set of prompts
- **WHEN** the user runs the dataset runner over a non-empty set of prompts
- **THEN** the system produces an aggregated `stats.json` whose histogram is the elementwise sum of acceptance counts across all processed prompts (and samples, if configured)

#### Scenario: Reject empty datasets
- **WHEN** the dataset runner is invoked but processes zero prompts
- **THEN** the system exits with an error indicating that no prompts were processed

### Requirement: Dataset runner writes standardized output artifacts
The dataset runner SHALL write calculator-ingestable stats artifacts using a stable directory layout.

#### Scenario: Default output layout is used
- **WHEN** the user provides an output directory `out/<run_id>/`
- **THEN** the dataset runner writes:
  - `out/<run_id>/stats.json`
  - `out/<run_id>/stats_meta.json`

#### Scenario: Metadata records dataset context
- **WHEN** `stats_meta.json` is written for a dataset run
- **THEN** it includes at least:
  - dataset identity (path and/or dataset name + split)
  - prompt construction rules (including whether prompts are truncated, skipped, or otherwise normalized)
  - number of prompts processed and any selection/shuffle configuration
  - model identifiers, generation parameters, and RNG seed(s)

### Requirement: Support prompt-length sweeps
The dataset runner SHALL support emitting one stats artifact per configured prompt length.

#### Scenario: Emit one file per prompt length
- **WHEN** the user configures a prompt-length sweep with lengths `L1..Ln`
- **THEN** the dataset runner writes one stats file per length:
  - `out/<run_id>/stats_Lprompt_<Li>.json` for each `Li`
- **AND THEN** it writes a matching metadata sidecar per length:
  - `out/<run_id>/stats_meta_Lprompt_<Li>.json` for each `Li`

#### Scenario: Sweep metadata records the effective prompt length
- **WHEN** `stats_meta_Lprompt_<Li>.json` is written
- **THEN** it records the effective tokenized prompt length used for that file’s histogram as `Li`
