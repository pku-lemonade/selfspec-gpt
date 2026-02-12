## Why

Today, `gpt-fast` prints speculative decoding acceptance statistics (e.g., accept-count histograms) to stdout, which requires manual parsing/translation before they can be consumed by downstream tooling like `selfspec-calculator` / `ppa-calculator`. This change standardizes acceptance-stat export and dataset aggregation so runs can directly feed calculators with reproducible, machine-readable artifacts.

## What Changes

- Add a CLI output mode to write a calculator-compatible `stats.json` artifact capturing the accepted-prefix-length histogram for a run.
- Add an optional `stats_meta.json` sidecar containing run/dataset/model metadata needed to reproduce the histogram (policy/noise/seed/model identifiers, prompt construction, sampling params, etc.).
- Add a dataset runner that aggregates acceptance statistics across many prompts, with an option to emit per-prompt-length buckets (e.g., `stats_Lprompt_64.json`, `stats_Lprompt_128.json`, ...).
- Document the canonical artifact format and where outputs are written so dataset runs can be used directly by calculators without ad-hoc parsing.

## Capabilities

### New Capabilities

- `selfspec-stats-export`: Export a canonical acceptance histogram (`stats.json`) and optional reproducibility sidecar (`stats_meta.json`) from speculative decoding runs.
- `selfspec-dataset-stats`: Run speculative decoding over a dataset and aggregate acceptance histograms (optionally emitting per prompt-length bucket files).

### Modified Capabilities

<!-- None. -->

## Impact

- **Code**: Likely touches `generate.py` (CLI and/or metrics export), plus new script/module for dataset runs and aggregation.
- **Outputs**: Introduces new JSON artifacts written to a stable `out/<run_id>/...` layout.
- **Tooling**: Aligns output format with `selfspec-calculator` / `ppa-calculator`, reducing manual glue code between repos.
