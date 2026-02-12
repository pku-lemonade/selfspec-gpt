## Context

`gpt-fast` already computes speculative decoding acceptance statistics during generation:

- `generate.generate(...)` returns `accept_counts: List[int]` of length `K+1`.
- Index `a ∈ [0..K]` corresponds to the number of accepted draft tokens in a “burst” (`a == K` means all `K` draft tokens were accepted).
- The CLI (`generate.py`) currently aggregates these counts across `--num_samples` and prints derived metrics (e.g., acceptance probabilities / mean accepted) to stdout.

Downstream tooling (notably `selfspec-calculator` / `ppa-calculator`) expects a machine-readable acceptance histogram in a canonical `stats.json` format (see `../selfspec-calculator/examples/stats.json`). Today that requires manual parsing/translation.

This change adds first-class JSON export and dataset aggregation so benchmark runs can directly feed the calculators.

## Goals / Non-Goals

**Goals:**
- Export a calculator-compatible `stats.json` from speculative runs, matching `selfspec-calculator`’s `SpeculationStats` schema (`k`, `histogram`).
- Optionally write a `stats_meta.json` sidecar with enough metadata to reproduce the run (model IDs, noise/policy knobs, seeds, sampling params, prompt construction).
- Provide a dataset runner that aggregates `accept_counts` across many prompts, with an option to emit one `stats*.json` per prompt-length bucket.
- Standardize an output layout (e.g., `out/<run_id>/stats.json` + `stats_meta.json`, and `stats_Lprompt_<N>.json` for sweeps).

**Non-Goals:**
- Change speculative decoding behavior or the definition of “accepted-prefix length”.
- Make `gpt-fast` depend on `selfspec-calculator` at runtime.
- Build a fully general dataset ingestion framework; initial support should cover the most common local prompt sources used for benchmarking.

## Decisions

### 1) `stats.json` stays minimal and matches calculator expectations

**Decision:** `stats.json` will contain only:

- `k` (int): the speculative depth `K` used for the run.
- `histogram` (object): mapping `a -> value` for `a ∈ [0..K]`, where `a` is accepted-prefix length per burst.

**Value semantics:** emit **counts** (integers) by default. The calculator normalizes internally, so counts or probabilities are both acceptable; counts preserve exact sampling weight across mixed-length runs.

**JSON shape details:**
- Emit all bins `0..K` (including zero-count bins) for stability and easy validation.
- Keys are emitted as strings (JSON object keys), but chosen to be parseable as ints (`"0".."K"`), matching the calculator’s coercion behavior.

**Rationale:** Keep the ingestion artifact small/stable and aligned with the existing `selfspec-calculator` schema to avoid format drift.

### 2) Reproducibility metadata is a sidecar: `stats_meta.json`

**Decision:** Write a separate `stats_meta.json` alongside `stats.json` (enabled by default when exporting stats) to avoid adding non-schema fields to `stats.json`.

Recommended fields (v1; exact keys to be finalized in specs):
- **Run identity:** `run_id`, timestamp, `git_commit` (and dirty flag if available).
- **Model identifiers:** target/draft checkpoint paths and/or resolved IDs (optionally hashes), tokenizer path.
- **Generation params:** `K`, `max_new_tokens`, `temperature`, `top_k`, `batch_size`, chat formatting flags.
- **Seeds:** sampling seed(s), draft-noise seed, and any per-prompt seed derivation rules.
- **Noise / policy knobs actually used:** draft noise settings (std or level-based), read-noise std, quantization knobs, attention backend.
- **Dataset context (if applicable):** dataset name/split, prompt construction, `L_prompt` (fixed) or bucket definition, limits/shuffling.
- **Aggregation summary:** number of prompts, number of bursts, total histogram sum, and (optionally) derived `mean_accepted`.

**Rationale:** Keep `stats.json` fully compatible with calculators while still capturing everything needed to reproduce a histogram.

### 3) Single-run export integrates into the existing `generate.py` CLI

**Decision:** Add CLI flags to `generate.py` to write stats artifacts after generation completes, reusing the already-collected `accept_counts`.

Proposed interface (exact names to be finalized in specs):
- `--out_dir out/<run_id>` (or `--run_id` + default `out/<run_id>`): destination directory.
- `--write_stats` (or `--stats_out <path>`): enable writing `stats.json`.
- `--no_stats_meta`: optional escape hatch to skip the sidecar.

Implementation notes:
- Export is only valid for speculative runs (`draft_model != None`). Target-only runs currently produce an all-zero `accept_counts`, so export should error clearly rather than producing an invalid histogram.
- When `--num_samples > 1`, aggregate counts across samples (the current stdout aggregation path already computes `counts_aggregated`).

### 4) Dataset aggregation is a separate runner script with optional prompt-length sweeps

**Decision:** Implement dataset aggregation as a new script (e.g., `scripts/dataset_selfspec_stats.py`) rather than overloading the interactive/single-prompt CLI.

Baseline behavior:
- Load prompts from a simple local source (e.g., `.jsonl` with a `prompt` field, or `.txt` one prompt per line).
- For each prompt, run speculative generation and collect `accept_counts`.
- Sum `accept_counts` across prompts (and optionally across multiple samples per prompt) into one histogram.
- Write `out/<run_id>/stats.json` + `stats_meta.json`.

Prompt-length bucket mode:
- Accept `--prompt_lengths 64 128 ...` (or a bucket definition) and produce one file per length:
  - `stats_Lprompt_64.json` (+ matching `stats_meta_Lprompt_64.json`), etc.
- Construction rules (truncate/pad/skip) and tokenizer handling are spec-defined; the implementation should make the behavior explicit in `stats_meta.json`.

**Rationale:** Keeps the common “single run” path lightweight while enabling scalable dataset sweeps and standardized output naming.

## Risks / Trade-offs

- **Format drift vs calculators:** If the calculator schema evolves, exports could break. Mitigation: keep `stats.json` strictly to the current `SpeculationStats` shape and add a lightweight validation check in the exporter.
- **Dataset ingestion dependencies:** Reading parquet/HF datasets may require new dependencies. Mitigation: start with JSONL/TXT prompts; add richer dataset loaders only if justified.
- **Reproducibility limits:** GPU nondeterminism and changing kernels can affect acceptance. Mitigation: record seeds, code version, and key runtime knobs in `stats_meta.json`.
- **Histogram bias from short generations:** Very small `max_new_tokens` can overweight edge bursts. Mitigation: document recommended generation lengths and record `max_new_tokens` / burst count in metadata.

## Migration Plan

1. Add `stats.json` (+ `stats_meta.json`) export to the existing `generate.py` CLI for speculative runs.
2. Add the dataset aggregation runner and standard output layout under `out/<run_id>/`.
3. Add a short “compatibility check” snippet / documentation showing that `ppa-calculator --stats <stats.json>` accepts the exported file.

## Open Questions

- Should the exporter support emitting normalized probabilities (in addition to counts) as an explicit option, or keep counts-only for v1?
- What is the minimum required metadata set for `stats_meta.json` to make histograms meaningfully reproducible across machines?
- For prompt-length sweeps, do we want strict “exact token length” buckets (skip mismatched prompts) or “truncate/pad to L” (include all prompts)?
