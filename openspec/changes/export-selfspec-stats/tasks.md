## 1. Stats Artifact Utilities

- [x] 1.1 Add a helper to convert `accept_counts` (len=K+1) into a `stats.json` payload (`k`, `histogram`) including all bins `0..K`
- [x] 1.2 Add a lightweight validation check for exported stats (`k>=0`, bins in-range, `sum(histogram)>0`) and fail with a clear error on invalid input
- [x] 1.3 Define and implement a `stats_meta.json` payload builder (run identity, model IDs/paths, generation params, noise/precision knobs, RNG seeds, aggregation summary)

## 2. `generate.py` Single-Run Export

- [x] 2.1 Add CLI flags `--stats_out <path>` and `--no_stats_meta` to `generate.py` (help text + defaults)
- [x] 2.2 Implement `--stats_out` path semantics: if `<path>` is a directory write `<path>/stats.json`, else write exactly `<path>`; create parent dirs as needed
- [x] 2.3 Export aggregated acceptance histogram for speculative runs (`draft_model != None`) using `speculate_k` as `k` and summed counts across `--num_samples`
- [x] 2.4 Reject `--stats_out` when speculative decoding is not enabled (no draft model) with an actionable error
- [x] 2.5 Ensure compile warmup (`i == -1`) does not affect exported/printed acceptance histograms (exclude warmup from aggregation)
- [x] 2.6 Write `stats_meta.json` next to `stats.json` by default and ensure it records the knobs actually used (noise/quant/attention backend) and RNG seed behavior

## 3. Dataset Runner + Aggregation

- [x] 3.1 Create a dataset runner script (e.g., `scripts/dataset_selfspec_stats.py`) that iterates over many prompts and aggregates acceptance histograms
- [x] 3.2 Implement prompt loading for at least: `.txt` (one prompt per line) and `.jsonl` (field `prompt`)
- [x] 3.3 Add dataset controls: `--limit`, deterministic `--seed`, and error out if zero prompts are processed
- [x] 3.4 Write outputs to `out/<run_id>/stats.json` + `out/<run_id>/stats_meta.json` and record dataset context + prompt construction rules in metadata
- [x] 3.5 Add optional prompt-length sweep mode that emits `stats_Lprompt_<N>.json` + `stats_meta_Lprompt_<N>.json` for each configured `N`

## 4. Validation + Docs

- [x] 4.1 Add a small smoke-test or unit-style script that validates exported `stats.json` shape and basic invariants (bins present, sum>0)
- [x] 4.2 Add a compatibility check snippet in docs showing `ppa-calculator --stats <exported stats.json>` works (without adding a runtime dependency on `selfspec-calculator`)
- [x] 4.3 Update `README.md` (or a short doc) with the new flags and the expected output layout under `out/<run_id>/`
