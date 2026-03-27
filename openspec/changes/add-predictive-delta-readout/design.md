## Context

The roadmap's "Predictive Delta Readout" optimization (§3.2) assumes the ADC reads a delta signal formed against the previous token's output, not the absolute output itself. The current simulator already has verify-side ADC/interface quantization, but that path is stateless and always fake-quantizes the absolute linear output after accumulation.

That means the repository can currently study "how many bits does absolute verify readout need?" but not "how much can temporal locality reduce the required residual ADC dynamic range?"

## Goals / Non-Goals

**Goals:**
- Add roadmap-aligned predictive delta readout to the functional simulator.
- Scope the first implementation to the verify/target ADC path.
- Allow the stored verify feedback baseline to use optional DAC-side quantization.
- Make sequence-boundary behavior explicit so runs are reproducible and comparable.
- Export the verify readout mode in metadata and sweep outputs.

**Non-Goals:**
- Model hardware-estimator PPA changes from the added DAC path in this change.
- Extend delta readout to draft ADC or finetuning paths in this first pass.
- Implement the roadmap's full draft/full-precision block policy or reuse-policy matrix in this change.

## Decisions

### Decision: Model delta readout as delta-domain output quantization around the previous reconstructed output

For a supported verify-side analog linear output stream, the simulator will use:

- first token in a stream: `y_hat_0 = Q(y_0)`
- later tokens: `y_hat_t = y_hat_{t-1} + Q(y_t - y_hat_{t-1})`

where `Q(.)` is the existing ADC/interface fake-quantization path and `y_hat` is the reconstructed output retained for the next token.

Rationale:
- matches the roadmap's "feedback register + DAC subtraction" abstraction
- avoids an overly optimistic simulator that feeds back the ideal fp previous output
- lets temporal locality emerge naturally from model behavior instead of being hard-coded into an "effective bits" shortcut

### Decision: Keep the existing verify ADC bit knob and add an explicit mode toggle

The first implementation should add a verify-side boolean flag such as `--verify_delta_readout` and keep using `--verify_adc_bits` as the physical ADC bitwidth.

Rules:
- `--verify_delta_readout` requires `--verify_adc_bits > 0`
- when disabled, verify ADC behavior remains the current absolute-output path
- `--verify_adc_clip_scale` continues to apply, but in delta mode it applies to the delta signal

Rationale:
- keeps the CLI small
- preserves replayability of existing experiments
- makes the new behavior opt-in instead of silently changing verify ADC semantics

### Decision: DAC feedback precision is modeled as optional quantization of the stored delta baseline

When verify delta readout is enabled, the simulator may optionally quantize the stored previous reconstructed output before analog subtraction by using a knob such as `--verify_delta_dac_bits`.

Rules:
- `--verify_delta_dac_bits = 0` keeps the DAC feedback path ideal / unmodeled
- `--verify_delta_dac_bits > 0` requires `--verify_delta_readout`
- the DAC-limited baseline is used for the subtraction step, while digital reconstruction still adds back the stored previous reconstructed output
- `--verify_adc_clip_scale` still applies only to the ADC-quantized delta signal, not to the DAC feedback baseline

Rationale:
- captures the roadmap's finite-DAC feedback concept without introducing a full analog circuit model
- lets experiments separate "ADC delta precision" from "feedback DAC precision"
- preserves the previous default behavior when the DAC knob is not used

### Decision: Reset delta-readout state only at explicit sequence boundaries

The simulator will clear stored delta baselines:

- before a new top-level generation run
- before each new dataset prompt / independent sample
- before any other independent sequence entry point

Within one sequence:

- prompt prefill and subsequent decode share the same delta stream
- if one forward pass contains multiple token positions, delta quantization is applied in token order within that call

Rationale:
- matches the roadmap's token-to-token temporal-locality assumption
- prevents leakage across unrelated prompts or experiments

### Decision: Export both ADC precision and readout mode

Metadata and sweep outputs should record:

- the configured verify ADC precision
- the configured verify delta-readout DAC precision
- whether verify readout used absolute or predictive-delta mode

Suggested resolved field:
- `verify_adc_quant_domain`: `absolute` or `delta`

Rationale:
- a bitwidth alone is no longer enough to reproduce simulator semantics
- analysis tooling needs to distinguish "10-bit absolute" from "10-bit delta-domain"

## Risks / Trade-offs

- **Stateful quantization vs. `torch.compile`** → mutating per-module token state may be less compiler-friendly than the current stateless fake-quant path. Initial implementation should prioritize correctness and may need an eager fallback for delta-enabled runs.
- **Accumulated reconstruction error** → feeding back reconstructed outputs compounds quantization error, but that is closer to the roadmap abstraction than reusing ideal fp outputs.
- **Verify-only scope** → this first pass will not cover future draft-side full-precision blocks or training-time delta readout.
