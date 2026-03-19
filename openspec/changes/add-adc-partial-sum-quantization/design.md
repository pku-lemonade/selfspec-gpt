## Context

The roadmap describes the simulator in hardware terms: draft mode should use a lower-resolution ADC path and verify mode can use a higher-resolution ADC path. The current codebase already has a fake-quantization mechanism for linear outputs (`post_matmul_quant_bits`), but the knobs are generic and do not make it clear that the behavior is intended to represent analog interface / partial-sum quantization rather than weight quantization.

This change is cross-cutting because it affects:

- simulation semantics in `generate.py`
- dataset evaluation tooling
- K-sweep tooling
- exported metadata used for later analysis

## Goals / Non-Goals

**Goals:**
- Expose explicit draft and verify ADC interface quantization knobs.
- Define those knobs as post-accumulation linear-output quantization for analog blocks.
- Preserve compatibility with the existing post-matmul flags where possible.
- Make the exported metadata reflect ADC interface precision explicitly.

**Non-Goals:**
- Implement the full residual-array reuse model from the roadmap.
- Simulate separate Array1 / Arrays2-4 paths or bonus-token reuse logic.
- Quantize non-linear digital operators such as softmax or elementwise activations.
- Add hardware estimator support in this change.

## Decisions

### Decision: Reuse the existing linear-output fake quantization path

We will map the new ADC-style interface knobs onto the existing `output_quant_bits` path in `quantize.py`.

Rationale:
- it already quantizes after linear accumulation, which matches the intended approximation
- it avoids duplicating quantization logic
- it keeps the implementation small and testable

Alternative considered:
- creating a second independent quantization implementation
  - rejected because it would duplicate behavior and increase inconsistency risk

### Decision: Introduce explicit draft and verify ADC knob names

We will add CLI flags such as:

- `--verify_adc_bits`
- `--draft_adc_bits`

These will be the preferred roadmap-aligned names.

Rationale:
- they encode hardware meaning directly
- they reduce ambiguity versus generic “post matmul” language

Alternative considered:
- keep only legacy `post_matmul_quant_bits`
  - rejected because it does not communicate hardware meaning clearly

### Decision: Keep legacy post-matmul flags as compatibility aliases

The simulator already exposes:

- `--post_matmul_quant_bits`
- `--draft_post_matmul_quant_bits`

These will remain available, but the code will define how they interact with the new ADC-style names.

Rationale:
- avoids breaking existing scripts
- lets old experiments continue to replay

### Decision: Restrict ADC quantization to analog linear outputs

ADC-style quantization will apply only to linear outputs that correspond to analog matmuls.

Rationale:
- this is the closest match to the roadmap’s “partial-sum / interface precision” concept
- quantizing softmax or other digital operators would model a different hardware limitation

## Risks / Trade-offs

- **Approximation risk** → This still models ADC precision with fake quantization, not a full analog residual-array simulator. Mitigation: document it clearly as an interface-quantization approximation.
- **User confusion from duplicate knobs** → Legacy and new flags may overlap. Mitigation: define precedence and validate conflicting usage.
- **Metadata inconsistency** → Old result files may only contain legacy quantization keys. Mitigation: write both legacy and explicit ADC fields where useful during the transition.

## Migration Plan

1. Add explicit ADC interface flags to `generate.py`.
2. Map those flags to the existing post-matmul quantization implementation.
3. Thread the new flags through dataset stats and K-sweep scripts.
4. Preserve legacy flags and document the precedence / alias behavior.
5. Validate with small smoke runs and one dataset evaluation.

## Open Questions

- Should `verify_adc_bits = 0` mean “disabled / ideal verify path”, or should verify always be assigned an explicit high bit value for roadmap reporting?
- Should the metadata export include both the raw ADC flag values and the resolved quantization bits after alias precedence is applied?
