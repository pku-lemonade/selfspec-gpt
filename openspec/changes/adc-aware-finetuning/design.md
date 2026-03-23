## Context

The repository now exposes explicit ADC-style interface quantization for simulator runs (`draft_adc_bits`, `verify_adc_bits`). However, the finetuning loop still assumes only write/read noise and cannot optimize the noisy draft under the same ADC-degraded conditions that the roadmap evaluation uses. Since `Qwen3-1.7B` is already the strongest model family under the roadmap objective, it is the best candidate for a first ADC-aware finetuning pilot.

## Goals / Non-Goals

**Goals:**
- Add student/teacher ADC quantization knobs to the finetuning script.
- Keep the teacher clean/frozen and allow independent verify-style ADC precision.
- Run a 1.7B self-target refinement pilot under write noise plus draft ADC quantization.

**Non-Goals:**
- Rework the full residual-array hardware reuse model.
- Add new hardware estimator outputs.
- Revisit the old 0.6B/1B checkpoints in this change.

## Decisions

### Decision: Reuse the existing ADC-interface quantization implementation in training

Finetuning will reuse the same post-accumulation output quantization path already used by inference/evaluation.

Rationale:
- keeps simulator semantics aligned across training and evaluation
- minimizes new code paths

### Decision: Use self-target refinement on the current 1.7B winner

The pilot will start from:

- `model_wikitext_noise_ft_rel10_write_cons_step400.pth`

and use the same checkpoint as the frozen clean teacher.

Rationale:
- the current 1.7B checkpoint already performs well under roadmap-style acceptance
- refinement is more likely to help than training from the clean base again

## Risks / Trade-offs

- **Training instability under ADC degradation** → use low LR and checkpoint early
- **No measurable gain despite extra degradation modeling** → compare directly against the current roadmap winner before running larger benchmarks
- **Teacher/student quantization mismatch ambiguity** → keep teacher ADC explicit and default it separately from student ADC
