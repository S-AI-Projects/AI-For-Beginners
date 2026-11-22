# Applied Lab R38-T020: Tensor-Shape Contract — Scenario Assessment

## Overview

In Computer Vision image preprocessing pipelines (see [Intro to CV](../../README.md)), raw image batches ($N \times H \times W$) are validated and scaled from raw $[0, 255]$ intensities to $[0.0, 1.0]$ floats while strictly preserving batch dimensions.

## Contract & Specification

- **Input Batch:** $N \times H \times W$ numeric pixels in $[0, 255]$.
- **Contract Fixture:** `[[[0, 255], [128, 64]]]` (shape `(1, 2, 2)`).
- **Normalized Output:** Preserves shape `(1, 2, 2)`; maps `0` $\to$ `0.0` and `255` $\to$ `1.0`.
- **Validation:** Values $< 0$ or $> 255$ raise `ValueError`.

## Applied Assessment Scenarios

1. **Q1 (Predict an Output):** Predict exact normalized tensor and shape for `[[[0, 255], [128, 64]]]`.
   - *Rationale:* Linear division by 255.0 maps `0` $\to$ `0.0`, `255` $\to$ `1.0`, `128` $\to$ `128/255`, and `64` $\to$ `64/255`, preserving `(1, 2, 2)`.
2. **Q2 (Diagnose a Failure):** Identify why `[[[-5, 255], [128, 64]]]` fails.
   - *Rationale:* Negative pixel intensities indicate corrupt input or uncalibrated sensors; fails fast with `ValueError`.
3. **Q3 (Choose a Boundary Policy):** Justify strict boundary rejection over silent clamping.
   - *Rationale:* Silent truncation masks upstream bugs, risking model degradation in production.

## Simulation Scope & Limitations

> [!NOTE]
> This assessment models preprocessing validation logic in pure Python without external dependencies, hardware acceleration, or model weights.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/06-IntroCV/applied-labs/r38-t020 -p "test_*.py" -v
```
