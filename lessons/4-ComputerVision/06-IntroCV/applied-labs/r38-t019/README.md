# Applied Lab R38-T019: Tensor-Shape Contract — Evaluation Experiment

## Overview

In Computer Vision preprocessing (see [Intro to CV](../../README.md)), raw image batches ($N \times H \times W$) must be validated and converted from integer pixel intensities $[0, 255]$ into unit-normalized floats $[0.0, 1.0]$.

## Contract & Specification

- **Input Batch:** $N \times H \times W$ numeric pixels in $[0, 255]$.
- **Contract Fixture:** `[[[0, 255], [128, 64]]]` (shape `(1, 2, 2)`).
- **Normalized Output:** Preserves shape `(1, 2, 2)`; maps `0` $\to$ `0.0` and `255` $\to$ `1.0`.
- **Validation:** Values $< 0$ or $> 255$ raise `ValueError`.

## Controlled Evaluation Experiment

Three controlled batch configurations were evaluated to compare processing overhead across dimension scaling:
1. **Config A (Baseline):** Shape `(1, 2, 2)` (4 pixels)
2. **Config B (Batch Scaling):** Shape `(2, 2, 2)` (8 pixels)
3. **Config C (Spatial Scaling):** Shape `(1, 4, 4)` (16 pixels)

All configurations strictly preserved shape invariants and output ranges in $[0.0, 1.0]$.

## Simulation Scope & Limitations

> [!NOTE]
> This experiment evaluates input validation and normalization scaling. It makes no unsupported claims regarding real-world model accuracy, GPU acceleration, or disk I/O performance.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/06-IntroCV/applied-labs/r38-t019 -p "test_*.py" -v
```
