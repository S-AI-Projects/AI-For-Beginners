# Applied Lab R38-T014: Tensor-Shape Contract & Invariant State Trace

## Overview

In Generative Adversarial Networks (see [GANs](../../README.md)), image preprocessing pipelines must guarantee shape invariance across batches while mapping integer pixel values $[0, 255]$ to standardized floats $[0.0, 1.0]$.

## Contract & Specification

- **Input Batch Shape:** $N \times H \times W$ numeric values in range $[0, 255]$.
- **Contract Fixture:** `[[[0, 255], [128, 64]]]` (shape: `(1, 2, 2)`).
- **Normalized Output:** `0` $\to$ `0.0`, `255` $\to$ `1.0`, preserving shape `(1, 2, 2)`.
- **Validation:** Pixels $< 0$ or $> 255$ are rejected with `ValueError`.
- **Invariant & State Trace:** Step-by-step intermediate execution states (`shape_inspection`, `pixel_validation`, `normalization`) are captured and the shape preservation invariant is explicitly checked.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/10-GANs/applied-labs/r38-t014 -p "test_*.py" -v
```
