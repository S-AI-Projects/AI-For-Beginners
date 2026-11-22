# Applied Lab R38-T013: Training-Only Imputation (Data Leakage Prevention)

## Overview

In Transfer Learning and Computer Vision pipelines (see [Transfer Learning](../../README.md)), feature extractors and preprocessors must maintain strict separation between training and evaluation splits. Computing statistics (mean, std, min, max) on validation or test data causes **data leakage**, resulting in overly optimistic evaluation.

## Contract & Specification

- **Training data:** `[2, None, 4]` $\to$ fitted mean = `3.0`.
- **Validation data:** `[100, None]` $\to$ transformed to `[100.0, 3.0]` (using the stored training mean).
- **Leakage check:** Result is **not** `[100.0, 100.0]` (which would occur if validation statistics were recalculated).
- **All-missing training input:** `[None, None]` is rejected with `ValueError`.

## Comparison of Two Approaches

1. **Object-Oriented (`TrainingMeanImputer`):** Encapsulates state (`fitted_mean`), disallows transform prior to fitting.
2. **Functional (`fit_mean` + `transform_with_mean`):** Explicit parameter passing, enabling pure functional pipeline pipelines.

Both approaches are verified for strict numerical equivalence across identical fixtures.

## Running Tests

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/08-TransferLearning/applied-labs/r38-t013 -p "test_*.py" -v
```
