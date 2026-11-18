# Applied Lab R38-T055: Binary Classification Evaluation Adapter

## Overview
Computes confusion matrix (TP, FN, FP, TN), precision, and recall offline without scikit-learn or external library dependencies.

## Contract
- **Input**: `y_true: Sequence[int]`, `y_pred: Sequence[int]` with values in `{0, 1}`.
- **Expected Metrics**: `TP`, `FN`, `FP`, `TN`, `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`.
- **Zero Denominator Policy**: Returns `0.0` when denominators are zero.
- **Validation**: Empty sequences or length mismatches raise `ValueError`.
