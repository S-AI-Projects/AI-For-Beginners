# Applied Lab R38-T012: Cosine Similarity Boundary-Case Extension

## Overview

This lab extends [Embeddings](../../README.md) with an explicit analysis of boundary conditions when computing cosine similarity between embedding vectors:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

## Boundary Decisions

1. **Smallest Valid Inputs:** 1D vectors with non-zero scalar values (e.g., `[5.0]` vs `[2.0]` $\to$ `1.0`).
2. **Orthogonal Boundary:** `[1, 0]` vs `[0, 1]` $\to$ `0.0`.
3. **Opposite Boundary:** `[1, 0]` vs `[-1, 0]` $\to$ `-1.0`.
4. **Zero-Norm Boundary:** `[0, 0]` vs `[1, 0]` $\to$ `UNDEFINED_SENTINEL` (`float('nan')`). Zero vectors have no direction, making normalization impossible.
5. **Invalid Inputs:** Dimension mismatch or empty sequences raise `ValueError`.

## Simulation Scope & Limitations

> [!NOTE]
> This is a minimal standard-library concept implementation. It does not train embeddings or download models.

## Verification

```bash
python3 -m unittest discover -s lessons/5-NLP/14-Embeddings/applied-labs/r38-t012 -p "test_*.py" -v
```
