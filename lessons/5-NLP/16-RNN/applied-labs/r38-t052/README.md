# Applied Lab R38-T052: Cosine-Similarity Lookup with Boundary Handling

## Overview
Extends the Recurrent Neural Networks lesson by computing cosine similarity for local embedding vectors with explicit zero-vector boundary handling.

## Problem Formulation & Contract
- **Inputs**: Two numeric vectors `vec_a` and `vec_b` of identical length.
- **Normal Output**: Cosine similarity in `[-1.0, 1.0]`.
- **Boundary Decision**: When either vector has Euclidean norm 0 (all zeros), the calculation is undefined; returns documented sentinel `None` (`UNDEFINED_SENTINEL`).
- **Validation**: Mismatched dimensions or empty vectors raise `ValueError`.

## Example
- `[1, 0]` vs `[1, 0]` -> `1.0`
- `[1, 0]` vs `[0, 1]` -> `0.0`
- `[1, 0]` vs `[0, 0]` -> `None` (undefined sentinel)

## Scope & Limitations
- Offline calculation using Python standard library without external dependencies.
- Models word or token vector comparisons without requiring pre-trained weights.
