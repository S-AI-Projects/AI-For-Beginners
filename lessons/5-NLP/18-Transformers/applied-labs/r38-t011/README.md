# Applied Lab R38-T011: Cosine Similarity Lookup Concept Example

## Overview

This applied lab provides a minimal, self-contained reference implementation of **cosine similarity** using exclusively the Python 3 standard library.

In Transformer architectures (see [Attention Mechanisms and Transformers](../../README.md)), the core self-attention layer computes compatibility between query vectors ($Q$) and key vectors ($K$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

When representation vectors are L2-normalized ($\|u\|_2 = 1$), their dot product is mathematically equivalent to the **cosine similarity**:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

## Contract and Expected Behavior

- `[1, 0]` versus `[1, 0]` $\to$ `1.0` (identical direction, maximum alignment)
- `[1, 0]` versus `[0, 1]` $\to$ `0.0` (orthogonal vectors, uncorrelated attention)
- `[1, 0]` versus `[-1, 0]` $\to$ `-1.0` (opposite direction)
- Zero vector input (e.g. `[0, 0]`) $\to$ `UNDEFINED_SENTINEL` (`float('nan')`)

### Zero-Vector Handling
A zero vector has an L2 norm of 0, meaning its direction is undefined and normalization would involve division by zero. The function explicitly catches zero-norm vectors and returns `float('nan')` as a sentinel value.

## Scope and Limitations

> [!NOTE]
> **Simulation and Pedagogical Focus:**
> - This module is a pure mathematical concept example illustrating the geometric relationship between normalization, dot products, and similarity scores.
> - It does **not** download models, require GPUs, access cloud services, or simulate full multi-head transformer weights.
> - No claims are made about trained model performance or production deployment constraints.

## Running the Example

```bash
python3 lessons/5-NLP/18-Transformers/applied-labs/r38-t011/example.py
```

## Running the Tests

```bash
python3 -m unittest discover -s lessons/5-NLP/18-Transformers/applied-labs/r38-t011 -p "test_*.py" -v
```
