# Applied Lab R38-T057: Worked Cosine-Similarity Example

## Overview
Provides a step-by-step worked trace of cosine similarity calculations between token representations in Transformers, including dot product, Euclidean norms, and zero-vector guard handling.

## Worked Trace
1. Dot Product: $\sum (a_i \cdot b_i)$
2. Vector Norms: $\|a\| = \sqrt{\sum a_i^2}$, $\|b\| = \sqrt{\sum b_i^2}$
3. Zero-vector Check: If $\|a\|=0$ or $\|b\|=0$, return `UNDEFINED_SENTINEL` (`None`).
4. Similarity: $rac{a \cdot b}{\|a\| \|b\|}$ clamped to `[-1.0, 1.0]`.
