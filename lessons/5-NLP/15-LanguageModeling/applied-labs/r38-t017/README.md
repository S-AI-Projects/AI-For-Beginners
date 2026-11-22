# Applied Lab R38-T017: Cosine-Similarity Lookup — Worked Example with Checks

## Overview

In Language Modeling (see [Language Modeling](../../README.md)), evaluating contextual alignment between word or token vectors relies on cosine similarity:

$$\text{CosineSimilarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

## Contract & Worked Explanation

- **Fixture 1 (Identical):** `[1, 0]` vs `[1, 0]`
  - $\text{dot} = 1.0$, $\|u\| = 1.0$, $\|v\| = 1.0 \to 1.0 / (1.0 \times 1.0) = 1.0$.
- **Fixture 2 (Orthogonal):** `[1, 0]` vs `[0, 1]`
  - $\text{dot} = 0.0$, $\|u\| = 1.0$, $\|v\| = 1.0 \to 0.0 / (1.0 \times 1.0) = 0.0$.
- **Fixture 3 (Zero Vector):** `[0, 0]` vs `[1, 0]`
  - $\|u\| = 0.0 \to$ division by zero prevented; returns `UNDEFINED_SENTINEL` (`float('nan')`).

## Simulation Scope & Limitations

> [!NOTE]
> This applied lab provides a minimal, self-contained educational example using the Python standard library. It does not load large pre-trained n-gram models or neural language model checkpoints.

## Running Tests

```bash
python3 -m unittest discover -s lessons/5-NLP/15-LanguageModeling/applied-labs/r38-t017 -p "test_*.py" -v
```
