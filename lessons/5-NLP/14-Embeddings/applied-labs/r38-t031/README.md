# Applied Lab R38-T031: Cosine-Similarity Lookup — Runnable Concept Example

## Overview & Lesson Concept Connection

In [Lesson 14: Embeddings](../../README.md), words, tokens, and documents are projected into continuous, dense vector spaces (such as Word2Vec, GloVe, or deep neural network embedding layers).

In these learned vector spaces:
- The **direction** of an embedding vector captures its semantic meaning and syntactic relationships.
- The **magnitude** (L2 norm) often reflects token frequency or corpus statistics rather than semantic meaning.
- Therefore, **cosine similarity** is the standard metric used to compare embeddings rather than Euclidean distance:

$$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$

When vectors are normalized to unit length ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), cosine similarity simplifies directly to their inner (dot) product $\mathbf{u} \cdot \mathbf{v}$.

---

## Zero-Norm Boundary Case

In practical embedding pipelines, vectors may have an L2 norm of zero due to:
1. All-zero padding representations (`<PAD>` tokens).
2. Out-of-vocabulary or uninitialized embedding table lookups.
3. Feature masking or dropout in offline preprocessing.

Because the angle between a zero vector and any other vector is geometrically undefined (leading to a division by zero in the denominator), this implementation explicitly detects zero-norm vectors ($\|\mathbf{v}\| \le 10^{-12}$) and returns the documented sentinel:

```python
UNDEFINED_SENTINEL = None
```

Returning `None` prevents unpredictable IEEE floating-point `NaN`/`ZeroDivisionError` exceptions while cleanly alerting calling pipelines to handle the missing/uninformative representation.

---

## Contract Specification

### Stated Fixtures & Expected Outputs:
| Operand $\mathbf{u}$ | Operand $\mathbf{v}$ | Relationship | Expected Result |
| :--- | :--- | :--- | :--- |
| `[1, 0]` | `[1, 0]` | Identical vectors | `1.0` |
| `[1, 0]` | `[0, 1]` | Orthogonal vectors | `0.0` |
| `[1, 0]` | `[-1, 0]` | Opposite vectors | `-1.0` |
| `[1, 0]` | `[0, 0]` | Zero-norm vector | `None` (`UNDEFINED_SENTINEL`) |
| `[0, 0]` | `[0, 0]` | Both zero vectors | `None` (`UNDEFINED_SENTINEL`) |
| `[3, 4]` | `[4, 3]` | Arbitrary vectors | `0.96` |

### Error Handling:
- **Dimension Mismatch**: Vectors of differing lengths raise `ValueError`.
- **Empty Vectors**: Vectors of length 0 raise `ValueError`.
- **Non-finite Values**: `NaN` or `Inf` elements raise `ValueError`.
- **Non-numeric Elements**: Strings or booleans raise `TypeError`.

---

## Scope & Limitations

- **Pure Python Standard Library**: Implemented using Python's built-in `math` module; no external ML dependencies (e.g. PyTorch, TensorFlow, Scikit-Learn) are required.
- **Educational Simulation**: Toy 2D fixtures (`[1, 0]`, `[0, 1]`) simulate high-dimensional dense embeddings without downloading multi-gigabyte pre-trained models.
- **Hardware Agnostic**: Runs locally on CPU in any standard Python 3.8+ environment without requiring GPUs or cloud services.

---

## Execution Instructions

Run the runnable concept example:
```bash
python3 lessons/5-NLP/14-Embeddings/applied-labs/r38-t031/example.py
```

Run the automated test suite:
```bash
python3 -m unittest discover -s lessons/5-NLP/14-Embeddings/applied-labs/r38-t031 -p "test_*.py" -v
```
