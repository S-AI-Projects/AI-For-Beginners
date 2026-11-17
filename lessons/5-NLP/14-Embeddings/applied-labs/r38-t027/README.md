# Applied Lab R38-T027: Cosine-Similarity Lookup — Worked Example with Checks

## Overview & Lesson Concept Connection

In **Word Embeddings** ([Lesson 14: Embeddings](../../README.md)), words, entities, or documents are mapped to continuous vector representations (such as Word2Vec, GloVe, or dense embedding layers).

In semantic search, similarity matching, and embedding lookups:
- Euclidean distance can be distorted by vector magnitude (e.g. document length or frequency biases).
- **Cosine Similarity** measures the cosine of the angle between two non-zero vectors in an inner product space, evaluating directional alignment irrespective of scale.

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \times \|v\|_2} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \times \sqrt{\sum_{i=1}^{n} v_i^2}}$$

---

## Step-by-Step Arithmetic Derivation (Worked Example)

Consider vectors $u = [3, 4]$ and $v = [4, 3]$:

1. **Compute Dot Product ($u \cdot v$):**
   $$u \cdot v = (3 \times 4) + (4 \times 3) = 12 + 12 = 24$$
2. **Compute Euclidean Norm of $u$ ($\|u\|_2$):**
   $$\|u\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
3. **Compute Euclidean Norm of $v$ ($\|v\|_2$):**
   $$\|v\|_2 = \sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = 5$$
4. **Compute Cosine Similarity:**
   $$\text{cos\_sim}(u, v) = \frac{24}{5 \times 5} = \frac{24}{25} = 0.96$$

This exact arithmetic derivation is encoded into the regression test suite (`test_worked_example_arithmetic_regression_check`) to prevent numeric drift or formulation errors.

---

## Technical Specifications

- **Identical Vectors:** $[1, 0]$ vs $[1, 0] \to 1.0$.
- **Orthogonal Vectors:** $[1, 0]$ vs $[0, 1] \to 0.0$.
- **Opposite Vectors:** $[1, 0]$ vs $[-1, 0] \to -1.0$.
- **Zero Vector Sentinel:** When either vector is a zero vector ($\|u\| = 0$ or $\|v\| = 0$), the geometric angle is undefined. The function explicitly returns the documented sentinel `None`.
- **Validation:**
  - Rejects empty vectors with `ValueError`.
  - Rejects mismatched dimensions with `ValueError`.
  - Rejects non-numeric, non-finite (`NaN`, `Inf`), or non-sequence types with `TypeError` / `ValueError`.

---

## Lab Files

- `example.py`: Implementation of `cosine_similarity`, `dot_product`, `vector_norm`, `find_most_similar`, and runnable CLI demonstration.
- `test_example.py`: Comprehensive test suite including regression checks for the worked arithmetic example, zero-vector sentinels, and boundary validation.
- `README.md`: Mathematical walkthrough and lab documentation.

---

## Running the Code

### Execute Runnable Demonstration
```bash
python3 lessons/5-NLP/14-Embeddings/applied-labs/r38-t027/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/5-NLP/14-Embeddings/applied-labs/r38-t027 -p "test_*.py"
```
