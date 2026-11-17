# Cosine-Similarity Lookup — Evaluation Experiment (R38-T046)

## 1. Task Objective

In Natural Language Processing (NLP) and Recurrent Neural Networks (RNNs), words and text sequences are represented as dense numeric vectors called **embeddings**. Measuring semantic relatedness between two words or sequences often relies on computing geometric angles between their embedding vectors.

This applied software-engineering lab implements **Cosine Similarity** using only Python's standard library. It demonstrates:
1. Normalization and dot product computations.
2. Explicit handling of zero-magnitude vectors using a documented undefined-result sentinel (`None`).
3. An evaluation experiment comparing three controlled vector configurations across dimensions (2D, 4D, 8D) to analyze correctness and computational scaling.

---

## 2. Mathematical Background

### What is Cosine Similarity?
Cosine similarity measures the cosine of the angle $\theta$ between two non-zero vectors in an inner product space. It evaluates directional orientation rather than magnitude:
- Value of $+1.0$: Vectors point in identical directions ($0^\circ$ angle).
- Value of $0.0$: Vectors are orthogonal ($90^\circ$ angle, linearly uncorrelated).
- Value of $-1.0$: Vectors point in diametrically opposite directions ($180^\circ$ angle).

### The Formula
$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i \cdot B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Where:
- $\mathbf{A} \cdot \mathbf{B}$ is the **dot product** (scalar product) of vectors $\mathbf{A}$ and $\mathbf{B}$.
- $\|\mathbf{A}\|$ and $\|\mathbf{B}\|$ are the **Euclidean norms** ($L_2$ magnitudes) of the vectors.

### Why Zero Vectors Need Special Handling
When a vector has magnitude zero ($\|\mathbf{A}\| = 0$ or $\|\mathbf{B}\| = 0$), the denominator becomes:
$$\|\mathbf{A}\| \cdot \|\mathbf{B}\| = 0$$
In geometry, a zero vector has no direction; hence, the angle between a zero vector and any other vector is geometrically and mathematically undefined.

Attempting to evaluate this division in computer arithmetic yields a `ZeroDivisionError` or IEEE 754 `NaN`. Rather than silently returning an arbitrary number (such as `0.0`, which would falsely imply orthogonality), our function explicitly returns the documented sentinel:
```python
UNDEFINED_SENTINEL = None
```
This guarantees fail-safe behavior and prevents misleading similarity scores in downstream NLP pipelines.

---

## 3. Mandatory Specification Fixtures

| Case | Vector A | Vector B | Dot Product | Norm(A) | Norm(B) | Expected Result | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1: Identical** | `[1, 0]` | `[1, 0]` | 1.0 | 1.0 | 1.0 | `1.0` | Exact same direction |
| **Case 2: Orthogonal** | `[1, 0]` | `[0, 1]` | 0.0 | 1.0 | 1.0 | `0.0` | Perpendicular ($90^\circ$) |
| **Case 3: Zero Vector** | `[0, 0]` | `[1, 0]` | 0.0 | 0.0 | 1.0 | `None` | Geometrically undefined |
| **Opposite** | `[1, 0]` | `[-1, 0]` | -1.0 | 1.0 | 1.0 | `-1.0` | Diametrically opposite |

---

## 4. Controlled Evaluation Experiments

To fulfill the evaluation experiment requirement, three distinct, deterministic vector configurations are compared across dimensionalities:

### Configuration 1: Small Vectors (2 Dimensions)
- $\mathbf{A} = [3.0, 4.0], \mathbf{B} = [4.0, 3.0]$
- Dot product: $(3 \times 4) + (4 \times 3) = 12 + 12 = 24.0$
- Norms: $\|\mathbf{A}\| = \sqrt{3^2 + 4^2} = 5.0$, $\|\mathbf{B}\| = \sqrt{4^2 + 3^2} = 5.0$
- Expected Similarity: $\frac{24.0}{5.0 \times 5.0} = \frac{24}{25} = 0.96$

### Configuration 2: Medium Vectors (4 Dimensions)
- $\mathbf{A} = [1.0, 2.0, 3.0, 4.0], \mathbf{B} = [4.0, 3.0, 2.0, 1.0]$
- Dot product: $(1 \times 4) + (2 \times 3) + (3 \times 2) + (4 \times 1) = 4 + 6 + 6 + 4 = 20.0$
- Norms: $\|\mathbf{A}\| = \sqrt{1 + 4 + 9 + 16} = \sqrt{30}$, $\|\mathbf{B}\| = \sqrt{30}$
- Expected Similarity: $\frac{20.0}{\sqrt{30} \cdot \sqrt{30}} = \frac{20}{30} = \frac{2}{3} \approx 0.666667$

### Configuration 3: Larger Local Vectors (8 Dimensions)
- $\mathbf{A} = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]$
- $\mathbf{B} = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]$
- Orthogonal block configuration: no overlapping non-zero dimensions.
- Dot product: $0.0$
- Norms: $\|\mathbf{A}\| = \sqrt{4} = 2.0, \|\mathbf{B}\| = \sqrt{4} = 2.0$
- Expected Similarity: $\frac{0.0}{2.0 \times 2.0} = 0.0$

### Correctness and Complexity Trade-Off
- **Mathematical Complexity:** Time complexity is strictly linear with respect to vector dimension: $\mathcal{O}(D)$, where $D$ is the length of the vector. Memory overhead is $\mathcal{O}(1)$ beyond input storage.
- **Microbenchmarking Observation:** Across 10,000 iterations per configuration, operation times remain on the order of a few microseconds per calculation, scaling predictably with vector dimension on CPU.

---

## 5. Scope and Explicit Limitations

This is a local mathematical and algorithmic demonstration. It does **not**:
1. Train, fine-tune, or download neural word embeddings (such as Word2Vec, GloVe, FastText, or BERT).
2. Measure linguistic semantic accuracy, synonym quality, or language modeling perplexity.
3. Connect to external vector databases, cloud APIs, or inference endpoints.
4. Utilize GPU/TPU vector acceleration libraries or SIMD hardware intrinsics.

---

## 6. Execution Instructions

### Prerequisites
- Python 3.8+ (standard library only; no third-party packages required).

### Run the Example
```bash
python3 lessons/5-NLP/16-RNN/applied-labs/r38-t046/example.py
```

### Run Automated Tests
```bash
python3 -m unittest discover -v -s lessons/5-NLP/16-RNN/applied-labs/r38-t046 -p "test_*.py"
```

### Expected Output
```
=================================================================
Cosine-Similarity Lookup — Evaluation Experiment (R38-T046)
=================================================================

--- Mandatory Specification Fixtures ---
Case 1 (Identical):  A=[1.0, 0.0], B=[1.0, 0.0] -> Result: 1.0 (Expected: 1.0)
Case 2 (Orthogonal): A=[1.0, 0.0], B=[0.0, 1.0] -> Result: 0.0 (Expected: 0.0)
Case 3 (Zero vector): A=[0.0, 0.0], B=[1.0, 0.0] -> Result: None (Expected: None)
Opposite Direction:  A=[1.0, 0.0], B=[-1.0, 0.0] -> Result: -1.0 (Expected: -1.0)

--- Controlled Evaluation Experiments ---
Configuration: Config 1: 2D Non-trivial Acute Vectors
  Dimension:           2
  Vector A:            [3.0, 4.0]
  Vector B:            [4.0, 3.0]
  Expected Similarity: 0.96
  Actual Similarity:   0.960000
  Correct:             True
...
```

