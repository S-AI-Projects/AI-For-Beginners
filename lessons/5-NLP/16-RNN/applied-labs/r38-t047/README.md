# Cosine-Similarity Lookup — Scenario Assessment (R38-T047)

## 1. Task Objective

In Natural Language Processing (NLP) and Recurrent Neural Networks (RNNs), vector embeddings represent words and sequence states in high-dimensional space. To evaluate candidate embeddings, search systems use **Cosine Similarity**.

This applied software-engineering lab builds directly upon the mathematical foundation established in **R38-T046**, framing three applied questions:
1. **Predict an Output:** Calculate expected similarities for identical ($1.0$) and orthogonal ($0.0$) vector pairs.
2. **Diagnose a Failure:** Analyze why naive implementations crash or emit `NaN` when presented with zero vectors.
3. **Choose a Boundary Policy:** Enforce an explicit, safe sentinel policy (`None`) for undefined zero-magnitude inputs rather than inventing an arbitrary similarity number.

---

## 2. Mathematical Foundation

### What is Cosine Similarity?
Cosine similarity measures the cosine of the angle $\theta$ between two non-zero vectors $\mathbf{A}$ and $\mathbf{B}$:

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i \cdot B_i}{\sqrt{\sum_{i=1}^n A_i^2} \cdot \sqrt{\sum_{i=1}^n B_i^2}}$$

- **Dot Product ($\mathbf{A} \cdot \mathbf{B}$):** Sum of element-wise products.
- **Vector Norms ($\|\mathbf{A}\|, \|\mathbf{B}\|$):** Geometric Euclidean lengths ($L_2$ magnitudes).
- **Range:** Results strictly fall in $[-1.0, 1.0]$.

### Zero-Vector Policy (Preserved from R38-T046)
If $\|\mathbf{A}\| = 0.0$ or $\|\mathbf{B}\| = 0.0$, the geometric angle between the vectors is undefined.
Under the documented contract:
```python
UNDEFINED_SENTINEL = None
```
The implementation intercepts zero norms prior to division, returning `None` instead of raising `ZeroDivisionError` or returning a misleading numeric score such as `0.0` or `1.0`.

---

## 3. The Three Applied Scenario Questions

### Question 1: Predict an Output
- **Question:** What cosine similarity should the example produce for identical vectors $[1, 0]$ vs $[1, 0]$ and orthogonal vectors $[1, 0]$ vs $[0, 1]$?
- **Correct Answer:** Identical: `1.0`; Orthogonal: `0.0`.
- **Rationale:** Identical vectors point in the exact same direction ($\theta = 0^\circ$). Dot product is $1.0$, magnitudes are $1.0$, yielding $\cos(0^\circ) = 1.0$. Orthogonal vectors are perpendicular ($\theta = 90^\circ$) with dot product $0.0$, yielding $\cos(90^\circ) = 0.0$.
- **Executable Check:** Evaluates `cosine_similarity([1, 0], [1, 0])` and asserts `== 1.0`; evaluates `cosine_similarity([1, 0], [0, 1])` and asserts `== 0.0`.

### Question 2: Diagnose a Failure
- **Question:** Why would a naive cosine-similarity implementation fail on input $\mathbf{A}=[0, 0], \mathbf{B}=[1, 0]$?
- **Correct Answer:** The denominator contains the product of vector norms ($\|\mathbf{A}\| \cdot \|\mathbf{B}\|$). The zero vector has magnitude zero, resulting in a zero denominator. Direct arithmetic division raises an unhandled `ZeroDivisionError` or emits IEEE `NaN`.
- **Rationale:** Formula division $\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}$ requires non-zero denominators. A naive implementation that lacks magnitude pre-checks crashes on zero inputs.
- **Executable Check:** Calls `cosine_similarity([0, 0], [1, 0])`, `[1, 0]` vs `[0, 0]`, and `[0, 0]` vs `[0, 0]`, asserting that no exception is raised and all return `None`.

### Question 3: Choose a Boundary Policy
- **Question:** What should the application do when cosine similarity is undefined because one or both vectors are zero?
- **Correct Answer:** Return the documented undefined-result sentinel (`None`) instead of inventing an arbitrary numerical similarity score.
- **Rationale:** A zero vector has no orientation in geometric space. Returning `0.0` would falsely claim the vectors are perpendicular; returning `1.0` would falsely claim they are parallel. Returning `None` explicitly signals an undefined calculation to calling code.
- **Executable Check:** Verifies that `UNDEFINED_SENTINEL is None`, zero inputs return `None`, while normal non-zero inputs continue returning valid numeric floats.

---

## 4. Scope and Educational Limitations

This lab is a local mathematical assessment. It does **not**:
1. Train or download neural word embeddings (such as Word2Vec, GloVe, or FastText).
2. Establish real-world NLP model accuracy or semantic text retrieval quality.
3. Call remote cloud inference services or require GPU hardware.

---

## 5. Execution Instructions

### Prerequisites
- Python 3.8+ (standard library only; no external packages needed).

### Run the Scenario Assessment
```bash
python3 lessons/5-NLP/16-RNN/applied-labs/r38-t047/example.py
```

### Run Automated Tests
```bash
python3 -m unittest discover -v -s lessons/5-NLP/16-RNN/applied-labs/r38-t047 -p "test_*.py"
```

### Expected Output
```
======================================================================
Cosine-Similarity Lookup — Scenario Assessment (R38-T047)
======================================================================

[Q1] Question Type:    predict_output
     Question:         What cosine similarity should the example produce for identical vectors [1, 0] vs [1, 0] and orthogonal vectors [1, 0] vs [0, 1]?
     Scenario:         A=[1.0, 0.0], B=[1.0, 0.0] (identical); A=[1.0, 0.0], B=[0.0, 1.0] (orthogonal)
     Expected Answer:  Identical: 1.0; Orthogonal: 0.0
     Actual Result:    Identical: 1.0; Orthogonal: 0.0
     Rationale:        Identical vectors point in the exact same direction (theta=0 deg), yielding dot=1.0, norms=1.0,1.0, and cos(0)=1.0. Orthogonal vectors are perpendicular (theta=90 deg) with dot=0.0, yielding cos(90)=0.0.
     Executable Check: [PASS]
----------------------------------------------------------------------
[Q2] Question Type:    diagnose_failure
     Question:         Why would a naive cosine-similarity implementation fail on input A=[0, 0], B=[1, 0]?
     Scenario:         A=[0.0, 0.0], B=[1.0, 0.0] (zero vector input)
     Expected Answer:  Denominator becomes zero because norm([0, 0]) = 0.0; naive direct division raises ZeroDivisionError or emits NaN.
     Actual Result:    Handled safely without crash: returned None (None) across all zero-vector combinations.
     Rationale:        Cosine similarity is dot(A, B) / (norm(A) * norm(B)). When either vector has zero magnitude, the denominator is zero. Direct division is arithmetic error. Handling must intercept zero norm before dividing.
     Executable Check: [PASS]
----------------------------------------------------------------------
[Q3] Question Type:    choose_boundary_policy
     Question:         What should the application do when cosine similarity is undefined because one or both vectors are zero?
     Scenario:         Boundary condition: norm(A) == 0.0 or norm(B) == 0.0
     Expected Answer:  Return the documented undefined-result sentinel (None) instead of an invented numeric score.
     Actual Result:    Policy returns None; normal vectors return valid float (1.0000).
     Rationale:        A zero vector has no orientation or angle in geometric space. Returning 0.0 would falsely claim the vectors are perpendicular; returning 1.0 would falsely claim they are parallel. Returning None explicitly signals an undefined calculation to calling code.
     Executable Check: [PASS]
----------------------------------------------------------------------

Summary: All 3 scenario assessment checks PASSED successfully.
```

