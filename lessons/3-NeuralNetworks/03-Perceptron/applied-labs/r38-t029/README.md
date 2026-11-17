# R38-T029: Single Perceptron Update — Scenario Assessment

**Lesson:** [Perceptron](../../README.md) | `lessons/3-NeuralNetworks/03-Perceptron`

---

## Overview

This lab implements and assesses the **single perceptron** — the simplest trainable unit in neural networks. It covers:

1. The **forward pass**: computing a weighted score and applying a binary threshold activation.
2. The **perceptron learning rule**: a gradient-free weight update applied when the prediction is wrong.
3. **Three applied scenario assessments**: predict output, diagnose failure, and choose a boundary policy.

---

## Perceptron Specification

| Component | Formula / Value |
|---|---|
| Input | 2D vector **x** = [x₀, x₁] |
| Parameters | weights **w** = [w₀, w₁], bias *b* |
| Score (pre-activation) | *score* = w₀·x₀ + w₁·x₁ + *b* |
| **Tie rule (documented)** | *score ≥ 0* → predict **1**; *score < 0* → predict **0** |
| Learning rule | δ = lr·(target − prediction); **w** ← **w** + δ·**x**; *b* ← *b* + δ |
| Update condition | Only applied when prediction ≠ target |

---

## Step-by-Step Canonical Example (Spec)

**Setup:** weights = [0, 0], bias = 0, input = [1, 2], target = 1, lr = 1, initial prediction = 0

```
score     = 0·1 + 0·2 + 0 = 0
prediction = 0   (spec states initial pred = 0, illustrating the update)

delta   = 1 · (1 − 0) = 1
w_new   = [0, 0] + 1 · [1, 2] = [1, 2]
b_new   = 0 + 1 = 1
```

**Result:** weights → [1, 2], bias → 1 ✅

> [!NOTE]
> With the documented tie rule (score ≥ 0 → 1), a zero-weight/zero-bias perceptron on input [1, 2]
> computes score = 0 and would actually predict **1** (no update needed). The spec's
> "initial prediction = 0" is treated as a formula illustration using an assumed pre-tie-rule state.
> In tests, `bias = -0.5` is used to force `score < 0` → `prediction = 0`, matching the spec's intent.

---

## Scenario Assessments

### Scenario A — Predict Output

Given fixed weights, bias, and an input vector, compute the score and binary prediction.

| Example | Weights | Bias | Input | Score | Prediction |
|---------|---------|------|-------|-------|------------|
| After update | [1, 2] | 1 | [1, 2] | 6.0 | 1 |
| Tie-rule test | [1, −1] | 0 | [1, 1] | 0.0 | **1** (tie rule) |
| Negative | [−1, −1] | 0 | [1, 1] | −2.0 | 0 |

**Key insight:** The zero-score tie rule is a *design choice* — it must be explicitly documented because it determines how the decision boundary handles on-boundary points.

---

### Scenario B — Diagnose Failure

Run a **frozen** (non-learning) perceptron over a labelled dataset and identify misclassified samples.

**AND-gate dataset** (weights = [1, 1], bias = −1.5):

| Input | Target | Score | Prediction | Correct? |
|-------|--------|-------|------------|----------|
| [0, 0] | 0 | −1.5 | 0 | ✅ |
| [1, 0] | 0 | −0.5 | 0 | ✅ |
| [0, 1] | 0 | −0.5 | 0 | ✅ |
| [1, 1] | 1 | +0.5 | 1 | ✅ |

**Broken model** (weights = [0, 0], bias = 0 — always predicts 1 via tie rule):

| Input | Target | Prediction | Correct? |
|-------|--------|------------|----------|
| [0, 0] | 0 | 1 | ❌ |
| [1, 0] | 0 | 1 | ❌ |
| [0, 1] | 0 | 1 | ❌ |
| [1, 1] | 1 | 1 | ✅ |

Accuracy: 25% — misclassified indices [0, 1, 2]

**Failure diagnosis:** Zero weights + zero bias → decision boundary at score = 0 → tie rule assigns all inputs to class 1. This is a **degenerate model** that needs training.

---

### Scenario C — Choose Boundary Policy

At score = 0 (a point exactly on the decision boundary), three policies differ:

| Policy | score = 0 | score > 0 | score < 0 | Use Case |
|--------|-----------|-----------|-----------|----------|
| `positive_tie` | **1** | 1 | 0 | Default (documented here) |
| `negative_tie` | **0** | 1 | 0 | Conservative: boundary → negative |
| `abstain` | **−1** | 1 | 0 | Reject ambiguous inputs |

**Key insight:** The choice of tie-breaking policy affects which class "owns" the decision boundary hyperplane. All three policies are functionally identical for non-zero scores; they diverge only at the exact boundary.

---

## Files

| File | Purpose |
|---|---|
| `example.py` | `Perceptron` class, `perceptron_update()`, three scenario functions, demo `main()` |
| `test_example.py` | 20+ unit tests across all components and all three scenarios |
| `README.md` | This document |

---

## Usage

```bash
# Run the demonstration
python3 lessons/3-NeuralNetworks/03-Perceptron/applied-labs/r38-t029/example.py

# Run the test suite
python3 -m unittest discover \
  -s lessons/3-NeuralNetworks/03-Perceptron/applied-labs/r38-t029 \
  -p "test_*.py" -v
```

---

## Key Learning Points

1. **Perceptron as a linear classifier:** It partitions the input space with a hyperplane defined by **w** and *b*. Only linearly separable data can be learned perfectly.
2. **Gradient-free update rule:** The perceptron rule adjusts weights proportionally to the error (target − prediction), not to a gradient — it predates backpropagation.
3. **Tie-breaking is a design decision:** The zero-score boundary assignment must be explicitly documented so downstream code and tests agree on the contract.
4. **Failure diagnosis:** Inspecting which samples are misclassified (not just aggregate accuracy) pinpoints whether a model is systematically biased toward one class.
5. **Policy comparison:** Swapping the tie-breaking policy is an example of separating *mechanism* (score computation) from *policy* (threshold decision), a core software design principle.
