# Single Perceptron Update with Offline Service Adapter (R38-T045)

## 1. Task Objective

This applied lab demonstrates a single forward-prediction and parameter-update step for a classical binary **Perceptron**.

Rather than tightly coupling the update logic to an in-process calculation, the prediction step is abstracted behind a **callable dependency interface** using **dependency injection**. An offline, deterministic local adapter is provided to satisfy this interface without requiring external network calls or cloud dependencies.

---

## 2. Theoretical Background

### What is a Perceptron?
A Perceptron is the fundamental building block of artificial neural networks. First introduced by Frank Rosenblatt in 1957, a single-layer perceptron acts as a linear binary classifier that maps an $n$-dimensional real input vector $\mathbf{x} = [x_1, x_2, \dots, x_n]$ to a discrete binary decision $y \in \{0, 1\}$.

### Input Features ($x_i$)
The input vector $\mathbf{x} = [x_1, x_2, \dots, x_n]$ contains measurable features or attributes of an observation (for example, height and weight).

### Synaptic Weights ($w_i$)
Each input feature $x_i$ is paired with a learnable weight $w_i$. The weight quantifies the relative importance and directional influence (positive or negative) of that feature on the classification decision.

### Bias ($b$)
The bias $b$ acts as an offset or threshold parameter that allows the decision boundary to shift away from the origin ($\mathbf{0}$). It can be viewed mathematically as a weight connected to an input feature that is permanently fixed at $+1$.

---

## 3. Mathematical Formulas

### Weighted Sum ($z$)
The perceptron computes the linear combination (dot product plus bias) of its inputs:

$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b = (w_1 \cdot x_1) + (w_2 \cdot x_2) + \dots + (w_n \cdot x_n) + b$$

### Activation Threshold & Zero-Score Tie Rule
The linear score $z$ is passed through a Heaviside step activation function to produce a binary prediction $\hat{y} \in \{0, 1\}$:

$$\hat{y} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

> **Important (Zero-Score Tie Rule):** When the weighted sum is exactly zero ($z = 0$), the decision boundary is tied. Under the documented contract, a zero score predicts class **0** (negative). A score must be strictly positive ($z > 0$) to trigger class **1**.

### Prediction Error ($e$)
The error represents the discrepancy between the ground-truth target $y \in \{0, 1\}$ and the model's initial prediction $\hat{y} \in \{0, 1\}$:

$$\text{error} = y - \hat{y}$$

Because both $y$ and $\hat{y}$ are binary:
- If $y = 1, \hat{y} = 1 \implies \text{error} = 0$ (Correct)
- If $y = 0, \hat{y} = 0 \implies \text{error} = 0$ (Correct)
- If $y = 1, \hat{y} = 0 \implies \text{error} = +1$ (False Negative — weights and bias must increase)
- If $y = 0, \hat{y} = 1 \implies \text{error} = -1$ (False Positive — weights and bias must decrease)

### Perceptron Weight and Bias Update Rules
Parameters are updated using Rosenblatt's perceptron learning rule scaled by learning rate $\eta \ge 0$:

$$w_i^{\text{new}} = w_i^{\text{old}} + \eta \cdot \text{error} \cdot x_i$$

$$b^{\text{new}} = b^{\text{old}} + \eta \cdot \text{error}$$

---

## 4. Step-by-Step Calculation (Main Fixture)

Given the specification fixture:
- $\mathbf{w}^{\text{old}} = [0.0, 0.0]$
- $b^{\text{old}} = 0.0$
- $\mathbf{x} = [1.0, 2.0]$
- $y = 1$
- $\eta = 1.0$

### Step 1: Compute Weighted Sum ($z$)
$$z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + b = (0.0 \cdot 1.0) + (0.0 \cdot 2.0) + 0.0 = 0.0$$

### Step 2: Apply Decision Threshold and Zero-Score Rule
$$z = 0.0 \implies \hat{y} = 0$$

### Step 3: Compute Prediction Error ($e$)
$$\text{error} = y - \hat{y} = 1 - 0 = 1$$

### Step 4: Update Synaptic Weights
$$w_1^{\text{new}} = 0.0 + (1.0 \times 1 \times 1.0) = 1.0$$
$$w_2^{\text{new}} = 0.0 + (1.0 \times 1 \times 2.0) = 2.0$$

### Step 5: Update Bias
$$b^{\text{new}} = 0.0 + (1.0 \times 1) = 1.0$$

### Final State
- **Updated Weights:** `[1.0, 2.0]`
- **Updated Bias:** `1.0`
- **Initial Prediction:** `0`
- **Target:** `1`
- **Error:** `1`

---

## 5. Software Architecture: Offline Service Adapter

### The Callable Interface
In enterprise software engineering, models or services often delegate inferences to external inference microservices or dedicated accelerators. Hardcoding direct computation inside the update function couples business logic to a specific implementation.

We define a standard callable type signature:
```python
PredictionCallable = Callable[[Sequence[float], float, Sequence[float]], int]
```

### Dependency Injection
The update routine accepts the predictor callable as a parameter:
```python
def perceptron_update_step(
    weights: Sequence[float],
    bias: float,
    inputs: Sequence[float],
    target: int,
    learning_rate: float,
    predictor: PredictionCallable,
) -> PerceptronStepResult:
    ...
    prediction = predictor(weights, bias, inputs)
    ...
```
This enables seamless swapping between local unit-test stubs, offline heuristic calculators, or remote client adapters without rewriting or re-testing the core perceptron update logic.

### Deterministic Local Adapter
`local_deterministic_predictor` implements this callable interface purely in-memory:
- Computes $z = \sum w_i x_i + b$
- Evaluates the strict threshold `1 if z > 0.0 else 0`
- Returns a deterministic integer prediction for identical inputs every time.

### Explicit Limitations of the Mock
This local simulation adapter is an in-process educational mock. It does **not** establish or simulate:
1. **Real Network Communication:** No HTTP, gRPC, REST, or TCP socket traffic is generated.
2. **Authentication & Authorization:** No API tokens, keys, mutual TLS, or OAuth handshakes are processed.
3. **Provider Availability & SLAs:** No uptime guarantees, service health checks, or failover configurations exist.
4. **Network Latency & Jitter:** Execution is instantaneous and in-memory; there are no network hops or connection timeouts.
5. **Remote Error Handling:** No handling of 429 (rate limits), 500/503 (server errors), or transient network drops.
6. **Real Hardware / Cloud Acceleration:** Arithmetic is computed on the local CPU with standard Python floats; no GPUs, TPUs, or distributed clusters are utilized.
7. **Production API Payload Compatibility:** Does not validate JSON schemas or wire protocols expected by commercial inference endpoints.

---

## 6. Execution and Verification

### Prerequisites
- Python 3.8+ (standard library only; no external packages needed)

### Run the Demonstration
From the repository root:
```bash
python3 lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t045/example.py
```

### Run Automated Tests
```bash
python3 -m unittest discover -v -s lessons/3-NeuralNetworks/04-OwnFramework/applied-labs/r38-t045 -p "test_*.py"
```

### Expected Output
```
============================================================
Single Perceptron Update — Offline Service Adapter (R38-T045)
============================================================

--- Scenario 1: Main Regression Fixture ---
Inputs (x):          [1.0, 2.0]
Initial Weights (w): [0.0, 0.0]
Initial Bias (b):    0.0
Learning Rate (eta): 1.0
Target (y):          1

Calculations:
  Weighted sum (z) = (0.0 * 1.0) + (0.0 * 2.0) + 0.0 = 0.0
  Zero-score tie:     z = 0.0 <= 0 -> prediction = 0
  Error (y - y_hat)  = 1 - 0 = 1

Weight & Bias Updates:
  w_1_new = 0.0 + (1.0 * 1 * 1.0) = 1.0
  w_2_new = 0.0 + (1.0 * 1 * 2.0) = 2.0
  b_new   = 0.0 + (1.0 * 1) = 1.0

Final State:
  Updated Weights:   [1.0, 2.0]
  Updated Bias:      1.0
```

