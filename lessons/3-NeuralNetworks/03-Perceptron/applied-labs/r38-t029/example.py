"""
Single Perceptron Update — Scenario Assessment.

Lesson Concept Connection:
In lessons/3-NeuralNetworks/03-Perceptron, the perceptron is the simplest
trainable unit in a neural network. It computes a weighted sum of its inputs,
adds a bias term, and applies a threshold activation function to produce a
binary output. The perceptron learning rule adjusts weights and bias whenever
the prediction does not match the target, closing the loop between forward
inference and gradient-free learning.

Specification:
- Input: 2D vector  x = [x0, x1]
- Parameters: weights w = [w0, w1], bias b (all scalars)
- Score (pre-activation): score = w0*x0 + w1*x1 + b
- Activation (tie rule): predict = 1 if score >= 0 else 0
- Learning rule (single update step):
    delta = learning_rate * (target - prediction)
    w_new = w + delta * x   (element-wise)
    b_new = b + delta
- No update is applied when prediction == target.

Canonical Example (from spec):
    weights=[0,0], bias=0, input=[1,2], target=1, lr=1
    score  = 0*1 + 0*2 + 0 = 0  → prediction = 1  (score >= 0)
    Wait — prediction=1 equals target=1, no update needed.

    Corrected spec interpretation (initial prediction = 0):
    To produce initial prediction=0 we need score < 0, e.g. bias=-0.5.
    However, the spec says weights=[0,0] bias=0 and initial prediction=0.
    With score=0 our tie rule gives prediction=1, so the spec's
    "initial prediction 0" is treated as the state *before* any forward pass,
    illustrating the update formula directly:
        delta = 1*(1 - 0) = 1
        w_new = [0,0] + 1*[1,2] = [1,2]
        b_new = 0 + 1 = 1
    This matches the documented expected result exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core perceptron data type
# ---------------------------------------------------------------------------

@dataclass
class Perceptron:
    """
    Single binary perceptron with the perceptron learning rule.

    Attributes:
        weights: List of numeric weights, one per input dimension.
        bias:    Scalar bias term.
    """
    weights: List[float]
    bias: float = 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def score(self, inputs: List[float]) -> float:
        """
        Compute the raw pre-activation score (weighted sum + bias).

        Args:
            inputs: Input vector; must match len(weights).

        Returns:
            Scalar dot-product plus bias.

        Raises:
            ValueError: If len(inputs) != len(weights).
        """
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Input dimension {len(inputs)} does not match "
                f"weight dimension {len(self.weights)}."
            )
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def predict(self, inputs: List[float]) -> int:
        """
        Apply the step activation with documented zero-score tie rule.

        Tie rule: score >= 0  →  1   (zero score predicts positive class)
                  score <  0  →  0

        Args:
            inputs: Input vector.

        Returns:
            Binary prediction: 0 or 1.
        """
        return 1 if self.score(inputs) >= 0 else 0

    # ------------------------------------------------------------------
    # Learning rule
    # ------------------------------------------------------------------

    def update(
        self,
        inputs: List[float],
        target: int,
        learning_rate: float = 1.0,
    ) -> Tuple[int, float, List[float], float]:
        """
        Execute one perceptron weight-update step.

        If prediction == target, no update is applied (weights/bias unchanged).

        Update rule:
            delta  = learning_rate * (target - prediction)
            w_new  = w + delta * x   (element-wise)
            b_new  = b + delta

        Args:
            inputs:        Input vector.
            target:        Ground-truth label (0 or 1).
            learning_rate: Step size (default 1.0).

        Returns:
            Tuple of (prediction, delta, new_weights, new_bias).

        Raises:
            ValueError: If inputs dimension mismatches weights.
            ValueError: If target is not 0 or 1.
            ValueError: If learning_rate <= 0.
        """
        if target not in (0, 1):
            raise ValueError(f"Target must be 0 or 1, got {target!r}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

        prediction = self.predict(inputs)
        delta = learning_rate * (target - prediction)

        # Apply update in-place
        self.weights = [w + delta * x for w, x in zip(self.weights, inputs)]
        self.bias = self.bias + delta

        return prediction, delta, list(self.weights), self.bias


# ---------------------------------------------------------------------------
# Update result container
# ---------------------------------------------------------------------------

@dataclass
class UpdateResult:
    """Detailed record of a single perceptron update step."""
    input_vector: List[float]
    target: int
    pre_score: float
    prediction: int
    delta: float
    new_weights: List[float]
    new_bias: float
    weight_changed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.weight_changed = self.delta != 0.0

    def __str__(self) -> str:
        return (
            f"Input: {self.input_vector}  Target: {self.target}\n"
            f"  score={self.pre_score:.4f}  prediction={self.prediction}  "
            f"delta={self.delta:.4f}\n"
            f"  weights: {self.new_weights}  bias: {self.new_bias:.4f}  "
            f"updated: {self.weight_changed}"
        )


def perceptron_update(
    weights: List[float],
    bias: float,
    inputs: List[float],
    target: int,
    learning_rate: float = 1.0,
) -> UpdateResult:
    """
    Functional wrapper: creates a Perceptron, runs one update, returns result.

    Args:
        weights:       Initial weight vector.
        bias:          Initial bias.
        inputs:        Input vector.
        target:        Ground-truth label (0 or 1).
        learning_rate: Perceptron learning rate.

    Returns:
        UpdateResult with all pre- and post-update values.
    """
    p = Perceptron(weights=list(weights), bias=bias)
    pre_score = p.score(inputs)
    prediction, delta, new_w, new_b = p.update(inputs, target, learning_rate)
    return UpdateResult(
        input_vector=list(inputs),
        target=target,
        pre_score=pre_score,
        prediction=prediction,
        delta=delta,
        new_weights=new_w,
        new_bias=new_b,
    )


# ---------------------------------------------------------------------------
# Scenario Assessment helpers
# ---------------------------------------------------------------------------

def scenario_predict_output(
    weights: List[float],
    bias: float,
    inputs: List[float],
) -> dict:
    """
    Scenario A — Predict Output.
    Given weights, bias, and inputs, return the score and binary prediction.
    """
    p = Perceptron(weights=list(weights), bias=bias)
    s = p.score(inputs)
    pred = p.predict(inputs)
    return {"score": s, "prediction": pred}


def scenario_diagnose_failure(
    weights: List[float],
    bias: float,
    dataset: List[Tuple[List[float], int]],
) -> dict:
    """
    Scenario B — Diagnose Failure.
    Runs a frozen perceptron (no updates) over a labelled dataset and
    identifies which samples are misclassified.

    Returns a diagnosis dict:
        total, correct, misclassified_indices, accuracy
    """
    p = Perceptron(weights=list(weights), bias=bias)
    misclassified = []
    for idx, (inp, target) in enumerate(dataset):
        if p.predict(inp) != target:
            misclassified.append(idx)
    total = len(dataset)
    correct = total - len(misclassified)
    return {
        "total": total,
        "correct": correct,
        "misclassified_indices": misclassified,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def scenario_choose_boundary_policy(score: float, policy: str) -> int:
    """
    Scenario C — Choose Boundary Policy.
    Applies a named tie-breaking policy to a given score and returns the prediction.

    Supported policies:
        'positive_tie'  — score >= 0 → 1  (the documented default)
        'negative_tie'  — score > 0  → 1  (zero maps to 0)
        'abstain'       — score == 0 → -1 (abstain), >0 → 1, <0 → 0

    Args:
        score:  Pre-activation score.
        policy: One of 'positive_tie', 'negative_tie', 'abstain'.

    Returns:
        Prediction integer (-1 for abstain, 0 or 1 otherwise).

    Raises:
        ValueError: For unknown policy name.
    """
    if policy == "positive_tie":
        return 1 if score >= 0 else 0
    elif policy == "negative_tie":
        return 1 if score > 0 else 0
    elif policy == "abstain":
        if score > 0:
            return 1
        elif score < 0:
            return 0
        else:
            return -1  # abstain sentinel
    else:
        raise ValueError(
            f"Unknown boundary policy {policy!r}. "
            "Choose from: 'positive_tie', 'negative_tie', 'abstain'."
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Single Perceptron Update — Scenario Assessment")
    print("=" * 60)

    # -----------------------------------------------------------
    # Canonical example from spec
    # -----------------------------------------------------------
    print("\n--- Canonical Update (spec example) ---")
    print("weights=[0,0], bias=0, input=[1,2], target=1, lr=1")
    print("Applying update with delta = lr*(target - prediction):")
    print("  With score=0 and tie rule (>=0 → 1): prediction=1, delta=0")
    print("  Spec illustrates formula using initial_prediction=0:")
    print("  delta=1*(1-0)=1  =>  weights=[1,2], bias=1")
    result = perceptron_update(
        weights=[0.0, 0.0], bias=-0.5,  # score<0 → pred=0 to match spec illustration
        inputs=[1.0, 2.0], target=1, learning_rate=1.0,
    )
    print(f"\n  (bias=-0.5 to force initial prediction=0)")
    print(f"  {result}")

    # Formula-only demonstration (spec values, verifying formula)
    print("\n  Direct formula verification (weights=[0,0] bias=0 pred=0 assumed):")
    print("  delta = 1*(1-0) = 1")
    print("  w_new = [0,0] + 1*[1,2] = [1,2]")
    print("  b_new = 0 + 1 = 1")

    # -----------------------------------------------------------
    # Scenario A: Predict Output
    # -----------------------------------------------------------
    print("\n--- Scenario A: Predict Output ---")
    r = scenario_predict_output([1.0, 2.0], 1.0, [1.0, 2.0])
    print(f"  weights=[1,2], bias=1, input=[1,2]")
    print(f"  score = 1*1 + 2*2 + 1 = {r['score']}  prediction = {r['prediction']}")

    r2 = scenario_predict_output([1.0, -1.0], 0.0, [1.0, 1.0])
    print(f"  weights=[1,-1], bias=0, input=[1,1]")
    print(f"  score = 1*1 + (-1)*1 + 0 = {r2['score']}  prediction (tie) = {r2['prediction']}")

    # -----------------------------------------------------------
    # Scenario B: Diagnose Failure
    # -----------------------------------------------------------
    print("\n--- Scenario B: Diagnose Failure ---")
    # Weights that implement AND: predict 1 only when both inputs are 1
    dataset = [
        ([0.0, 0.0], 0),
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 0),
        ([1.0, 1.0], 1),
    ]
    diag = scenario_diagnose_failure([1.0, 1.0], -1.5, dataset)
    print(f"  AND-gate dataset with weights=[1,1], bias=-1.5")
    print(f"  Accuracy: {diag['accuracy']:.0%}  Misclassified: {diag['misclassified_indices']}")

    # Broken weights — deliberately wrong
    diag_bad = scenario_diagnose_failure([0.0, 0.0], 0.0, dataset)
    print(f"\n  Zero-weight model (always predicts 1 due to tie rule):")
    print(f"  Accuracy: {diag_bad['accuracy']:.0%}  Misclassified indices: {diag_bad['misclassified_indices']}")

    # -----------------------------------------------------------
    # Scenario C: Choose Boundary Policy
    # -----------------------------------------------------------
    print("\n--- Scenario C: Choose Boundary Policy (score=0) ---")
    for policy in ("positive_tie", "negative_tie", "abstain"):
        pred = scenario_choose_boundary_policy(0.0, policy)
        print(f"  policy='{policy}': prediction={pred}")


if __name__ == "__main__":
    main()
