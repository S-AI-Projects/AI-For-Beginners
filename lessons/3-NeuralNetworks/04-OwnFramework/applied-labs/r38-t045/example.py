"""Single Perceptron Update with Offline Service Adapter (R38-T045).

TEACHING LAB: This is a local educational demonstration of a single perceptron
forward-pass prediction and weight-update step using an offline service adapter.
It uses only the Python standard library.

It does NOT:
- Connect to any real cloud service or external API.
- Require network access, authentication tokens, or external dependencies.
- Train a complete multilayer neural network over multiple epochs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

# Type alias for a prediction dependency:
# Signature: predictor(weights, bias, inputs) -> binary prediction (0 or 1)
PredictionCallable = Callable[[Sequence[float], float, Sequence[float]], int]

# ── Standard Fixture ─────────────────────────────────────────────────
MAIN_INITIAL_WEIGHTS: list[float] = [0.0, 0.0]
MAIN_INITIAL_BIAS: float = 0.0
MAIN_INPUTS: list[float] = [1.0, 2.0]
MAIN_TARGET: int = 1
MAIN_LEARNING_RATE: float = 1.0


@dataclass(frozen=True)
class PerceptronStepResult:
    """Structured result of a single perceptron prediction and update step."""

    initial_weights: list[float]
    initial_bias: float
    inputs: list[float]
    target: int
    learning_rate: float
    weighted_sum: float
    initial_prediction: int
    error: int
    updated_weights: list[float]
    updated_bias: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return asdict(self)


# ── Mathematical Functions ───────────────────────────────────────────

def calculate_weighted_sum(
    weights: Sequence[float],
    bias: float,
    inputs: Sequence[float],
) -> float:
    """Calculate the linear combination z = sum(w_i * x_i) + bias.

    Args:
        weights: Sequence of synaptic weights [w_1, w_2, ..., w_n].
        bias: Additive bias scalar b.
        inputs: Sequence of input features [x_1, x_2, ..., x_n].

    Returns:
        The scalar weighted sum z.
    """
    return sum(float(w) * float(x) for w, x in zip(weights, inputs)) + float(bias)


# ── Offline Service Adapter ──────────────────────────────────────────

def local_deterministic_predictor(
    weights: Sequence[float],
    bias: float,
    inputs: Sequence[float],
) -> int:
    """Local, offline, deterministic adapter implementing the perceptron threshold.

    Decision Boundary & Zero-Score Tie Rule:
        z = sum(w_i * x_i) + bias
        prediction = 1 if z > 0 else 0

    IMPORTANT: A score of exactly zero (z == 0.0) produces prediction 0.
    This resolves boundary ties in favor of the negative class.

    This function serves as an offline mock/adapter for a prediction service.
    It executes entirely locally in-process without network access.

    Args:
        weights: Model weights.
        bias: Model bias.
        inputs: Input feature vector.

    Returns:
        Binary classification decision: 1 (positive) or 0 (negative).
    """
    z = calculate_weighted_sum(weights, bias, inputs)
    # Zero-score tie rule: strictly greater than 0 yields 1; zero or less yields 0.
    return 1 if z > 0.0 else 0


# ── Validation Helper ────────────────────────────────────────────────

def _validate_numeric(val: Any, name: str) -> float:
    """Ensure a value is an int or float, rejecting booleans, NaN, and Inf."""
    if isinstance(val, bool):
        raise TypeError(f"'{name}' must be numeric, got boolean {val!r}")
    if not isinstance(val, (int, float)):
        raise TypeError(f"'{name}' must be int or float, got {type(val).__name__}")
    f_val = float(val)
    if math.isnan(f_val) or math.isinf(f_val):
        raise ValueError(f"'{name}' cannot be NaN or Infinite: {val!r}")
    return f_val


def validate_perceptron_inputs(
    weights: Sequence[float],
    bias: float,
    inputs: Sequence[float],
    target: int,
    learning_rate: float,
    predictor: Any,
) -> None:
    """Validate all inputs for a single perceptron update step."""
    if not callable(predictor):
        raise TypeError(f"'predictor' must be callable, got {type(predictor).__name__}")

    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise TypeError(f"'weights' must be a Sequence, got {type(weights).__name__}")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError(f"'inputs' must be a Sequence, got {type(inputs).__name__}")

    if len(weights) == 0:
        raise ValueError("'weights' cannot be empty")
    if len(inputs) == 0:
        raise ValueError("'inputs' cannot be empty")
    if len(weights) != len(inputs):
        raise ValueError(
            f"Dimension mismatch: len(weights)={len(weights)} != len(inputs)={len(inputs)}"
        )

    for i, w in enumerate(weights):
        _validate_numeric(w, f"weights[{i}]")
    for i, x in enumerate(inputs):
        _validate_numeric(x, f"inputs[{i}]")

    _validate_numeric(bias, "bias")

    lr = _validate_numeric(learning_rate, "learning_rate")
    if lr < 0.0:
        raise ValueError(f"'learning_rate' must be non-negative, got {lr}")

    if isinstance(target, bool):
        raise TypeError(f"'target' cannot be a boolean, got {target!r}")
    if target not in (0, 1):
        raise ValueError(f"'target' must be binary (0 or 1), got {target!r}")


# ── Perceptron Update Step (with Dependency Injection) ───────────────

def perceptron_update_step(
    weights: Sequence[float],
    bias: float,
    inputs: Sequence[float],
    target: int,
    learning_rate: float,
    predictor: PredictionCallable,
) -> PerceptronStepResult:
    """Perform exactly one perceptron prediction and weight update step.

    This function does NOT hardcode an internal prediction engine. Instead,
    it receives a prediction callable via dependency injection ('predictor').

    Algorithm:
        1. Query the injected predictor for an initial prediction:
           initial_pred = predictor(weights, bias, inputs)
        2. Calculate the prediction error:
           error = target - initial_pred
        3. Update synaptic weights using the perceptron learning rule:
           w_new[i] = w_old[i] + learning_rate * error * x[i]
        4. Update bias:
           b_new = b_old + learning_rate * error

    Args:
        weights: Sequence of current weights.
        bias: Current scalar bias.
        inputs: Input feature values.
        target: True binary label (0 or 1).
        learning_rate: Learning rate scalar eta >= 0.
        predictor: Injected callable adhering to PredictionCallable.

    Returns:
        PerceptronStepResult containing initial and updated model state.
    """
    validate_perceptron_inputs(weights, bias, inputs, target, learning_rate, predictor)

    # 1. Query the injected callable for the prediction
    initial_prediction = predictor(weights, bias, inputs)

    # Validate the prediction returned by the injected callable
    if isinstance(initial_prediction, bool):
        raise TypeError(f"predictor returned boolean {initial_prediction!r}, expected int (0 or 1)")
    if initial_prediction not in (0, 1):
        raise ValueError(
            f"predictor returned non-binary prediction {initial_prediction!r}, expected 0 or 1"
        )

    # Compute weighted sum for documentation/inspection
    weighted_sum = calculate_weighted_sum(weights, bias, inputs)

    # 2. Compute error: error = target - prediction
    error = int(target) - int(initial_prediction)

    # 3. Update weights: w_new = w_old + eta * error * x
    updated_weights = [
        float(w) + float(learning_rate) * float(error) * float(x)
        for w, x in zip(weights, inputs)
    ]

    # 4. Update bias: b_new = b_old + eta * error
    updated_bias = float(bias) + float(learning_rate) * float(error)

    return PerceptronStepResult(
        initial_weights=[float(w) for w in weights],
        initial_bias=float(bias),
        inputs=[float(x) for x in inputs],
        target=int(target),
        learning_rate=float(learning_rate),
        weighted_sum=float(weighted_sum),
        initial_prediction=int(initial_prediction),
        error=int(error),
        updated_weights=updated_weights,
        updated_bias=updated_bias,
    )


# ── Pretty-Printing and CLI Runner ───────────────────────────────────

def render_result(result: PerceptronStepResult, title: str) -> None:
    """Format and print a step result cleanly to stdout."""
    print(f"--- {title} ---")
    print(f"Inputs (x):          {result.inputs}")
    print(f"Initial Weights (w): {result.initial_weights}")
    print(f"Initial Bias (b):    {result.initial_bias}")
    print(f"Learning Rate (eta): {result.learning_rate}")
    print(f"Target (y):          {result.target}")
    print()
    print("Calculations:")
    terms = " + ".join(
        f"({w} * {x})" for w, x in zip(result.initial_weights, result.inputs)
    )
    print(f"  Weighted sum (z) = {terms} + {result.initial_bias} = {result.weighted_sum}")
    if result.weighted_sum > 0.0:
        print(f"  Decision threshold: z = {result.weighted_sum} > 0 -> prediction = {result.initial_prediction}")
    else:
        print(f"  Zero-score tie:     z = {result.weighted_sum} <= 0 -> prediction = {result.initial_prediction}")
    print(f"  Error (y - y_hat)  = {result.target} - {result.initial_prediction} = {result.error}")
    print()
    print("Weight & Bias Updates:")
    for i, (w_old, x, w_new) in enumerate(
        zip(result.initial_weights, result.inputs, result.updated_weights)
    ):
        print(f"  w_{i+1}_new = {w_old} + ({result.learning_rate} * {result.error} * {x}) = {w_new}")
    print(f"  b_new   = {result.initial_bias} + ({result.learning_rate} * {result.error}) = {result.updated_bias}")
    print()
    print("Final State:")
    print(f"  Updated Weights:   {result.updated_weights}")
    print(f"  Updated Bias:      {result.updated_bias}")
    print()


def main() -> int:
    """Execute the standard perceptron update demonstration."""
    print("=" * 60)
    print("Single Perceptron Update — Offline Service Adapter (R38-T045)")
    print("=" * 60)
    print()

    # Scenario 1: Main required regression fixture
    main_result = perceptron_update_step(
        weights=MAIN_INITIAL_WEIGHTS,
        bias=MAIN_INITIAL_BIAS,
        inputs=MAIN_INPUTS,
        target=MAIN_TARGET,
        learning_rate=MAIN_LEARNING_RATE,
        predictor=local_deterministic_predictor,
    )
    render_result(main_result, "Scenario 1: Main Regression Fixture")

    # Scenario 2: Zero-score tie rule explicitly demonstrated
    zero_score_result = perceptron_update_step(
        weights=[0.0, 0.0],
        bias=0.0,
        inputs=[5.0, -2.0],
        target=0,
        learning_rate=1.0,
        predictor=local_deterministic_predictor,
    )
    render_result(zero_score_result, "Scenario 2: Zero-Score Tie Rule (Target = 0, Error = 0)")

    # Scenario 3: Positive score (z > 0 -> prediction = 1)
    pos_score_result = perceptron_update_step(
        weights=[1.0, 1.0],
        bias=0.0,
        inputs=[1.0, 1.0],
        target=1,
        learning_rate=0.5,
        predictor=local_deterministic_predictor,
    )
    render_result(pos_score_result, "Scenario 3: Positive Score (z=2 > 0 -> prediction=1, Error=0)")

    # Scenario 4: Negative score (z < 0 -> prediction = 0)
    neg_score_result = perceptron_update_step(
        weights=[-1.0, -1.0],
        bias=0.0,
        inputs=[2.0, 1.0],
        target=1,
        learning_rate=1.0,
        predictor=local_deterministic_predictor,
    )
    render_result(neg_score_result, "Scenario 4: Negative Score (z=-3 < 0 -> prediction=0, Error=1)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
