"""
R38-T058: Single perceptron update — failure and recovery example
Extend “Introduction to Neural Networks: Perceptron” at lessons/3-NeuralNetworks/03-Perceptron/README.md.
Implement one perceptron prediction/update step on a 2D fixture with a documented zero-score tie rule.
Weights [0,0], bias 0, input [1,2], target 1, initial prediction 0: learning-rate 1 update gives weights [1,2], bias 1.
Focus: Exercise a documented invalid/missing-data case, explain the error, and demonstrate recovery on the next valid input.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

@dataclass
class PerceptronState:
    weights: Tuple[float, float]
    bias: float
    learning_rate: float = 1.0

def predict_score(state: PerceptronState, x: Sequence[Union[int, float]]) -> Tuple[float, int]:
    """
    Computes linear score and step activation.
    Documented zero-score tie rule:
    - score > 0 -> prediction = 1
    - score <= 0 -> prediction = 0 (tie-breaking: score == 0 produces 0)
    """
    if len(x) != 2:
        raise ValueError(f"Expected 2-dimensional feature input, got length {len(x)}")
    for val in x:
        if val is None or not isinstance(val, (int, float)):
            raise ValueError(f"Invalid non-numeric feature value: {val}")

    score = state.weights[0] * x[0] + state.weights[1] * x[1] + state.bias
    prediction = 1 if score > 0.0 else 0
    return score, prediction

def train_step(
    state: PerceptronState,
    x: Sequence[Union[int, float]],
    target: int
) -> Tuple[PerceptronState, int, int]:
    """
    Performs one perceptron update step.
    Returns:
    - (new_state, initial_prediction, error)
    """
    if target not in (0, 1):
        raise ValueError(f"Target must be binary 0 or 1, got {target}")

    _, prediction = predict_score(state, x)
    error = target - prediction

    new_w0 = state.weights[0] + state.learning_rate * error * x[0]
    new_w1 = state.weights[1] + state.learning_rate * error * x[1]
    new_b = state.bias + state.learning_rate * error

    new_state = PerceptronState(
        weights=(new_w0, new_w1),
        bias=new_b,
        learning_rate=state.learning_rate
    )
    return new_state, prediction, error

def main():
    state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
    new_state, pred, err = train_step(state, [1, 2], 1)
    print(f"Updated weights: {new_state.weights}, bias: {new_state.bias}, initial pred: {pred}, error: {err}")

if __name__ == "__main__":
    main()
