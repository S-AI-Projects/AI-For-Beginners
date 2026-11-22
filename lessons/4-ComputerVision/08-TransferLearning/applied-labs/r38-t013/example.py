"""
Training-Only Imputation — Compare Two Approaches
Applied lab for lessons/4-ComputerVision/08-TransferLearning (Task R38-T013)

Concept & Contract:
In transfer learning and data pipelines, pre-processing parameters (such as imputation
statistics, normalization means/std) must be learned ONLY from the training dataset.
Re-computing statistics on the validation or test set introduces data leakage, corrupting
evaluation metrics.

Contract:
- Fit missing-value mean on training observations only.
- Training [2, None, 4] yields mean 3.0.
- Transform validation [100, None] using fitted training mean -> [100, 3.0] (NOT [100, 100.0]).
- All-missing training input (e.g. [None, None]) must be rejected (raises ValueError).

Focus:
Compare two approaches:
Approach 1: Stateful Object-Oriented Imputer (`TrainingMeanImputer`)
Approach 2: Pure Functional Pipeline (`fit_mean` + `transform_with_mean`)
Verify exact mathematical equivalence across identical inputs.
"""

from typing import List, Optional, Tuple, Union

Number = Union[int, float]
Value = Optional[Number]


class TrainingMeanImputer:
    """Stateful transformer enforcing fit-transform separation to prevent data leakage."""

    def __init__(self):
        self.fitted_mean: Optional[float] = None

    def fit(self, train_data: List[Value]) -> "TrainingMeanImputer":
        """Compute mean of non-missing values in training data. Reject all-missing."""
        if not train_data:
            raise ValueError("Training data cannot be empty.")

        valid_vals = [x for x in train_data if x is not None]
        if not valid_vals:
            raise ValueError("All-missing training input is rejected.")

        self.fitted_mean = float(sum(valid_vals) / len(valid_vals))
        return self

    def transform(self, data: List[Value]) -> List[float]:
        """Impute missing values using the stored training mean."""
        if self.fitted_mean is None:
            raise RuntimeError("Imputer must be fitted before transform.")

        return [float(x) if x is not None else self.fitted_mean for x in data]


# Approach 2: Functional Pipeline
def fit_mean(train_data: List[Value]) -> float:
    """Fit missing-value mean on training observations only. Reject all-missing."""
    if not train_data:
        raise ValueError("Training data cannot be empty.")
    valid_vals = [x for x in train_data if x is not None]
    if not valid_vals:
        raise ValueError("All-missing training input is rejected.")
    return float(sum(valid_vals) / len(valid_vals))


def transform_with_mean(data: List[Value], fill_value: float) -> List[float]:
    """Apply stored mean without refitting."""
    return [float(x) if x is not None else fill_value for x in data]


def run_demonstration() -> Tuple[List[float], List[float]]:
    """Execute standard fixtures and display leakage comparison."""
    train_input: List[Value] = [2, None, 4]
    val_input: List[Value] = [100, None]

    imputer = TrainingMeanImputer()
    imputer.fit(train_input)
    train_transformed = imputer.transform(train_input)
    val_transformed = imputer.transform(val_input)

    # Functional approach
    mean_val = fit_mean(train_input)
    val_functional = transform_with_mean(val_input, mean_val)

    print("=== Training-Only Imputation Demo (R38-T013) ===")
    print(f"Training Input: {train_input}")
    print(f"Fitted Training Mean: {imputer.fitted_mean} (expected 3.0)")
    print(f"Transformed Training: {train_transformed}")
    print(f"Validation Input: {val_input}")
    print(f"Correct Transformed Validation (training mean applied): {val_transformed}")
    print(f"Functional Equivalence: {val_transformed == val_functional}")
    print("Anti-Pattern (Data Leakage): refitting validation would yield [100, 100.0] - AVOIDED!\n")

    return train_transformed, val_transformed


if __name__ == "__main__":
    run_demonstration()
