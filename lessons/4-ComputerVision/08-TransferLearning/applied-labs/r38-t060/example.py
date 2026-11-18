"""
R38-T060: Training-only imputation — scenario assessment
Extend “Pre-trained Networks and Transfer Learning” at lessons/4-ComputerVision/08-TransferLearning/README.md.
Fit a missing-value mean on training observations only, then apply the stored value to validation observations without refitting.
Training [2,None,4] yields mean 3; validation [100,None] becomes [100,3], not [100,100]. All-missing training input is rejected.
Focus: Add three applied questions about the local example: predict an output, diagnose a failure and choose a boundary policy. Provide a rationale and executable check for each factual answer.
"""
from typing import Sequence, Optional, List, Dict, Any

class TrainingOnlyMeanImputer:
    def __init__(self):
        self.fitted_mean: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, training_data: Sequence[Optional[float]]) -> "TrainingOnlyMeanImputer":
        if not training_data:
            raise ValueError("Training data cannot be empty")

        valid_vals = [v for v in training_data if v is not None]
        if not valid_vals:
            raise ValueError("All-missing training input is rejected: cannot compute mean")

        self.fitted_mean = sum(valid_vals) / len(valid_vals)
        self.is_fitted = True
        return self

    def transform(self, data: Sequence[Optional[float]]) -> List[float]:
        if not self.is_fitted or self.fitted_mean is None:
            raise RuntimeError("Imputer is not fitted. Call fit() first.")
        return [self.fitted_mean if v is None else float(v) for v in data]

    def fit_transform(self, training_data: Sequence[Optional[float]]) -> List[float]:
        return self.fit(training_data).transform(training_data)

SCENARIO_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "Q1_PREDICT_OUTPUT",
        "prompt": "Given training data [2, None, 4] and validation data [100, None], what is the imputed validation output?",
        "expected_answer": [100.0, 3.0],
        "rationale": "Mean is computed solely on training non-missing values: (2 + 4) / 2 = 3.0. Applying to validation replaces None with 3.0.",
    },
    {
        "id": "Q2_DIAGNOSE_FAILURE",
        "prompt": "What happens when training data is completely missing [None, None]?",
        "expected_exception": ValueError,
        "rationale": "An imputer must reject all-missing training observations with ValueError to prevent fabricating arbitrary fill values.",
    },
    {
        "id": "Q3_CHOOSE_BOUNDARY_POLICY",
        "prompt": "Why must the imputer NOT refit on validation data [100, None]?",
        "leakage_output": [100.0, 100.0],
        "rationale": "Refitting on validation data leaks validation distribution into the features (yielding mean 100.0), compromising evaluation validity.",
    }
]

def main():
    imputer = TrainingOnlyMeanImputer()
    imputer.fit([2, None, 4])
    print("Fitted mean:", imputer.fitted_mean)
    val_trans = imputer.transform([100, None])
    print("Validation transformed:", val_trans)

if __name__ == "__main__":
    main()
