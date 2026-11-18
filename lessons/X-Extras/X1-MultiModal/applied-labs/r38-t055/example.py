"""
R38-T055: Classification evaluation — offline service adapter
Extend “Multi-Modal Networks” at lessons/X-Extras/X1-MultiModal/README.md.
Compute a confusion matrix, precision and recall for a binary classifier using fixed predicted and true labels; make the zero-denominator policy explicit.
y_true=[1,1,0,0], y_pred=[1,0,1,0] gives TP=1,FN=1,FP=1,TN=1, precision=recall=0.5; empty or mismatched-length inputs are rejected.
"""
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class EvaluationMetrics:
    tp: int
    fn: int
    fp: int
    tn: int
    precision: float
    recall: float
    zero_denominator_policy: str = "Return 0.0 when denominator is 0"

def evaluate_binary_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int]
) -> EvaluationMetrics:
    """
    Compute binary classification confusion matrix, precision, and recall.

    Zero-denominator policy:
    - If TP + FP == 0, precision is defined as 0.0.
    - If TP + FN == 0, recall is defined as 0.0.

    Validation:
    - Empty inputs or mismatched lengths raise ValueError.
    - Non-binary values raise ValueError.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatched input lengths: {len(y_true)} vs {len(y_pred)}")

    for val in y_true:
        if val not in (0, 1):
            raise ValueError(f"Invalid value in y_true: {val}; must be 0 or 1")
    for val in y_pred:
        if val not in (0, 1):
            raise ValueError(f"Invalid value in y_pred: {val}; must be 0 or 1")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return EvaluationMetrics(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        precision=precision,
        recall=recall
    )

def main():
    m = evaluate_binary_predictions([1, 1, 0, 0], [1, 0, 1, 0])
    print(f"TP={m.tp}, FN={m.fn}, FP={m.fp}, TN={m.tn}, Precision={m.precision}, Recall={m.recall}")

if __name__ == "__main__":
    main()
