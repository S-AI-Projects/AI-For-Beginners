"""Binary Classification Evaluation — Worked Example (R38-T044).

TEACHING EXAMPLE: This is a local educational demonstration of binary
classification evaluation metrics using fixed label fixtures. It does not
train a model, perform real inference, or evaluate a production classifier.
It uses only the Python standard library.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# ── Standard fixtures ────────────────────────────────────────────────
MAIN_Y_TRUE: list[int] = [1, 1, 0, 0]
MAIN_Y_PRED: list[int] = [1, 0, 1, 0]

ZERO_DENOM_Y_TRUE: list[int] = [1, 1, 0, 0]
ZERO_DENOM_Y_PRED: list[int] = [0, 0, 0, 0]


@dataclass(frozen=True)
class ClassificationResult:
    """Structured result of a binary classification evaluation."""

    tp: int
    fn: int
    fp: int
    tn: int
    precision: float
    recall: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dictionary representation."""
        return asdict(self)


# ── Core evaluation function ─────────────────────────────────────────

def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ClassificationResult:
    """Evaluate a binary classifier by comparing true and predicted labels.

    Positive class = 1, negative class = 0.

    Zero-denominator policy:
        If TP + FP == 0 (no positive predictions), precision = 0.0.
        If TP + FN == 0 (no actual positives),     recall    = 0.0.
        This avoids division-by-zero while clearly indicating that the
        metric is undefined / degenerate for the given inputs.

    Args:
        y_true: Sequence of actual binary labels (0 or 1).
        y_pred: Sequence of predicted binary labels (0 or 1).

    Returns:
        A ClassificationResult with confusion-matrix counts, precision
        and recall.

    Raises:
        TypeError:  If y_true or y_pred is not a sequence, or contains
                    non-integer elements.
        ValueError: If the two sequences have different lengths, are
                    empty, or contain values other than 0 and 1.
    """
    # ── Input validation ─────────────────────────────────────────────
    if not isinstance(y_true, (list, tuple)):
        raise TypeError(
            f"y_true must be a list or tuple, got {type(y_true).__name__}"
        )
    if not isinstance(y_pred, (list, tuple)):
        raise TypeError(
            f"y_pred must be a list or tuple, got {type(y_pred).__name__}"
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length; "
            f"got {len(y_true)} vs {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if not isinstance(t, int) or isinstance(t, bool):
            raise TypeError(
                f"y_true[{i}] must be an int (0 or 1), got {type(t).__name__}"
            )
        if not isinstance(p, int) or isinstance(p, bool):
            raise TypeError(
                f"y_pred[{i}] must be an int (0 or 1), got {type(p).__name__}"
            )
        if t not in (0, 1):
            raise ValueError(
                f"y_true[{i}] must be 0 or 1, got {t}"
            )
        if p not in (0, 1):
            raise ValueError(
                f"y_pred[{i}] must be 0 or 1, got {p}"
            )

    # ── Confusion matrix ─────────────────────────────────────────────
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:  # actual == 0 and predicted == 0
            tn += 1

    # ── Metrics with zero-denominator policy ─────────────────────────
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        precision_recall_denom_zero = True
        recall = 0.0
    else:
        precision_recall_denom_zero = False
        recall = tp / (tp + fn)

    return ClassificationResult(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        precision=precision,
        recall=recall,
        n_samples=len(y_true),
    )


# ── CLI rendering ────────────────────────────────────────────────────

def render_result(
    label: str,
    y_true: list[int],
    y_pred: list[int],
    result: ClassificationResult,
) -> str:
    """Format a human-readable evaluation report."""
    lines = [
        "=" * 60,
        f"Fixture: {label}",
        f"y_true = {y_true}",
        f"y_pred = {y_pred}",
        "-" * 60,
        f"TP = {result.tp}",
        f"FN = {result.fn}",
        f"FP = {result.fp}",
        f"TN = {result.tn}",
        f"Precision = {result.precision}",
        f"Recall = {result.recall}",
        f"Total samples = {result.n_samples}",
        f"Confusion-matrix sum check: "
        f"{result.tp + result.fn + result.fp + result.tn} "
        f"== {result.n_samples}",
        "=" * 60,
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """CLI entrypoint: run both standard fixtures."""
    # Main fixture
    res_main = evaluate_classification(MAIN_Y_TRUE, MAIN_Y_PRED)
    print(render_result("Main", MAIN_Y_TRUE, MAIN_Y_PRED, res_main))
    print()

    # Zero-denominator fixture
    res_zero = evaluate_classification(ZERO_DENOM_Y_TRUE, ZERO_DENOM_Y_PRED)
    print(render_result(
        "No positive predictions (zero-denominator)",
        ZERO_DENOM_Y_TRUE,
        ZERO_DENOM_Y_PRED,
        res_zero,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

