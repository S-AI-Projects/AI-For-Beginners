"""
Segmentation-Mask Overlap — Refactor the Teaching Example
Applied lab for lessons/4-ComputerVision/12-Segmentation (Task R38-T016)

Concept & Contract:
In image semantic segmentation evaluation, the overlap between predicted masks and ground-truth
masks is evaluated using Intersection over Union (IoU / Jaccard Index):
    IoU(c) = |A_c ∩ B_c| / |A_c ∪ B_c|

Contract:
- Calculate per-class IoU from two small integer label masks.
- Explicit policy for classes absent from both ground truth and prediction: return None.
- Standard fixture:
    truth = [0, 1, 1, 0]
    prediction = [0, 1, 0, 0]
  Outcomes:
    - class 1: intersection = 1, union = 2 -> IoU = 1/2 = 0.5
    - class 0: intersection = 2, union = 3 -> IoU = 2/3 ≈ 0.6666667
    - absent classes (e.g. class 2): absent from both -> IoU = None

Focus:
Implement a correct monolithic baseline (`calculate_iou_baseline`), then refactor
by extracting the single responsibility of set-overlap calculation into a dedicated
component (`ConfusionMatrixEvaluator` / `calculate_binary_iou`).
Prove strict numerical before/after equivalence on all fixtures.
"""

from typing import Dict, List, Optional, Set, Tuple


# Baseline monolithic implementation
def calculate_iou_baseline(
    truth: List[int],
    prediction: List[int],
    target_classes: Optional[Set[int]] = None,
) -> Dict[int, Optional[float]]:
    """Monolithic baseline calculating per-class IoU."""
    if len(truth) != len(prediction):
        raise ValueError(f"Mask length mismatch: truth={len(truth)}, pred={len(prediction)}")
    if not truth:
        raise ValueError("Masks cannot be empty.")

    all_classes = set(truth) | set(prediction)
    if target_classes is not None:
        all_classes = all_classes | set(target_classes)

    iou_results: Dict[int, Optional[float]] = {}
    for c in sorted(all_classes):
        intersection = sum(1 for t, p in zip(truth, prediction) if t == c and p == c)
        union = sum(1 for t, p in zip(truth, prediction) if t == c or p == c)

        if union == 0:
            iou_results[c] = None  # Explicit absent-class policy
        else:
            iou_results[c] = intersection / union

    return iou_results


# Refactored Architecture: Extracted Responsibility
class BinaryOverlapMetrics:
    """Encapsulates isolated single responsibility of binary set overlap calculations."""

    @staticmethod
    def compute_iou(intersection: int, union: int) -> Optional[float]:
        """Compute IoU given counts, returning None if class is absent (union == 0)."""
        if union == 0:
            return None
        return intersection / union


class SegmentationEvaluator:
    """Evaluates multi-class segmentation masks by delegating overlap computation."""

    def __init__(self, metrics_calculator=BinaryOverlapMetrics):
        self.metrics = metrics_calculator

    def evaluate(
        self,
        truth: List[int],
        prediction: List[int],
        target_classes: Optional[Set[int]] = None,
    ) -> Dict[int, Optional[float]]:
        if len(truth) != len(prediction):
            raise ValueError(f"Mask length mismatch: truth={len(truth)}, pred={len(prediction)}")
        if not truth:
            raise ValueError("Masks cannot be empty.")

        classes = set(truth) | set(prediction)
        if target_classes is not None:
            classes = classes | set(target_classes)

        results: Dict[int, Optional[float]] = {}
        for c in sorted(classes):
            intersection = sum(1 for t, p in zip(truth, prediction) if t == c and p == c)
            union = sum(1 for t, p in zip(truth, prediction) if t == c or p == c)
            results[c] = self.metrics.compute_iou(intersection, union)

        return results


def run_demonstration():
    truth = [0, 1, 1, 0]
    prediction = [0, 1, 0, 0]

    baseline_res = calculate_iou_baseline(truth, prediction, target_classes={0, 1, 2})
    evaluator = SegmentationEvaluator()
    refactored_res = evaluator.evaluate(truth, prediction, target_classes={0, 1, 2})

    print("=== Segmentation Mask Overlap & Refactor Demo (R38-T016) ===")
    print(f"Ground Truth: {truth}")
    print(f"Prediction:   {prediction}")
    print(f"Baseline Results:   {baseline_res}")
    print(f"Refactored Results: {refactored_res}")
    print(f"Equivalence Confirmed: {baseline_res == refactored_res}")
    print(f"Class 1 IoU: {refactored_res[1]:.4f} (expected 0.5000 = 1/2)")
    print(f"Class 0 IoU: {refactored_res[0]:.4f} (expected 0.6667 = 2/3)")
    print(f"Class 2 IoU: {refactored_res[2]} (expected None - absent class policy)")


if __name__ == "__main__":
    run_demonstration()
