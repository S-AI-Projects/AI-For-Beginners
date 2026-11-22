"""
Segmentation-Mask Overlap — Failure and Recovery Example
Applied lab for lessons/4-ComputerVision/12-Segmentation (Task R38-T018)

Concept & Contract:
In image semantic segmentation evaluation, per-class Intersection over Union (IoU):
    IoU(c) = |Truth_c ∩ Pred_c| / |Truth_c ∪ Pred_c|

Contract:
- Calculate per-class IoU from two small integer label masks.
- Explicit policy for classes absent from both ground truth and prediction: return None.
- Standard fixture:
    truth = [0, 1, 1, 0]
    prediction = [0, 1, 0, 0]
  Outcomes:
    - class 1: intersection = 1, union = 2 -> IoU = 1/2 = 0.5
    - class 0: intersection = 2, union = 3 -> IoU = 2/3 ≈ 0.6666667
    - absent classes: absent from both -> IoU = None

Focus:
Exercise a documented invalid/missing-data case (dimension mismatch, non-integer or None in mask),
explain the error, and demonstrate complete recovery on the next valid input.
"""

from typing import Dict, List, Optional, Set, Tuple, Union


class SegmentationPipeline:
    """Evaluates semantic segmentation masks with robust error diagnostics and recovery."""

    def __init__(self):
        self.last_error: Optional[str] = None

    def evaluate_mask_pair(
        self,
        truth: List[int],
        prediction: List[int],
        target_classes: Optional[Set[int]] = None,
    ) -> Dict[int, Optional[float]]:
        """
        Evaluate per-class IoU. Validates inputs, clearing any previous error state.
        Raises ValueError with explanation on invalid input.
        """
        self.last_error = None

        if truth is None or prediction is None:
            self.last_error = "Input mask cannot be None."
            raise ValueError(self.last_error)

        if not truth or not prediction:
            self.last_error = "Masks must not be empty."
            raise ValueError(self.last_error)

        if len(truth) != len(prediction):
            self.last_error = (
                f"Dimension mismatch: truth length ({len(truth)}) != "
                f"prediction length ({len(prediction)})."
            )
            raise ValueError(self.last_error)

        for idx, (t, p) in enumerate(zip(truth, prediction)):
            if not isinstance(t, int) or not isinstance(p, int):
                self.last_error = f"Non-integer label encountered at index {idx}: truth={t}, pred={p}"
                raise TypeError(self.last_error)

        all_classes = set(truth) | set(prediction)
        if target_classes is not None:
            all_classes = all_classes | set(target_classes)

        results: Dict[int, Optional[float]] = {}
        for c in sorted(all_classes):
            intersection = sum(1 for t, p in zip(truth, prediction) if t == c and p == c)
            union = sum(1 for t, p in zip(truth, prediction) if t == c or p == c)

            if union == 0:
                results[c] = None  # Explicit absent-class policy
            else:
                results[c] = intersection / union

        return results

    def safe_process_stream(
        self, stream: List[Tuple[str, List[int], List[int]]]
    ) -> List[Dict[str, Union[str, Dict[int, Optional[float]]]]]:
        """Process a stream of mask pairs, demonstrating failure handling and recovery."""
        outcomes = []
        for name, truth, pred in stream:
            try:
                res = self.evaluate_mask_pair(truth, pred)
                outcomes.append({"name": name, "status": "SUCCESS", "result": res})
            except (ValueError, TypeError) as err:
                outcomes.append({"name": name, "status": "ERROR", "error": str(err)})
        return outcomes


def run_demonstration():
    pipeline = SegmentationPipeline()

    stream = [
        ("Valid Standard", [0, 1, 1, 0], [0, 1, 0, 0]),
        ("Invalid Mismatch", [0, 1, 1], [0, 1, 0, 0]),  # Length 3 vs 4
        ("Recovered Valid", [0, 1, 1, 0], [0, 1, 0, 0]),
    ]

    print("=== Segmentation Mask Overlap: Failure and Recovery Demo (R38-T018) ===")
    results = pipeline.safe_process_stream(stream)
    for r in results:
        print(f"[{r['name']}] Status: {r['status']}")
        if r["status"] == "SUCCESS":
            print(f"  Result: {r['result']}")
        else:
            print(f"  Handled Error: {r['error']}")


if __name__ == "__main__":
    run_demonstration()
