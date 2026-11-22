"""
Unit tests for Segmentation Mask Overlap Refactor (Task R38-T016)
"""

import unittest

from example import (
    BinaryOverlapMetrics,
    SegmentationEvaluator,
    calculate_iou_baseline,
)


class TestSegmentationOverlapContract(unittest.TestCase):
    """Verify primary acceptance criteria on contract fixture."""

    def setUp(self):
        self.truth = [0, 1, 1, 0]
        self.pred = [0, 1, 0, 0]
        self.evaluator = SegmentationEvaluator()

    def test_contract_fixture_class_ious(self):
        """truth [0,1,1,0], prediction [0,1,0,0]: class1 IoU=1/2, class0 IoU=2/3."""
        res = self.evaluator.evaluate(self.truth, self.pred)

        # Class 1: inter=1, union=2 -> 1/2
        self.assertAlmostEqual(res[1], 1.0 / 2.0, places=7)

        # Class 0: inter=2, union=3 -> 2/3
        self.assertAlmostEqual(res[0], 2.0 / 3.0, places=7)

    def test_absent_class_policy_returns_none(self):
        """Classes absent from both truth and prediction yield None."""
        res = self.evaluator.evaluate(self.truth, self.pred, target_classes={0, 1, 99})
        self.assertIsNone(res[99])

    def test_refactor_equivalence_with_baseline(self):
        """Verify before/after refactor equivalence."""
        baseline = calculate_iou_baseline(self.truth, self.pred, target_classes={0, 1, 2})
        refactored = self.evaluator.evaluate(self.truth, self.pred, target_classes={0, 1, 2})
        self.assertEqual(baseline, refactored)

    def test_dimension_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([0, 1], [0, 1, 0])

    def test_empty_mask_raises_error(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([], [])

    def test_deliberate_wrong_result_fails(self):
        """Deliberate incorrect expectation triggers test failure."""
        res = self.evaluator.evaluate(self.truth, self.pred)
        with self.assertRaises(AssertionError):
            self.assertEqual(res[1], 1.0)


if __name__ == "__main__":
    unittest.main()
