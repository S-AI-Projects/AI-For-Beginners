"""
Unit tests for Segmentation Mask Overlap Failure and Recovery (Task R38-T018)
"""

import unittest

from example import SegmentationPipeline


class TestSegmentationOverlapContract(unittest.TestCase):
    """Verify standard contract and absent-class policy."""

    def setUp(self):
        self.pipeline = SegmentationPipeline()
        self.truth = [0, 1, 1, 0]
        self.pred = [0, 1, 0, 0]

    def test_standard_contract_values(self):
        """truth [0,1,1,0], prediction [0,1,0,0]: class1 IoU=1/2, class0 IoU=2/3."""
        res = self.pipeline.evaluate_mask_pair(self.truth, self.pred)
        self.assertAlmostEqual(res[1], 1.0 / 2.0, places=7)
        self.assertAlmostEqual(res[0], 2.0 / 3.0, places=7)

    def test_absent_class_returns_none(self):
        """Classes absent from both truth and prediction yield None."""
        res = self.pipeline.evaluate_mask_pair(self.truth, self.pred, target_classes={0, 1, 5})
        self.assertIsNone(res[5])


class TestFailureAndRecovery(unittest.TestCase):
    """Verify failure diagnostics and immediate recovery on next input."""

    def setUp(self):
        self.pipeline = SegmentationPipeline()

    def test_failure_on_dimension_mismatch(self):
        """Dimension mismatch raises ValueError with diagnostic message."""
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.evaluate_mask_pair([0, 1, 1], [0, 1, 0, 0])
        self.assertIn("Dimension mismatch", str(ctx.exception))
        self.assertIsNotNone(self.pipeline.last_error)

    def test_recovery_on_subsequent_valid_input(self):
        """Demonstrate that pipeline recovers cleanly immediately after a failure."""
        # 1. Trigger failure
        with self.assertRaises(ValueError):
            self.pipeline.evaluate_mask_pair([0, 1], [0, 1, 0])

        # 2. Immediate valid input recovers completely
        recovered_res = self.pipeline.evaluate_mask_pair([0, 1, 1, 0], [0, 1, 0, 0])
        self.assertAlmostEqual(recovered_res[1], 0.5, places=7)
        self.assertAlmostEqual(recovered_res[0], 2.0 / 3.0, places=7)
        self.assertIsNone(self.pipeline.last_error)

    def test_deliberate_wrong_result_fails(self):
        """Deliberate wrong expectation triggers test failure."""
        res = self.pipeline.evaluate_mask_pair([0, 1, 1, 0], [0, 1, 0, 0])
        with self.assertRaises(AssertionError):
            self.assertEqual(res[1], 0.99)


if __name__ == "__main__":
    unittest.main()
