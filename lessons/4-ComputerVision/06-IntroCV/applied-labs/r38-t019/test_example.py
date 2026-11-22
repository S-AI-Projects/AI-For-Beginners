"""
Unit tests for Tensor-Shape Contract Evaluation Experiment (Task R38-T019)
"""

import unittest

from example import (
    Batch3D,
    get_shape,
    normalize_image_batch,
    run_evaluation_experiment,
)


class TestTensorShapeContract(unittest.TestCase):
    """Verify primary acceptance criteria and contract specifications."""

    def test_contract_batch_shape_and_normalization(self):
        """Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = normalize_image_batch(batch)

        self.assertEqual(get_shape(batch), (1, 2, 2))
        self.assertEqual(get_shape(out), (1, 2, 2))
        self.assertEqual(out[0][0][0], 0.0)
        self.assertEqual(out[0][0][1], 1.0)
        self.assertAlmostEqual(out[0][1][0], 128.0 / 255.0, places=7)
        self.assertAlmostEqual(out[0][1][1], 64.0 / 255.0, places=7)

    def test_negative_pixel_rejected(self):
        """Negative pixels (< 0) are rejected with ValueError."""
        with self.assertRaises(ValueError):
            normalize_image_batch([[[-1, 255], [128, 64]]])

    def test_pixel_greater_than_255_rejected(self):
        """Pixels > 255 are rejected with ValueError."""
        with self.assertRaises(ValueError):
            normalize_image_batch([[[0, 256], [128, 64]]])

    def test_ragged_batch_rejected(self):
        """Non-uniform tensor structures raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_image_batch([[[0, 255]], [[128, 64, 32]]])

    def test_evaluation_experiment_execution(self):
        """Verify evaluation experiment compares three configurations."""
        reports = run_evaluation_experiment()
        self.assertEqual(len(reports), 3)
        for r in reports:
            self.assertTrue(r["shape_preserved"])
            self.assertGreaterEqual(r["normalized_min"], 0.0)
            self.assertLessEqual(r["normalized_max"], 1.0)

    def test_deliberate_wrong_result_fails(self):
        """Deliberate incorrect expectation triggers test failure."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = normalize_image_batch(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out[0][0][1], 255.0)


if __name__ == "__main__":
    unittest.main()
