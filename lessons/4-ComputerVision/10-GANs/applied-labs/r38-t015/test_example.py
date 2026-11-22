"""
Unit tests for Tensor-Shape Contract Offline Adapter (Task R38-T015)
"""

import unittest

from example import (
    Batch3D,
    ImagePreprocessorService,
    LocalPreprocessorAdapter,
    get_shape,
)


class TestOfflineServiceAdapter(unittest.TestCase):
    """Verify primary acceptance criteria through callable interface."""

    def setUp(self):
        self.service: ImagePreprocessorService = LocalPreprocessorAdapter()

    def test_contract_batch_shape_and_scaling(self):
        """Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = self.service.preprocess_batch(batch)

        self.assertEqual(get_shape(batch), (1, 2, 2))
        self.assertEqual(get_shape(out), (1, 2, 2))
        self.assertEqual(out[0][0][0], 0.0)
        self.assertEqual(out[0][0][1], 1.0)
        self.assertAlmostEqual(out[0][1][0], 128.0 / 255.0, places=7)
        self.assertAlmostEqual(out[0][1][1], 64.0 / 255.0, places=7)

    def test_negative_pixel_rejected(self):
        """Negative pixels (< 0) are rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.service.preprocess_batch([[[-1, 255], [128, 64]]])

    def test_high_pixel_rejected(self):
        """Pixels exceeding 255 are rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.service.preprocess_batch([[[0, 256], [128, 64]]])

    def test_ragged_batch_rejected(self):
        """Non-rectangular arrays raise ValueError."""
        with self.assertRaises(ValueError):
            self.service.preprocess_batch([[[0, 255]], [[128]]])

    def test_deliberate_wrong_result_fails(self):
        """Deliberate wrong expectation triggers test failure."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = self.service.preprocess_batch(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out[0][0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
