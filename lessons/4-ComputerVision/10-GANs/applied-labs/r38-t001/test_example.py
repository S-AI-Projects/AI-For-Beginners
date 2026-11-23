"""
Unit tests for Applied Lab R38-T001: tensor-shape contract & normalization.
"""

import unittest
from example import get_batch_shape, validate_and_normalize_batch


class TestTensorShapeContract(unittest.TestCase):

    def test_canonical_fixture_shape_and_values(self):
        """
        Verify canonical fixture: [[[0, 255], [128, 64]]] stays shape (1, 2, 2)
        0 -> 0.0 and 255 -> 1.0
        """
        batch = [[[0, 255], [128, 64]]]
        self.assertEqual(get_batch_shape(batch), (1, 2, 2))
        
        result = validate_and_normalize_batch(batch)
        self.assertEqual(get_batch_shape(result), (1, 2, 2))
        
        # Check normalization bounds and values
        self.assertAlmostEqual(result[0][0][0], 0.0, places=6)
        self.assertAlmostEqual(result[0][0][1], 1.0, places=6)
        self.assertAlmostEqual(result[0][1][0], 128 / 255.0, places=6)
        self.assertAlmostEqual(result[0][1][1], 64 / 255.0, places=6)

    def test_negative_pixel_rejected(self):
        """Negative pixels must be rejected with ValueError."""
        batch = [[[0, -1], [128, 64]]]
        with self.assertRaises(ValueError):
            validate_and_normalize_batch(batch)

    def test_exceeding_pixel_rejected(self):
        """Pixels > 255 must be rejected with ValueError."""
        batch = [[[0, 256], [128, 64]]]
        with self.assertRaises(ValueError):
            validate_and_normalize_batch(batch)

    def test_boundary_pixels(self):
        """Boundary values 0 and 255 are strictly valid."""
        batch = [[[0, 0], [255, 255]]]
        result = validate_and_normalize_batch(batch)
        self.assertEqual(result, [[[0.0, 0.0], [1.0, 1.0]]])

    def test_multi_sample_batch_shape_preservation(self):
        """Ensure multi-item batch preserves shape (2, 2, 3)."""
        batch = [
            [[0, 100, 200], [50, 150, 250]],
            [[10, 20, 30], [40, 50, 60]]
        ]
        self.assertEqual(get_batch_shape(batch), (2, 2, 3))
        res = validate_and_normalize_batch(batch)
        self.assertEqual(get_batch_shape(res), (2, 2, 3))

    def test_invalid_shapes(self):
        """Non-3D batches or jagged arrays raise appropriate errors."""
        with self.assertRaises(TypeError):
            validate_and_normalize_batch("not a list")
        with self.assertRaises(ValueError):
            validate_and_normalize_batch([])
        with self.assertRaises(ValueError):
            validate_and_normalize_batch([[1, 2], [3, 4]])  # 2D, not 3D
        with self.assertRaises(ValueError):
            # Jagged width
            validate_and_normalize_batch([[[1, 2], [3]]])

    def test_deliberate_wrong_result_fails(self):
        """Verify that a deliberate wrong result fails the assertion."""
        batch = [[[0, 255], [128, 64]]]
        result = validate_and_normalize_batch(batch)
        # Verify that asserting 255 normalized to 0.5 deliberately fails
        with self.assertRaises(AssertionError):
            self.assertEqual(result[0][0][1], 0.5)


if __name__ == "__main__":
    unittest.main()
