"""
Unit tests for tensor-shape contract & normalization (Issue #21 / R38-T021).
"""

import unittest
from example import get_shape, validate_and_normalize_tensor


class TestTensorShapeContract(unittest.TestCase):
    def test_shape_preservation_and_normalization(self):
        """Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0 -> 0.0 and 255 -> 1.0."""
        input_tensor = [[[0, 255], [128, 64]]]
        self.assertEqual(get_shape(input_tensor), (1, 2, 2))

        output_tensor = validate_and_normalize_tensor(input_tensor)
        self.assertEqual(get_shape(output_tensor), (1, 2, 2))

        # Check exact values and mappings
        self.assertEqual(output_tensor[0][0][0], 0.0)
        self.assertEqual(output_tensor[0][0][1], 1.0)
        self.assertAlmostEqual(output_tensor[0][1][0], 128 / 255.0, places=6)
        self.assertAlmostEqual(output_tensor[0][1][1], 64 / 255.0, places=6)

    def test_multi_batch_preservation(self):
        """Test with batch size N > 1 with rectangular dimensions (2, 3, 2)."""
        input_tensor = [
            [[0, 255], [100, 200], [50, 150]],
            [[255, 0], [10, 20], [30, 40]]
        ]
        self.assertEqual(get_shape(input_tensor), (2, 3, 2))
        normalized = validate_and_normalize_tensor(input_tensor)
        self.assertEqual(get_shape(normalized), (2, 3, 2))
        self.assertEqual(normalized[0][0][0], 0.0)
        self.assertEqual(normalized[0][0][1], 1.0)
        self.assertEqual(normalized[1][0][0], 1.0)
        self.assertEqual(normalized[1][0][1], 0.0)

    def test_rejection_negative_pixels(self):
        """Reject pixels < 0."""
        tensor = [[[0, -1], [128, 64]]]
        with self.assertRaises(ValueError) as ctx:
            validate_and_normalize_tensor(tensor)
        self.assertIn("out of range", str(ctx.exception))

    def test_rejection_greater_than_255_pixels(self):
        """Reject pixels > 255."""
        tensor = [[[0, 256], [128, 64]]]
        with self.assertRaises(ValueError) as ctx:
            validate_and_normalize_tensor(tensor)
        self.assertIn("out of range", str(ctx.exception))

    def test_rejection_non_numeric_values(self):
        """Reject non-numeric pixel values (strings, booleans, None, etc.)."""
        # String
        tensor_str = [[[0, "255"], [128, 64]]]
        with self.assertRaises(TypeError):
            validate_and_normalize_tensor(tensor_str)

        # Boolean (bool is subclass of int in Python, must be caught explicitly)
        tensor_bool = [[[0, True], [128, 64]]]
        with self.assertRaises(TypeError):
            validate_and_normalize_tensor(tensor_bool)

        # None
        tensor_none = [[[None, 255], [128, 64]]]
        with self.assertRaises(TypeError):
            validate_and_normalize_tensor(tensor_none)

    def test_rejection_non_finite_float(self):
        """Reject float('nan') and float('inf')."""
        tensor_nan = [[[0, float('nan')], [128, 64]]]
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor(tensor_nan)

        tensor_inf = [[[0, float('inf')], [128, 64]]]
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor(tensor_inf)

    def test_rejection_inconsistent_dimensions_ragged(self):
        """Reject ragged/inconsistent tensor dimensions."""
        # Row length mismatch
        ragged_rows = [[[0, 255], [128]]]
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor(ragged_rows)

        # Height mismatch between batch items
        ragged_batch = [
            [[0, 255], [128, 64]],
            [[0, 255]]
        ]
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor(ragged_batch)

    def test_rejection_invalid_rank(self):
        """Reject tensors that are not 3D (N x H x W)."""
        # 2D tensor
        tensor_2d = [[0, 255], [128, 64]]
        with self.assertRaises(ValueError) as ctx:
            validate_and_normalize_tensor(tensor_2d)
        self.assertIn("Expected 3D tensor", str(ctx.exception))

        # 4D tensor
        tensor_4d = [[[[0, 255], [128, 64]]]]
        with self.assertRaises(ValueError) as ctx:
            validate_and_normalize_tensor(tensor_4d)
        self.assertIn("Expected 3D tensor", str(ctx.exception))

        # Empty 1D tensor
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor([])

    def test_rejection_non_sequence_input(self):
        """Reject top-level non-sequence input."""
        with self.assertRaises(TypeError):
            validate_and_normalize_tensor(42)  # type: ignore

    def test_rejection_zero_dimension(self):
        """Reject empty dimensions like (1, 0, 0)."""
        with self.assertRaises(ValueError):
            validate_and_normalize_tensor([[]])


if __name__ == "__main__":
    unittest.main()
