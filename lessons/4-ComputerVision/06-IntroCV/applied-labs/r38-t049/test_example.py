"""Unit tests for Tensor-Shape Contract — Runnable Concept Example (R38-T049)."""

from __future__ import annotations

import unittest

from example import (
    MAIN_BATCH,
    NormalizedBatchResult,
    get_batch_shape,
    normalize_image_batch,
)


class TestRequiredFixture(unittest.TestCase):
    """Test the mandatory fixture specified in the assignment contract."""

    def test_mandatory_fixture_values_and_shape(self) -> None:
        """Verify [[[0, 255], [128, 64]]] normalizes to [[[0.0, 1.0], [128/255, 64/255]]] with shape (1, 2, 2)."""
        input_copy = [
            [
                [0, 255],
                [128, 64],
            ]
        ]
        res: NormalizedBatchResult = normalize_image_batch(input_copy)

        # Shape verification
        self.assertEqual(res.original_shape, (1, 2, 2))
        self.assertEqual(res.normalized_shape, (1, 2, 2))
        self.assertTrue(res.shape_preserved)

        # Value verification
        norm_pixels = res.normalized_batch[0]
        self.assertAlmostEqual(norm_pixels[0][0], 0.0, places=9)
        self.assertAlmostEqual(norm_pixels[0][1], 1.0, places=9)
        self.assertAlmostEqual(norm_pixels[1][0], 128.0 / 255.0, places=9)
        self.assertAlmostEqual(norm_pixels[1][1], 64.0 / 255.0, places=9)

    def test_immutability_of_original_input(self) -> None:
        """Normalization must not mutate the original nested lists in-place."""
        original = [
            [
                [0, 255],
                [128, 64],
            ]
        ]
        _ = normalize_image_batch(original)
        self.assertEqual(original, [[[0, 255], [128, 64]]])


class TestShapePreservation(unittest.TestCase):
    """Test shape preservation across diverse spatial and batch dimensions."""

    def test_single_image_shape(self) -> None:
        batch = [[[10, 20], [30, 40], [50, 60]]]
        res = normalize_image_batch(batch)
        self.assertEqual(res.original_shape, (1, 3, 2))
        self.assertEqual(res.normalized_shape, (1, 3, 2))
        self.assertTrue(res.shape_preserved)

    def test_multi_image_batch_shape(self) -> None:
        batch = [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
            [[120, 130, 140], [150, 160, 170]],
        ]
        res = normalize_image_batch(batch)
        self.assertEqual(res.original_shape, (3, 2, 3))
        self.assertEqual(res.normalized_shape, (3, 2, 3))
        self.assertTrue(res.shape_preserved)

    def test_non_square_dimensions(self) -> None:
        # N=1, H=1, W=5
        batch = [[[10, 20, 30, 40, 50]]]
        res = normalize_image_batch(batch)
        self.assertEqual(res.original_shape, (1, 1, 5))
        self.assertEqual(res.normalized_shape, (1, 1, 5))

    def test_single_pixel_image(self) -> None:
        batch = [[[128]]]
        res = normalize_image_batch(batch)
        self.assertEqual(res.original_shape, (1, 1, 1))
        self.assertEqual(res.normalized_shape, (1, 1, 1))
        self.assertAlmostEqual(res.normalized_batch[0][0][0], 128.0 / 255.0, places=9)


class TestPixelValidation(unittest.TestCase):
    """Test validation of pixel ranges and types."""

    def test_negative_pixel_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[[-1, 100], [50, 50]]])

    def test_pixel_greater_than_255_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[[256, 100], [50, 50]]])

    def test_nan_pixel_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[[float("nan"), 100], [50, 50]]])

    def test_inf_pixel_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[[float("inf"), 100], [50, 50]]])
        with self.assertRaises(ValueError):
            normalize_image_batch([[[float("-inf"), 100], [50, 50]]])

    def test_boolean_pixel_rejected(self) -> None:
        """Booleans must be rejected even though bool is a subclass of int in Python."""
        with self.assertRaises(TypeError):
            normalize_image_batch([[[True, 100], [50, 50]]])
        with self.assertRaises(TypeError):
            normalize_image_batch([[[False, 100], [50, 50]]])

    def test_non_numeric_pixel_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_image_batch([[["white", 100], [50, 50]]])  # type: ignore


class TestStructureValidation(unittest.TestCase):
    """Test validation of batch hierarchy, rectangularity, and container types."""

    def test_empty_batch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([])

    def test_empty_image_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[]])

    def test_empty_row_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_image_batch([[[]]])

    def test_ragged_rows_within_image_rejected(self) -> None:
        # Row 0 has 2 pixels, Row 1 has 3 pixels
        with self.assertRaises(ValueError):
            normalize_image_batch([[[0, 100], [50, 100, 150]]])

    def test_inconsistent_image_heights_rejected(self) -> None:
        # Image 0 has 2 rows, Image 1 has 1 row
        with self.assertRaises(ValueError):
            normalize_image_batch([
                [[0, 100], [50, 100]],
                [[200, 255]],
            ])

    def test_non_sequence_batch_container_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_image_batch(12345)  # type: ignore
        with self.assertRaises(TypeError):
            normalize_image_batch("not a batch")  # type: ignore

    def test_non_sequence_image_container_rejected(self) -> None:
        with self.assertRaises(TypeError):
            normalize_image_batch([123])  # type: ignore


class TestBoundaryCases(unittest.TestCase):
    """Test boundary values: all zeros, all 255s, and floating-point pixels."""

    def test_all_zeros_normalized(self) -> None:
        batch = [[[0, 0], [0, 0]]]
        res = normalize_image_batch(batch)
        for row in res.normalized_batch[0]:
            for p in row:
                self.assertEqual(p, 0.0)

    def test_all_255_normalized(self) -> None:
        batch = [[[255, 255], [255, 255]]]
        res = normalize_image_batch(batch)
        for row in res.normalized_batch[0]:
            for p in row:
                self.assertEqual(p, 1.0)

    def test_mixed_integer_and_float_pixels(self) -> None:
        batch = [[[0, 127.5], [255.0, 63.75]]]
        res = normalize_image_batch(batch)
        self.assertAlmostEqual(res.normalized_batch[0][0][1], 127.5 / 255.0, places=9)
        self.assertAlmostEqual(res.normalized_batch[0][1][0], 1.0, places=9)
        self.assertAlmostEqual(res.normalized_batch[0][1][1], 63.75 / 255.0, places=9)


if __name__ == "__main__":
    unittest.main()

