"""
Unit tests for Tensor-Shape Contract (Task R38-T014)
"""

import unittest

from example import (
    Batch3D,
    PreprocessingStateTracer,
    get_shape,
)


class TestTensorShapeContract(unittest.TestCase):
    """Verify primary acceptance criteria and invariants."""

    def test_contract_batch_shape_and_normalization(self):
        """Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        tracer = PreprocessingStateTracer()
        normalized, trace = tracer.validate_and_normalize(batch)

        # Shape check
        self.assertEqual(get_shape(batch), (1, 2, 2))
        self.assertEqual(get_shape(normalized), (1, 2, 2))

        # Value mapping check
        self.assertEqual(normalized[0][0][0], 0.0)
        self.assertEqual(normalized[0][0][1], 1.0)
        self.assertAlmostEqual(normalized[0][1][0], 128.0 / 255.0, places=7)
        self.assertAlmostEqual(normalized[0][1][1], 64.0 / 255.0, places=7)

        # Invariant check across trace
        self.assertEqual(len(trace), 3)
        self.assertTrue(trace[2]["shape_invariant_preserved"])

    def test_negative_pixel_rejected(self):
        """Negative pixels (< 0) are rejected with ValueError."""
        invalid_batch: Batch3D = [[[-1, 255], [128, 64]]]
        tracer = PreprocessingStateTracer()
        with self.assertRaises(ValueError):
            tracer.validate_and_normalize(invalid_batch)

    def test_out_of_bounds_high_pixel_rejected(self):
        """Pixels exceeding 255 are rejected with ValueError."""
        invalid_batch: Batch3D = [[[0, 256], [128, 64]]]
        tracer = PreprocessingStateTracer()
        with self.assertRaises(ValueError):
            tracer.validate_and_normalize(invalid_batch)

    def test_non_uniform_batch_rejected(self):
        """Ragged/non-rectangular dimensions raise ValueError."""
        ragged_batch = [[[0, 255]], [[128, 64, 32]]]
        tracer = PreprocessingStateTracer()
        with self.assertRaises(ValueError):
            tracer.validate_and_normalize(ragged_batch)

    def test_deliberate_wrong_result_fails(self):
        """Deliberate incorrect expectation fails test."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        tracer = PreprocessingStateTracer()
        normalized, _ = tracer.validate_and_normalize(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(normalized[0][0][1], 0.5)


if __name__ == "__main__":
    unittest.main()
