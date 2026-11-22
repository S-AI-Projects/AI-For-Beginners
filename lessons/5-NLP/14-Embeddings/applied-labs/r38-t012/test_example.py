"""
Unit tests for Cosine Similarity Boundary-Case Lesson Extension (Task R38-T012)
"""

import math
import unittest

from example import (
    UNDEFINED_SENTINEL,
    cosine_similarity,
    dot_product,
    is_undefined,
    vector_norm,
)


class TestBoundaryCaseContract(unittest.TestCase):
    """Test standard contract fixtures."""

    def test_identical_vectors_yield_one(self):
        """[1,0] vs [1,0] gives 1.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0, places=7)

    def test_orthogonal_vectors_yield_zero(self):
        """[1,0] vs [0,1] gives 0.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=7)

    def test_zero_vector_returns_sentinel(self):
        """Zero vector returns documented undefined-result sentinel."""
        res_a = cosine_similarity([0, 0], [1, 0])
        res_b = cosine_similarity([1, 0], [0, 0])
        res_both = cosine_similarity([0, 0], [0, 0])
        self.assertTrue(is_undefined(res_a))
        self.assertTrue(is_undefined(res_b))
        self.assertTrue(is_undefined(res_both))


class TestSmallestValidAndBoundaryCases(unittest.TestCase):
    """Test smallest valid and boundary edge cases."""

    def test_smallest_valid_1d_vectors(self):
        """1-dimensional non-zero vectors represent the minimal valid input."""
        self.assertAlmostEqual(cosine_similarity([5.0], [2.0]), 1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([5.0], [-2.0]), -1.0, places=7)

    def test_boundary_opposite_direction(self):
        """Opposite direction represents the minimum similarity boundary (-1.0)."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0, places=7)

    def test_subnormal_near_zero_boundary(self):
        """Magnitudes below tolerance trigger the zero-norm boundary decision."""
        self.assertTrue(is_undefined(cosine_similarity([1e-15, 0], [1, 0])))

    def test_deliberate_wrong_result_fails(self):
        """Verify test suite fails when assertion is deliberately violated."""
        result = cosine_similarity([1, 0], [1, 0])
        with self.assertRaises(AssertionError):
            self.assertEqual(result, 0.5)


class TestInvalidInputs(unittest.TestCase):
    """Test invalid input rejection."""

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])


if __name__ == "__main__":
    unittest.main()
