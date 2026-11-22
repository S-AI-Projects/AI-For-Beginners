"""
Unit tests for Cosine Similarity Lookup Concept Example (Task R38-T011)
lessons/5-NLP/18-Transformers/applied-labs/r38-t011/
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


class TestCosineSimilarityContract(unittest.TestCase):
    """Verify primary acceptance criteria and contract specifications."""

    def test_identical_vectors_yield_one(self):
        """[1,0] versus [1,0] must yield 1.0."""
        result = cosine_similarity([1, 0], [1, 0])
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_orthogonal_vectors_yield_zero(self):
        """[1,0] versus [0,1] must yield 0.0."""
        result = cosine_similarity([1, 0], [0, 1])
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_opposite_vectors_yield_negative_one(self):
        """[1,0] versus [-1,0] must yield -1.0."""
        result = cosine_similarity([1, 0], [-1, 0])
        self.assertAlmostEqual(result, -1.0, places=7)

    def test_zero_vector_first_returns_sentinel(self):
        """Zero vector as first argument returns the documented sentinel."""
        result = cosine_similarity([0, 0], [1, 0])
        self.assertTrue(is_undefined(result))
        self.assertTrue(math.isnan(result))

    def test_zero_vector_second_returns_sentinel(self):
        """Zero vector as second argument returns the documented sentinel."""
        result = cosine_similarity([1, 0], [0, 0])
        self.assertTrue(is_undefined(result))
        self.assertTrue(math.isnan(result))

    def test_both_zero_vectors_return_sentinel(self):
        """Both inputs being zero vectors returns the documented sentinel."""
        result = cosine_similarity([0, 0, 0], [0, 0, 0])
        self.assertTrue(is_undefined(result))
        self.assertTrue(math.isnan(result))

    def test_collinear_scaled_vectors_yield_one(self):
        """Scaled positive collinear vectors must yield 1.0 regardless of magnitude."""
        result = cosine_similarity([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(result, 1.0, places=7)


class TestBoundaryAndFailureScenarios(unittest.TestCase):
    """Test boundary conditions, invalid inputs, and error handling."""

    def test_dimension_mismatch_raises_value_error(self):
        """Vectors of unequal lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_empty_vectors_raise_value_error(self):
        """Empty vectors must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_near_zero_below_tolerance_returns_sentinel(self):
        """Vectors with magnitude <= tolerance are treated as zero norm."""
        near_zero = [1e-15, 1e-15]
        result = cosine_similarity(near_zero, [1.0, 0.0])
        self.assertTrue(is_undefined(result))

    def test_deliberate_wrong_result_fails(self):
        """Verify that an incorrect assertion fails, ensuring test sensitivity."""
        result = cosine_similarity([1, 0], [1, 0])
        # Asserting deliberate wrong value (0.0 instead of 1.0) must raise AssertionError
        with self.assertRaises(AssertionError):
            self.assertEqual(result, 0.0)


class TestVectorMathPrimitives(unittest.TestCase):
    """Unit tests for underlying math helper functions."""

    def test_vector_norm(self):
        self.assertAlmostEqual(vector_norm([3, 4]), 5.0, places=7)
        self.assertEqual(vector_norm([0, 0]), 0.0)

    def test_dot_product(self):
        self.assertAlmostEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0, places=7)

    def test_dot_product_mismatch(self):
        with self.assertRaises(ValueError):
            dot_product([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
