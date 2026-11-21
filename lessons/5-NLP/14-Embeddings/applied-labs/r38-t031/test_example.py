"""
Unit tests for R38-T031: Cosine-similarity lookup runnable concept example.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    UNDEFINED_SENTINEL,
    cosine_similarity,
    dot_product,
    find_most_similar,
    vector_norm,
)


class TestCosineSimilarityContract(unittest.TestCase):
    """Verify the core stated contracts for cosine similarity."""

    def test_identical_vectors_yield_one(self):
        """[1,0] versus [1,0] gives 1."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([0, 1], [0, 1]), 1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([3.0, 4.0], [3.0, 4.0]), 1.0, places=7)

    def test_orthogonal_vectors_yield_zero(self):
        """[1,0] versus [0,1] gives 0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=7)
        self.assertAlmostEqual(cosine_similarity([0, 2], [5, 0]), 0.0, places=7)

    def test_zero_vector_returns_undefined_sentinel(self):
        """A zero vector returns the documented undefined-result sentinel (None)."""
        # Right operand zero
        self.assertIs(cosine_similarity([1, 0], [0, 0]), UNDEFINED_SENTINEL)
        # Left operand zero
        self.assertIs(cosine_similarity([0, 0], [1, 0]), UNDEFINED_SENTINEL)
        # Both zero
        self.assertIs(cosine_similarity([0, 0], [0, 0]), UNDEFINED_SENTINEL)
        # Multi-dimensional zero
        self.assertIs(cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), UNDEFINED_SENTINEL)

    def test_opposite_vectors_yield_negative_one(self):
        """Opposite direction vectors yield -1.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([0, -5], [0, 5]), -1.0, places=7)

    def test_arbitrary_vector_similarity(self):
        """Cosine similarity on arbitrary known non-trivial 2D vectors: [3, 4] vs [4, 3] = 24/25 = 0.96."""
        res = cosine_similarity([3, 4], [4, 3])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res, 0.96, places=7)

    def test_vector_norm_calculation(self):
        """vector_norm returns correct Euclidean norm sqrt(sum(x_i^2))."""
        self.assertAlmostEqual(vector_norm([3, 4]), 5.0, places=7)
        self.assertAlmostEqual(vector_norm([1, 0]), 1.0, places=7)
        self.assertAlmostEqual(vector_norm([0, 0]), 0.0, places=7)

    def test_dot_product_calculation(self):
        """dot_product returns correct sum of element-wise products."""
        self.assertAlmostEqual(dot_product([1, 2], [3, 4]), 11.0, places=7)
        self.assertAlmostEqual(dot_product([1, 0], [0, 1]), 0.0, places=7)

    def test_dimension_mismatch_raises_value_error(self):
        """Vectors of differing lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 0])
        with self.assertRaises(ValueError):
            dot_product([1, 0], [1, 0, 0])

    def test_empty_vectors_raise_value_error(self):
        """Empty vectors must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([], [])
        with self.assertRaises(ValueError):
            vector_norm([])

    def test_non_finite_values_raise_value_error(self):
        """NaN and infinity values must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan"), 0], [1, 0])
        with self.assertRaises(ValueError):
            cosine_similarity([float("inf"), 0], [1, 0])

    def test_non_numeric_types_raise_type_error(self):
        """Non-numeric or boolean values must raise TypeError."""
        with self.assertRaises(TypeError):
            cosine_similarity(["a", "b"], [1, 0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity([True, False], [1, 0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity("not-a-list", [1, 0])  # type: ignore

    def test_find_most_similar_ranking(self):
        """find_most_similar correctly ranks vocabulary candidates by similarity."""
        candidates = {
            "close": [0.99, 0.01],
            "medium": [0.707, 0.707],
            "orthogonal": [0.0, 1.0],
            "zero": [0.0, 0.0],
        }
        query = [1.0, 0.0]
        results = find_most_similar(query, candidates)

        labels = [item[0] for item in results]
        self.assertEqual(labels, ["close", "medium", "orthogonal", "zero"])
        self.assertIs(results[-1][1], UNDEFINED_SENTINEL)

    def test_deliberate_wrong_result_fails(self):
        """
        Verify that a deliberate incorrect expectation produces a test failure.
        Done When criterion: 'a deliberate wrong result makes the test fail'.
        """
        actual_sim = cosine_similarity([1, 0], [0, 1])
        # Actual is 0.0; asserting it equals 1.0 fails
        self.assertNotEqual(actual_sim, 1.0)
        with self.assertRaises(AssertionError):
            self.assertEqual(actual_sim, 1.0)


if __name__ == "__main__":
    unittest.main()
