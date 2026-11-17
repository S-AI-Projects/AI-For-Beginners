"""
Unit tests and worked-example regression checks for Cosine Similarity (Issue #27 / R38-T027).
"""

import unittest
from example import (
    cosine_similarity,
    dot_product,
    find_most_similar,
    vector_norm,
)


class TestCosineSimilarityLookup(unittest.TestCase):
    def test_identical_vectors(self):
        """[1,0] vs [1,0] = 1.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([5, 12], [5, 12]), 1.0, places=7)

    def test_orthogonal_vectors(self):
        """[1,0] vs [0,1] = 0.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=7)
        self.assertAlmostEqual(cosine_similarity([0, 5], [3, 0]), 0.0, places=7)

    def test_opposite_vectors(self):
        """Opposite direction vectors yield -1.0."""
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0, places=7)
        self.assertAlmostEqual(cosine_similarity([2, 3], [-2, -3]), -1.0, places=7)

    def test_zero_vector_sentinel(self):
        """Explicitly handle zero vectors by returning documented sentinel None."""
        # u is zero vector
        self.assertIsNone(cosine_similarity([0, 0], [1, 0]))
        # v is zero vector
        self.assertIsNone(cosine_similarity([1, 0], [0, 0]))
        # both zero vectors
        self.assertIsNone(cosine_similarity([0, 0, 0], [0, 0, 0]))

    def test_dimension_mismatch_rejection(self):
        """Mismatched vector lengths raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            cosine_similarity([1, 0], [1, 0, 0])
        self.assertIn("dimension mismatch", str(ctx.exception))

    def test_empty_vector_rejection(self):
        """Empty vectors raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            cosine_similarity([], [])
        self.assertIn("non-empty", str(ctx.exception))

    def test_type_validation(self):
        """Non-numeric elements or non-sequence types raise TypeError."""
        with self.assertRaises(TypeError):
            cosine_similarity("not a list", [1, 0])  # type: ignore

        with self.assertRaises(TypeError):
            cosine_similarity([1, "zero"], [1, 0])  # type: ignore

        with self.assertRaises(TypeError):
            cosine_similarity([1, True], [1, 0])  # type: ignore

    def test_non_finite_rejection(self):
        """Reject float('nan') or float('inf')."""
        with self.assertRaises(ValueError):
            cosine_similarity([1, float('nan')], [1, 0])

        with self.assertRaises(ValueError):
            cosine_similarity([1, float('inf')], [1, 0])

    def test_worked_example_arithmetic_regression_check(self):
        """
        Regression check for documented worked example:
        Let u = [3, 4] and v = [4, 3]
        Step 1: dot_product(u, v) = (3*4) + (4*3) = 12 + 12 = 24
        Step 2: norm(u) = sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        Step 3: norm(v) = sqrt(4^2 + 3^2) = sqrt(16 + 9) = sqrt(25) = 5
        Step 4: cos_sim = 24 / (5 * 5) = 24 / 25 = 0.96
        """
        u = [3, 4]
        v = [4, 3]

        # Verify arithmetic components
        self.assertEqual(dot_product(u, v), 24)
        self.assertAlmostEqual(vector_norm(u), 5.0, places=7)
        self.assertAlmostEqual(vector_norm(v), 5.0, places=7)

        # Verify final cosine similarity
        result = cosine_similarity(u, v)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.96, places=7)

    def test_find_most_similar_batch(self):
        """Batch lookup across candidate embeddings."""
        query = [1, 0]
        candidates = [
            [1, 0],   # identical (1.0)
            [0, 1],   # orthogonal (0.0)
            [-1, 0],  # opposite (-1.0)
            [0, 0],   # zero-vector (None)
        ]
        results = find_most_similar(query, candidates)
        self.assertEqual(results[0], (0, 1.0))
        self.assertEqual(results[1], (1, 0.0))
        self.assertEqual(results[2], (2, -1.0))
        self.assertEqual(results[3], (3, None))


if __name__ == "__main__":
    unittest.main()
