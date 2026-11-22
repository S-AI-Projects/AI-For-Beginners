"""
Unit tests for Cosine-Similarity Lookup Worked Example (Task R38-T017)
"""

import math
import unittest

from example import (
    UNDEFINED_SENTINEL,
    cosine_similarity,
    explain_cosine_similarity,
    is_undefined,
)


class TestWorkedExampleContract(unittest.TestCase):
    """Verify primary acceptance criteria and worked trace regression checks."""

    def test_identical_vectors_yield_one(self):
        """[1,0] versus [1,0] gives 1.0."""
        res = cosine_similarity([1, 0], [1, 0])
        self.assertAlmostEqual(res, 1.0, places=7)

    def test_orthogonal_vectors_yield_zero(self):
        """[1,0] versus [0,1] gives 0.0."""
        res = cosine_similarity([1, 0], [0, 1])
        self.assertAlmostEqual(res, 0.0, places=7)

    def test_zero_vector_returns_sentinel(self):
        """Zero vector returns documented undefined sentinel."""
        res = cosine_similarity([0, 0], [1, 0])
        self.assertTrue(is_undefined(res))

    def test_worked_trace_explanation(self):
        """Execute exact worked input-to-output explanation as regression check."""
        trace = explain_cosine_similarity([1, 0], [1, 0])
        self.assertEqual(trace["dot_product"], 1.0)
        self.assertEqual(trace["norm_a"], 1.0)
        self.assertEqual(trace["norm_b"], 1.0)
        self.assertEqual(trace["result"], 1.0)

        zero_trace = explain_cosine_similarity([0, 0], [1, 0])
        self.assertTrue(is_undefined(zero_trace["result"]))
        self.assertIn("Zero norm encountered", zero_trace["explanation"])

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_deliberate_wrong_result_fails(self):
        """Deliberate incorrect expectation triggers test failure."""
        res = cosine_similarity([1, 0], [1, 0])
        with self.assertRaises(AssertionError):
            self.assertEqual(res, 0.0)


if __name__ == "__main__":
    unittest.main()
