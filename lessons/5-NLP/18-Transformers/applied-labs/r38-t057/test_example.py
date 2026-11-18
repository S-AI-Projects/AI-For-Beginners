import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import compute_cosine_similarity_worked, UNDEFINED_SENTINEL

class TestWorkedCosineSimilarity(unittest.TestCase):
    def test_stated_scenario(self):
        # [1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns sentinel
        res_ident = compute_cosine_similarity_worked([1, 0], [1, 0])
        self.assertAlmostEqual(res_ident["similarity"], 1.0)
        self.assertEqual(res_ident["dot_product"], 1.0)

        res_ortho = compute_cosine_similarity_worked([1, 0], [0, 1])
        self.assertAlmostEqual(res_ortho["similarity"], 0.0)
        self.assertEqual(res_ortho["dot_product"], 0.0)

        res_zero = compute_cosine_similarity_worked([1, 0], [0, 0])
        self.assertIs(res_zero["similarity"], UNDEFINED_SENTINEL)
        self.assertTrue(res_zero["is_zero_vector"])

    def test_dimension_validation(self):
        with self.assertRaises(ValueError):
            compute_cosine_similarity_worked([1], [1, 2])
        with self.assertRaises(ValueError):
            compute_cosine_similarity_worked([], [])

    def test_deliberate_failure_check(self):
        res = compute_cosine_similarity_worked([1, 0], [1, 0])
        self.assertNotEqual(res["similarity"], -1.0)

if __name__ == "__main__":
    unittest.main()
