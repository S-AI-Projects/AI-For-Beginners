import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import cosine_similarity, UNDEFINED_SENTINEL

class TestCosineSimilarityRNN(unittest.TestCase):
    def test_stated_scenario(self):
        # [1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns sentinel
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertIs(cosine_similarity([1, 0], [0, 0]), UNDEFINED_SENTINEL)
        self.assertIs(cosine_similarity([0, 0], [0, 0]), UNDEFINED_SENTINEL)

    def test_negative_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1, 0, 1])

    def test_empty_vectors(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_deliberate_failure_check(self):
        result = cosine_similarity([1, 0], [0, 1])
        self.assertNotEqual(result, 1.0)

if __name__ == "__main__":
    unittest.main()
