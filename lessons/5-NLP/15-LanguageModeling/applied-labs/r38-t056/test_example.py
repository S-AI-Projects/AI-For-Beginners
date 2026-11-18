import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    compute_cosine_similarity,
    format_similarity_report,
    UNDEFINED_SENTINEL,
)

class TestRefactoredCosineSimilarity(unittest.TestCase):
    def test_stated_scenario(self):
        # [1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns sentinel
        self.assertAlmostEqual(compute_cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(compute_cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertIs(compute_cosine_similarity([1, 0], [0, 0]), UNDEFINED_SENTINEL)

    def test_presentation_separation(self):
        score = compute_cosine_similarity([1, 0], [0, 1])
        report = format_similarity_report("word1", "word2", score)
        self.assertEqual(report["comparison"], "word1 vs word2")
        self.assertEqual(report["score"], 0.0)
        self.assertEqual(report["status"], "VALID")

        zero_report = format_similarity_report("zero", "word", None)
        self.assertEqual(zero_report["status"], "UNDEFINED_ZERO_VECTOR")

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            compute_cosine_similarity([1, 0], [1])
        with self.assertRaises(ValueError):
            compute_cosine_similarity([], [])

    def test_deliberate_failure_check(self):
        self.assertNotEqual(compute_cosine_similarity([1, 0], [1, 0]), 0.0)

if __name__ == "__main__":
    unittest.main()
