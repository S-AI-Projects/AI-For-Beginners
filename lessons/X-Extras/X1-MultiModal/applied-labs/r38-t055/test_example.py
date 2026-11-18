import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import evaluate_binary_predictions

class TestBinaryEvaluation(unittest.TestCase):
    def test_stated_scenario(self):
        # y_true=[1,1,0,0], y_pred=[1,0,1,0] gives TP=1,FN=1,FP=1,TN=1, precision=recall=0.5
        m = evaluate_binary_predictions([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(m.tp, 1)
        self.assertEqual(m.fn, 1)
        self.assertEqual(m.fp, 1)
        self.assertEqual(m.tn, 1)
        self.assertAlmostEqual(m.precision, 0.5)
        self.assertAlmostEqual(m.recall, 0.5)

    def test_zero_denominator_policy(self):
        # When model predicts all 0s, TP=0, FP=0 -> precision denominator is 0
        m = evaluate_binary_predictions([1, 1], [0, 0])
        self.assertEqual(m.precision, 0.0)
        self.assertEqual(m.recall, 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions([1, 0], [1])

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions([], [])

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary_predictions([1, 2], [1, 0])

    def test_deliberate_failure_check(self):
        m = evaluate_binary_predictions([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertNotEqual(m.precision, 1.0)

if __name__ == "__main__":
    unittest.main()
