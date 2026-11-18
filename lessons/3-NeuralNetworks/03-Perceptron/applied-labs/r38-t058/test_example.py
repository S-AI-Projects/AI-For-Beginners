import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import PerceptronState, train_step, predict_score

class TestPerceptronUpdate(unittest.TestCase):
    def test_stated_scenario(self):
        # Weights [0,0], bias 0, input [1,2], target 1, initial prediction 0:
        # learning-rate 1 update gives weights [1,2], bias 1.
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
        new_state, initial_pred, error = train_step(state, [1, 2], 1)

        self.assertEqual(initial_pred, 0)
        self.assertEqual(error, 1)
        self.assertEqual(new_state.weights, (1.0, 2.0))
        self.assertEqual(new_state.bias, 1.0)

    def test_zero_score_tie_rule(self):
        # score = 0 * 1 + 0 * 2 + 0 = 0 -> prediction must be 0
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0)
        score, pred = predict_score(state, [1, 2])
        self.assertEqual(score, 0.0)
        self.assertEqual(pred, 0)

    def test_failure_and_recovery(self):
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)

        # 1. Invalid input: missing/None value raises ValueError
        with self.assertRaises(ValueError):
            train_step(state, [1, None], 1)

        # 2. Invalid input: wrong dimension raises ValueError
        with self.assertRaises(ValueError):
            train_step(state, [1, 2, 3], 1)

        # 3. Recovery: next valid input succeeds and correctly transitions state
        recovered_state, pred, err = train_step(state, [1, 2], 1)
        self.assertEqual(recovered_state.weights, (1.0, 2.0))
        self.assertEqual(recovered_state.bias, 1.0)

    def test_deliberate_failure_check(self):
        state = PerceptronState(weights=(0.0, 0.0), bias=0.0, learning_rate=1.0)
        new_state, _, _ = train_step(state, [1, 2], 1)
        self.assertNotEqual(new_state.weights, (0.0, 0.0))

if __name__ == "__main__":
    unittest.main()
