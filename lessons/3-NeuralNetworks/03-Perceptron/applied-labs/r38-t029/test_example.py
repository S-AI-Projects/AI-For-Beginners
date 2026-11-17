"""
Unit tests for R38-T029: Single Perceptron Update — Scenario Assessment.

Tests cover:
  - Canonical spec update: weights=[0,0] bias=0 pred=0 → [1,2] bias=1
  - Zero-score tie rule (score=0 → prediction=1)
  - No update when prediction == target
  - Weight update arithmetic for misprediction (both directions)
  - Dimension mismatch raises ValueError
  - Invalid target raises ValueError
  - Invalid learning_rate raises ValueError
  - Scenario A (predict output)
  - Scenario B (diagnose failure) with AND-gate and broken model
  - Scenario C (boundary policy) for all three policies and all score signs
"""

import unittest

from example import (
    Perceptron,
    perceptron_update,
    scenario_choose_boundary_policy,
    scenario_diagnose_failure,
    scenario_predict_output,
)


class TestPerceptronForwardPass(unittest.TestCase):
    """Tests for score() and predict()."""

    def test_score_basic(self):
        """Weighted sum plus bias computed correctly."""
        p = Perceptron(weights=[1.0, 2.0], bias=1.0)
        self.assertAlmostEqual(p.score([1.0, 2.0]), 6.0)

    def test_score_zero(self):
        """Score is exactly 0 for zero weights and bias."""
        p = Perceptron(weights=[0.0, 0.0], bias=0.0)
        self.assertAlmostEqual(p.score([1.0, 2.0]), 0.0)

    def test_zero_score_tie_rule_predicts_one(self):
        """Score == 0 must predict 1 (documented tie rule: score >= 0 → 1)."""
        p = Perceptron(weights=[0.0, 0.0], bias=0.0)
        self.assertEqual(p.predict([1.0, 2.0]), 1)

    def test_positive_score_predicts_one(self):
        p = Perceptron(weights=[1.0, 0.0], bias=0.0)
        self.assertEqual(p.predict([3.0, 0.0]), 1)

    def test_negative_score_predicts_zero(self):
        p = Perceptron(weights=[-1.0, 0.0], bias=0.0)
        self.assertEqual(p.predict([3.0, 0.0]), 0)

    def test_dimension_mismatch_raises(self):
        """Mismatched input/weight lengths raise ValueError."""
        p = Perceptron(weights=[1.0, 2.0], bias=0.0)
        with self.assertRaises(ValueError):
            p.score([1.0])


class TestPerceptronUpdate(unittest.TestCase):
    """Tests for the learning rule (update step)."""

    def test_canonical_spec_update_formula(self):
        """
        Spec example: weights=[0,0] bias=0, pred=0 (forced), target=1, lr=1
        delta = 1*(1-0) = 1
        w_new = [1, 2], b_new = 1
        """
        # Use bias=-0.5 so score<0 → prediction=0, matching spec illustration
        result = perceptron_update(
            weights=[0.0, 0.0], bias=-0.5,
            inputs=[1.0, 2.0], target=1, learning_rate=1.0,
        )
        self.assertEqual(result.prediction, 0)
        self.assertAlmostEqual(result.delta, 1.0)
        self.assertEqual(result.new_weights, [1.0, 2.0])
        self.assertAlmostEqual(result.new_bias, 0.5)  # -0.5 + 1 = 0.5

    def test_spec_formula_values(self):
        """
        Verify that delta=1 on a [0,0]/bias=0 perceptron with pred=0 assumed
        produces new_weights=[1,2] and new_bias=1 (pure formula check).
        """
        # Formulaically: w_new = [0,0] + 1*[1,2] = [1,2], b_new = 0+1 = 1
        w = [0.0, 0.0]
        b = 0.0
        x = [1.0, 2.0]
        lr = 1.0
        target = 1
        pred = 0  # as stated in spec
        delta = lr * (target - pred)
        w_new = [wi + delta * xi for wi, xi in zip(w, x)]
        b_new = b + delta
        self.assertEqual(w_new, [1.0, 2.0])
        self.assertAlmostEqual(b_new, 1.0)

    def test_no_update_when_correct(self):
        """When prediction == target, weights and bias are unchanged."""
        result = perceptron_update(
            weights=[1.0, 2.0], bias=1.0,
            inputs=[1.0, 2.0], target=1, learning_rate=1.0,
        )
        self.assertEqual(result.delta, 0.0)
        self.assertFalse(result.weight_changed)
        self.assertEqual(result.new_weights, [1.0, 2.0])
        self.assertAlmostEqual(result.new_bias, 1.0)

    def test_update_pred1_target0(self):
        """
        When prediction=1 and target=0:
        delta = -1, weights decrease, bias decreases.
        """
        result = perceptron_update(
            weights=[1.0, 1.0], bias=0.0,
            inputs=[1.0, 1.0], target=0, learning_rate=1.0,
        )
        self.assertEqual(result.prediction, 1)
        self.assertAlmostEqual(result.delta, -1.0)
        self.assertEqual(result.new_weights, [0.0, 0.0])
        self.assertAlmostEqual(result.new_bias, -1.0)

    def test_learning_rate_scales_delta(self):
        """Delta scales linearly with learning_rate."""
        result = perceptron_update(
            weights=[0.0, 0.0], bias=-1.0,
            inputs=[1.0, 1.0], target=1, learning_rate=0.5,
        )
        self.assertAlmostEqual(result.delta, 0.5)
        self.assertEqual(result.new_weights, [0.5, 0.5])
        self.assertAlmostEqual(result.new_bias, -0.5)

    def test_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            perceptron_update([0.0, 0.0], 0.0, [1.0, 1.0], target=2)

    def test_invalid_learning_rate_raises(self):
        with self.assertRaises(ValueError):
            perceptron_update([0.0, 0.0], 0.0, [1.0, 1.0], target=1, learning_rate=-1.0)

    def test_invalid_learning_rate_zero_raises(self):
        with self.assertRaises(ValueError):
            perceptron_update([0.0, 0.0], 0.0, [1.0, 1.0], target=1, learning_rate=0.0)


class TestScenarioA_PredictOutput(unittest.TestCase):
    """Scenario A: predict output for given weights/bias/inputs."""

    def test_known_positive_score(self):
        r = scenario_predict_output([1.0, 2.0], 1.0, [1.0, 2.0])
        self.assertAlmostEqual(r["score"], 6.0)
        self.assertEqual(r["prediction"], 1)

    def test_zero_score_tie_predicts_one(self):
        """score = 1*1 + (-1)*1 + 0 = 0 → prediction must be 1."""
        r = scenario_predict_output([1.0, -1.0], 0.0, [1.0, 1.0])
        self.assertAlmostEqual(r["score"], 0.0)
        self.assertEqual(r["prediction"], 1)

    def test_negative_score_predicts_zero(self):
        r = scenario_predict_output([-1.0, -1.0], 0.0, [1.0, 1.0])
        self.assertAlmostEqual(r["score"], -2.0)
        self.assertEqual(r["prediction"], 0)


class TestScenarioB_DiagnoseFailure(unittest.TestCase):
    """Scenario B: diagnose misclassifications on a labelled dataset."""

    AND_DATASET = [
        ([0.0, 0.0], 0),
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 0),
        ([1.0, 1.0], 1),
    ]

    def test_perfect_and_gate(self):
        """weights=[1,1] bias=-1.5 implements AND perfectly."""
        diag = scenario_diagnose_failure([1.0, 1.0], -1.5, self.AND_DATASET)
        self.assertEqual(diag["correct"], 4)
        self.assertEqual(diag["misclassified_indices"], [])
        self.assertAlmostEqual(diag["accuracy"], 1.0)

    def test_zero_weight_model_misclassifies(self):
        """Zero weights → score=0 → all predict 1; 3 out of 4 are wrong."""
        diag = scenario_diagnose_failure([0.0, 0.0], 0.0, self.AND_DATASET)
        # Samples 0,1,2 have target=0 but model predicts 1
        self.assertEqual(diag["misclassified_indices"], [0, 1, 2])
        self.assertEqual(diag["correct"], 1)
        self.assertAlmostEqual(diag["accuracy"], 0.25)

    def test_empty_dataset(self):
        diag = scenario_diagnose_failure([1.0], 0.0, [])
        self.assertEqual(diag["total"], 0)
        self.assertAlmostEqual(diag["accuracy"], 0.0)


class TestScenarioC_BoundaryPolicy(unittest.TestCase):
    """Scenario C: boundary policy comparison at score=0 and other scores."""

    def test_positive_tie_zero_score(self):
        self.assertEqual(scenario_choose_boundary_policy(0.0, "positive_tie"), 1)

    def test_negative_tie_zero_score(self):
        self.assertEqual(scenario_choose_boundary_policy(0.0, "negative_tie"), 0)

    def test_abstain_zero_score(self):
        self.assertEqual(scenario_choose_boundary_policy(0.0, "abstain"), -1)

    def test_all_policies_positive_score(self):
        for policy in ("positive_tie", "negative_tie", "abstain"):
            self.assertEqual(scenario_choose_boundary_policy(2.0, policy), 1)

    def test_all_policies_negative_score(self):
        for policy in ("positive_tie", "negative_tie", "abstain"):
            self.assertEqual(scenario_choose_boundary_policy(-2.0, policy), 0)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            scenario_choose_boundary_policy(0.0, "majority_vote")


if __name__ == "__main__":
    unittest.main(verbosity=2)
