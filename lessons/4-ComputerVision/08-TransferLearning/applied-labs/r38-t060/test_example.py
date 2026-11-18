import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import TrainingOnlyMeanImputer, SCENARIO_QUESTIONS

class TestTrainingOnlyImputation(unittest.TestCase):
    def test_stated_scenario(self):
        # Training [2,None,4] yields mean 3; validation [100,None] becomes [100,3], not [100,100].
        imputer = TrainingOnlyMeanImputer()
        imputer.fit([2, None, 4])
        self.assertAlmostEqual(imputer.fitted_mean, 3.0)

        val_transformed = imputer.transform([100, None])
        self.assertEqual(val_transformed, [100.0, 3.0])
        self.assertNotEqual(val_transformed, [100.0, 100.0])

    def test_all_missing_training_rejected(self):
        imputer = TrainingOnlyMeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([None, None])

    def test_empty_training_rejected(self):
        imputer = TrainingOnlyMeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([])

    def test_transform_before_fit_rejected(self):
        imputer = TrainingOnlyMeanImputer()
        with self.assertRaises(RuntimeError):
            imputer.transform([1, None])

    def test_scenario_questions_assessment(self):
        # Q1: Predict output
        imputer = TrainingOnlyMeanImputer().fit([2, None, 4])
        self.assertEqual(imputer.transform([100, None]), SCENARIO_QUESTIONS[0]["expected_answer"])

        # Q2: Diagnose failure
        with self.assertRaises(SCENARIO_QUESTIONS[1]["expected_exception"]):
            TrainingOnlyMeanImputer().fit([None, None])

        # Q3: Boundary policy rationale verification
        self.assertIn("leak", SCENARIO_QUESTIONS[2]["rationale"].lower())

    def test_deliberate_failure_check(self):
        imputer = TrainingOnlyMeanImputer().fit([2, None, 4])
        self.assertNotEqual(imputer.transform([100, None]), [100.0, 100.0])

if __name__ == "__main__":
    unittest.main()
