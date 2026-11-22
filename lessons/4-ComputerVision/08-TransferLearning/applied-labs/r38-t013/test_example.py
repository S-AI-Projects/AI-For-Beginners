"""
Unit tests for Training-Only Imputation (Task R38-T013)
"""

import unittest

from example import (
    TrainingMeanImputer,
    fit_mean,
    transform_with_mean,
)


class TestImputationContract(unittest.TestCase):
    """Verify primary acceptance criteria."""

    def test_training_and_validation_contract(self):
        """Training [2,None,4] yields mean 3; validation [100,None] becomes [100,3], not [100,100]."""
        train_input = [2, None, 4]
        val_input = [100, None]

        imputer = TrainingMeanImputer()
        imputer.fit(train_input)
        self.assertEqual(imputer.fitted_mean, 3.0)

        val_transformed = imputer.transform(val_input)
        self.assertEqual(val_transformed, [100.0, 3.0])
        self.assertNotEqual(val_transformed, [100.0, 100.0])

    def test_all_missing_training_input_rejected(self):
        """All-missing training input must be rejected with ValueError."""
        imputer = TrainingMeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([None, None])

        with self.assertRaises(ValueError):
            fit_mean([None, None, None])

    def test_empty_training_input_rejected(self):
        """Empty input raises ValueError."""
        imputer = TrainingMeanImputer()
        with self.assertRaises(ValueError):
            imputer.fit([])


class TestApproachComparison(unittest.TestCase):
    """Compare Approach 1 (OOP) and Approach 2 (Functional) for equivalence."""

    def test_approach_equivalence_on_contract_fixture(self):
        train = [2, None, 4]
        val = [100, None]

        # Approach 1
        imp = TrainingMeanImputer().fit(train)
        out1_train = imp.transform(train)
        out1_val = imp.transform(val)

        # Approach 2
        m = fit_mean(train)
        out2_train = transform_with_mean(train, m)
        out2_val = transform_with_mean(val, m)

        self.assertEqual(out1_train, out2_train)
        self.assertEqual(out1_val, out2_val)

    def test_deliberate_wrong_result_fails(self):
        """Deliberate wrong result (leakage outcome [100, 100]) fails assertion."""
        imputer = TrainingMeanImputer().fit([2, None, 4])
        val_out = imputer.transform([100, None])
        with self.assertRaises(AssertionError):
            self.assertEqual(val_out, [100.0, 100.0])

    def test_transform_before_fit_raises_error(self):
        imputer = TrainingMeanImputer()
        with self.assertRaises(RuntimeError):
            imputer.transform([1, None])


if __name__ == "__main__":
    unittest.main()
