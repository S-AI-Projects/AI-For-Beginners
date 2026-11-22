"""
Unit tests for Tensor-Shape Contract Scenario Assessment (Task R38-T020)
"""

import unittest

from example import (
    APPLIED_SCENARIOS,
    Batch3D,
    assess_scenarios,
    get_shape,
    normalize_image_batch,
)


class TestTensorShapeContract(unittest.TestCase):
    """Verify primary acceptance criteria."""

    def test_contract_batch_shape_and_normalization(self):
        """Batch [[[0,255],[128,64]]] stays shape (1,2,2); 0->0 and 255->1."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = normalize_image_batch(batch)

        self.assertEqual(get_shape(batch), (1, 2, 2))
        self.assertEqual(get_shape(out), (1, 2, 2))
        self.assertEqual(out[0][0][0], 0.0)
        self.assertEqual(out[0][0][1], 1.0)
        self.assertAlmostEqual(out[0][1][0], 128.0 / 255.0, places=7)
        self.assertAlmostEqual(out[0][1][1], 64.0 / 255.0, places=7)

    def test_negative_pixel_rejected(self):
        """Negative pixels (< 0) are rejected with ValueError."""
        with self.assertRaises(ValueError):
            normalize_image_batch([[[-1, 255], [128, 64]]])

    def test_pixel_greater_than_255_rejected(self):
        """Pixels > 255 are rejected with ValueError."""
        with self.assertRaises(ValueError):
            normalize_image_batch([[[0, 256], [128, 64]]])


class TestScenarioAssessment(unittest.TestCase):
    """Verify executable checks and rationales for applied assessment questions."""

    def test_all_scenario_assessments_pass(self):
        """Verify all three applied questions pass their factual checks."""
        assessment = assess_scenarios()
        self.assertTrue(assessment["Q1_predict_output"])
        self.assertTrue(assessment["Q2_diagnose_failure"])
        self.assertTrue(assessment["Q3_boundary_policy"])

    def test_q1_output_prediction_accuracy(self):
        """Verify Q1 output prediction matches contract."""
        q1 = APPLIED_SCENARIOS["Q1_predict_output"]
        out = normalize_image_batch([[[0, 255], [128, 64]]])
        self.assertEqual(get_shape(out), q1["expected_shape"])
        self.assertEqual(out[0][0][0], q1["expected_values"][0][0][0])
        self.assertEqual(out[0][0][1], q1["expected_values"][0][0][1])

    def test_q2_failure_diagnosis(self):
        """Verify Q2 failure diagnosis correctly catches -5 error."""
        with self.assertRaises(ValueError) as ctx:
            normalize_image_batch([[[-5, 255], [128, 64]]])
        self.assertIn("out of range [0, 255]", str(ctx.exception))

    def test_deliberate_wrong_result_fails(self):
        """Deliberate incorrect expectation triggers test failure."""
        batch: Batch3D = [[[0, 255], [128, 64]]]
        out = normalize_image_batch(batch)
        with self.assertRaises(AssertionError):
            self.assertEqual(out[0][0][0], 1.0)


if __name__ == "__main__":
    unittest.main()
