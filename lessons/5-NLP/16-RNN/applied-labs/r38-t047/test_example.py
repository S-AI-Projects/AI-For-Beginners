"""Unit tests for Cosine-Similarity Lookup — Scenario Assessment (R38-T047)."""

from __future__ import annotations

import math
import unittest

from example import (
    CASE1_A,
    CASE1_B,
    CASE2_A,
    CASE2_B,
    CASE3_A,
    CASE3_B,
    UNDEFINED_SENTINEL,
    check_question_1_predict_output,
    check_question_2_diagnose_failure,
    check_question_3_choose_boundary_policy,
    cosine_similarity,
    dot_product,
    run_all_scenario_assessments,
    vector_norm,
)


class TestCoreCosineSimilarity(unittest.TestCase):
    """Test fundamental mathematical behavior of cosine similarity."""

    def test_identical_vectors_produce_one(self) -> None:
        result = cosine_similarity(CASE1_A, CASE1_B)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_orthogonal_vectors_produce_zero(self) -> None:
        result = cosine_similarity(CASE2_A, CASE2_B)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_opposite_direction_vectors_produce_negative_one(self) -> None:
        result = cosine_similarity([2.0, 0.0], [-5.0, 0.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, -1.0, places=9)

    def test_known_nontrivial_acute_vectors(self) -> None:
        # A=[3, 4], B=[4, 3] -> dot=24, norms=5,5 -> 24/25 = 0.96
        result = cosine_similarity([3.0, 4.0], [4.0, 3.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.96, places=9)


class TestQuestion1PredictOutput(unittest.TestCase):
    """Test Question 1: Predict an output and its executable check."""

    def test_q1_assessment_passes(self) -> None:
        assessment = check_question_1_predict_output()
        self.assertEqual(assessment.question_id, "Q1")
        self.assertEqual(assessment.question_type, "predict_output")
        self.assertTrue(assessment.check_passed)

    def test_q1_identical_fixture_matches_expected(self) -> None:
        # Independently calculated expected value = 1.0
        expected = 1.0
        actual = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual, expected, places=9)

    def test_q1_orthogonal_fixture_matches_expected(self) -> None:
        # Independently calculated expected value = 0.0
        expected = 0.0
        actual = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual, expected, places=9)


class TestQuestion2DiagnoseFailure(unittest.TestCase):
    """Test Question 2: Diagnose zero-vector failure and verify safe handling."""

    def test_q2_assessment_passes(self) -> None:
        assessment = check_question_2_diagnose_failure()
        self.assertEqual(assessment.question_id, "Q2")
        self.assertEqual(assessment.question_type, "diagnose_failure")
        self.assertTrue(assessment.check_passed)

    def test_q2_first_vector_zero_returns_sentinel(self) -> None:
        result = cosine_similarity([0.0, 0.0], [1.0, 0.0])
        self.assertIs(result, UNDEFINED_SENTINEL)
        self.assertIsNone(result)

    def test_q2_second_vector_zero_returns_sentinel(self) -> None:
        result = cosine_similarity([1.0, 0.0], [0.0, 0.0])
        self.assertIs(result, UNDEFINED_SENTINEL)
        self.assertIsNone(result)

    def test_q2_both_vectors_zero_returns_sentinel(self) -> None:
        result = cosine_similarity([0.0, 0.0], [0.0, 0.0])
        self.assertIs(result, UNDEFINED_SENTINEL)
        self.assertIsNone(result)

    def test_q2_no_division_by_zero_raised(self) -> None:
        try:
            res = cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.assertIsNone(res)
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised instead of returning sentinel!")


class TestQuestion3ChooseBoundaryPolicy(unittest.TestCase):
    """Test Question 3: Verify the boundary policy matches R38-T046."""

    def test_q3_assessment_passes(self) -> None:
        assessment = check_question_3_choose_boundary_policy()
        self.assertEqual(assessment.question_id, "Q3")
        self.assertEqual(assessment.question_type, "choose_boundary_policy")
        self.assertTrue(assessment.check_passed)

    def test_q3_sentinel_policy_is_none(self) -> None:
        # Contract must preserve None from R38-T046
        self.assertIsNone(UNDEFINED_SENTINEL)

    def test_q3_nonzero_vectors_return_valid_numeric_floats(self) -> None:
        result = cosine_similarity([1.0, 2.0], [2.0, 4.0])
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_q3_does_not_invent_numeric_score_for_zero_vectors(self) -> None:
        # Must NOT return 0.0 or 1.0 or any numeric value
        result = cosine_similarity([0.0, 0.0], [1.0, 0.0])
        self.assertNotIsInstance(result, float)
        self.assertNotIsInstance(result, int)
        self.assertIsNone(result)


class TestInputValidation(unittest.TestCase):
    """Test validation of dimensions, types, and numeric sanity."""

    def test_dimension_mismatch_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vector_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_non_sequence_input_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            cosine_similarity(123, [1.0, 2.0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity("ab", "cd")  # type: ignore

    def test_boolean_elements_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            cosine_similarity([True, 0.0], [1.0, 0.0])  # type: ignore

    def test_non_numeric_elements_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            cosine_similarity(["a", 0.0], [1.0, 0.0])  # type: ignore

    def test_nan_element_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan"), 0.0], [1.0, 0.0])

    def test_inf_element_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([float("inf"), 0.0], [1.0, 0.0])

    def test_valid_negative_components_accepted(self) -> None:
        result = cosine_similarity([-2.0, -4.0], [-1.0, -2.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_tuple_sequences_accepted(self) -> None:
        result = cosine_similarity((1.0, 0.0), (0.0, 1.0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=9)


class TestAssessmentIntegrity(unittest.TestCase):
    """Verify that all three scenario assessments are complete and verified."""

    def test_run_all_scenario_assessments_count_and_types(self) -> None:
        assessments = run_all_scenario_assessments()
        self.assertEqual(len(assessments), 3)

        expected_types = ["predict_output", "diagnose_failure", "choose_boundary_policy"]
        for a, exp_type in zip(assessments, expected_types):
            self.assertEqual(a.question_type, exp_type)
            self.assertTrue(a.check_passed)
            self.assertTrue(len(a.question_text) > 0)
            self.assertTrue(len(a.rationale) > 0)
            self.assertTrue(len(a.expected_answer) > 0)


if __name__ == "__main__":
    unittest.main()

