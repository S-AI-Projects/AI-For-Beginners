"""Unit tests for Cosine-Similarity Lookup — Evaluation Experiment (R38-T046)."""

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
    cosine_similarity,
    dot_product,
    run_all_experiments,
    run_experiment,
    vector_norm,
)


class TestMandatoryFixtures(unittest.TestCase):
    """Test the three mandatory specification cases."""

    def test_case1_identical_vectors(self) -> None:
        """Case 1: A=[1, 0], B=[1, 0] must return 1.0."""
        result = cosine_similarity(CASE1_A, CASE1_B)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_case2_orthogonal_vectors(self) -> None:
        """Case 2: A=[1, 0], B=[0, 1] must return 0.0."""
        result = cosine_similarity(CASE2_A, CASE2_B)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_case3_zero_vector_returns_sentinel(self) -> None:
        """Case 3: A=[0, 0], B=[1, 0] must return UNDEFINED_SENTINEL (None)."""
        result = cosine_similarity(CASE3_A, CASE3_B)
        self.assertIs(result, UNDEFINED_SENTINEL)
        self.assertIsNone(result)


class TestCoreMathematicalCorrectness(unittest.TestCase):
    """Test geometric vector properties and bounds."""

    def test_opposite_direction_returns_negative_one(self) -> None:
        """Vectors pointing in diametrically opposite directions return -1.0."""
        result = cosine_similarity([2.0, 0.0], [-5.0, 0.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, -1.0, places=9)

    def test_known_nontrivial_acute_angle(self) -> None:
        """A=[3, 4], B=[4, 3] -> dot=24, norms=5,5 -> 24/25 = 0.96."""
        result = cosine_similarity([3.0, 4.0], [4.0, 3.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.96, places=9)

    def test_scaled_collinear_vectors(self) -> None:
        """Vectors of different lengths in the same direction have cosine similarity 1.0."""
        result = cosine_similarity([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_range_bounds_invariant(self) -> None:
        """All valid cosine similarities must fall strictly within [-1.0, 1.0]."""
        test_pairs = [
            ([1.0, 0.0], [1.0, 1.0]),
            ([5.0, -2.0, 1.0], [-1.0, 3.0, 4.0]),
            ([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, -1.0, 2.0]),
        ]
        for a, b in test_pairs:
            sim = cosine_similarity(a, b)
            self.assertIsNotNone(sim)
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)

    def test_symmetry_property(self) -> None:
        """Cosine similarity is commutative: sim(A, B) == sim(B, A)."""
        a = [1.0, 3.0, -2.0]
        b = [4.0, -1.0, 2.0]
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a), places=9)


class TestZeroVectorHandling(unittest.TestCase):
    """Explicitly verify zero-magnitude vectors return the documented sentinel."""

    def test_first_vector_zero(self) -> None:
        self.assertIsNone(cosine_similarity([0.0, 0.0], [3.0, 4.0]))

    def test_second_vector_zero(self) -> None:
        self.assertIsNone(cosine_similarity([5.0, 1.0], [0.0, 0.0]))

    def test_both_vectors_zero(self) -> None:
        self.assertIsNone(cosine_similarity([0.0, 0.0], [0.0, 0.0]))

    def test_higher_dimensional_zero_vector(self) -> None:
        zero_4d = [0.0, 0.0, 0.0, 0.0]
        vec_4d = [1.0, 2.0, 3.0, 4.0]
        self.assertIsNone(cosine_similarity(zero_4d, vec_4d))
        self.assertIsNone(cosine_similarity(vec_4d, zero_4d))


class TestInputValidation(unittest.TestCase):
    """Test validation of dimensions, sequence types, and values."""

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
            cosine_similarity("12", "34")  # type: ignore

    def test_boolean_elements_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            cosine_similarity([True, 1.0], [1.0, 0.0])  # type: ignore

    def test_non_numeric_elements_raise_type_error(self) -> None:
        with self.assertRaises(TypeError):
            cosine_similarity(["a", 1.0], [1.0, 0.0])  # type: ignore

    def test_nan_element_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan"), 1.0], [1.0, 0.0])

    def test_inf_element_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([float("inf"), 1.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [float("-inf"), 1.0])

    def test_valid_negative_components_accepted(self) -> None:
        sim = cosine_similarity([-1.0, -2.0], [-1.0, -2.0])
        self.assertAlmostEqual(sim, 1.0, places=9)

    def test_tuple_sequences_accepted(self) -> None:
        sim = cosine_similarity((1.0, 0.0), (0.0, 1.0))
        self.assertAlmostEqual(sim, 0.0, places=9)


class TestNoMutation(unittest.TestCase):
    """Verify input vectors are never modified."""

    def test_inputs_unmodified(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        _ = cosine_similarity(a, b)
        self.assertEqual(a, [1.0, 2.0, 3.0])
        self.assertEqual(b, [4.0, 5.0, 6.0])


class TestControlledEvaluationExperiments(unittest.TestCase):
    """Verify the execution and correctness of the three controlled evaluation configurations."""

    def test_experiment_1_2d_config(self) -> None:
        # A=[3,4], B=[4,3] -> expected = 24/25 = 0.96
        res = run_experiment("2D Test", [3.0, 4.0], [4.0, 3.0], 0.96, iterations=100)
        self.assertEqual(res.dimension, 2)
        self.assertTrue(res.is_correct)
        self.assertAlmostEqual(res.actual_similarity, 0.96, places=9)

    def test_experiment_2_4d_config(self) -> None:
        # A=[1,2,3,4], B=[4,3,2,1] -> expected = 20/30 = 2/3
        expected = 20.0 / 30.0
        res = run_experiment("4D Test", [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], expected, iterations=100)
        self.assertEqual(res.dimension, 4)
        self.assertTrue(res.is_correct)
        self.assertAlmostEqual(res.actual_similarity, expected, places=9)

    def test_experiment_3_8d_config(self) -> None:
        # A and B orthogonal blocks -> expected = 0.0
        a = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        res = run_experiment("8D Test", a, b, 0.0, iterations=100)
        self.assertEqual(res.dimension, 8)
        self.assertTrue(res.is_correct)
        self.assertAlmostEqual(res.actual_similarity, 0.0, places=9)

    def test_run_all_experiments_returns_three_correct_results(self) -> None:
        results = run_all_experiments(iterations=100)
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertTrue(res.is_correct)
            self.assertGreater(res.dimension, 0)
            self.assertGreater(res.total_time_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

