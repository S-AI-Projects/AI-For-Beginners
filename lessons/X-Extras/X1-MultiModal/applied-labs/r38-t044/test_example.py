"""Automated tests for Binary Classification Evaluation (R38-T044)."""

from __future__ import annotations

import unittest

from example import (
    MAIN_Y_PRED,
    MAIN_Y_TRUE,
    ZERO_DENOM_Y_PRED,
    ZERO_DENOM_Y_TRUE,
    evaluate_classification,
)


class TestMainFixture(unittest.TestCase):
    """Verify the primary worked-example fixture."""

    def setUp(self) -> None:
        self.res = evaluate_classification(MAIN_Y_TRUE, MAIN_Y_PRED)

    def test_tp_equals_1(self) -> None:
        self.assertEqual(self.res.tp, 1)

    def test_fn_equals_1(self) -> None:
        self.assertEqual(self.res.fn, 1)

    def test_fp_equals_1(self) -> None:
        self.assertEqual(self.res.fp, 1)

    def test_tn_equals_1(self) -> None:
        self.assertEqual(self.res.tn, 1)

    def test_precision_equals_0_5(self) -> None:
        self.assertAlmostEqual(self.res.precision, 0.5)

    def test_recall_equals_0_5(self) -> None:
        self.assertAlmostEqual(self.res.recall, 0.5)

    def test_confusion_matrix_sums_to_n_samples(self) -> None:
        total = self.res.tp + self.res.fn + self.res.fp + self.res.tn
        self.assertEqual(total, self.res.n_samples)
        self.assertEqual(total, 4)


class TestZeroDenominatorFixture(unittest.TestCase):
    """Verify the no-positive-predictions / zero-denominator fixture."""

    def setUp(self) -> None:
        self.res = evaluate_classification(ZERO_DENOM_Y_TRUE, ZERO_DENOM_Y_PRED)

    def test_tp_equals_0(self) -> None:
        self.assertEqual(self.res.tp, 0)

    def test_fn_equals_2(self) -> None:
        self.assertEqual(self.res.fn, 2)

    def test_fp_equals_0(self) -> None:
        self.assertEqual(self.res.fp, 0)

    def test_tn_equals_2(self) -> None:
        self.assertEqual(self.res.tn, 2)

    def test_precision_is_0_not_exception(self) -> None:
        """Precision must be 0.0 (not NaN, not Inf, no exception)."""
        self.assertAlmostEqual(self.res.precision, 0.0)

    def test_recall_is_0_not_exception(self) -> None:
        """Recall must be 0.0 (not NaN, not Inf, no exception)."""
        self.assertAlmostEqual(self.res.recall, 0.0)

    def test_no_division_by_zero(self) -> None:
        """Must not raise ZeroDivisionError."""
        try:
            evaluate_classification(ZERO_DENOM_Y_TRUE, ZERO_DENOM_Y_PRED)
        except ZeroDivisionError:
            self.fail("evaluate_classification raised ZeroDivisionError")


class TestAllCorrectPredictions(unittest.TestCase):
    """Every prediction matches the actual label."""

    def test_all_correct(self) -> None:
        res = evaluate_classification([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(res.tp, 2)
        self.assertEqual(res.tn, 2)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 0)
        self.assertAlmostEqual(res.precision, 1.0)
        self.assertAlmostEqual(res.recall, 1.0)


class TestAllIncorrectPredictions(unittest.TestCase):
    """Every prediction is the opposite of the actual label."""

    def test_all_incorrect(self) -> None:
        res = evaluate_classification([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fn, 2)
        self.assertEqual(res.fp, 2)
        self.assertEqual(res.tn, 0)
        self.assertAlmostEqual(res.precision, 0.0)
        self.assertAlmostEqual(res.recall, 0.0)


class TestAllPositiveActualLabels(unittest.TestCase):
    """All actual labels are 1."""

    def test_all_positive(self) -> None:
        res = evaluate_classification([1, 1, 1, 1], [1, 0, 1, 0])
        self.assertEqual(res.tp, 2)
        self.assertEqual(res.fn, 2)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.tn, 0)
        self.assertAlmostEqual(res.precision, 1.0)
        self.assertAlmostEqual(res.recall, 0.5)


class TestAllNegativeActualLabels(unittest.TestCase):
    """All actual labels are 0."""

    def test_all_negative(self) -> None:
        res = evaluate_classification([0, 0, 0, 0], [0, 1, 0, 0])
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fn, 0)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.tn, 3)
        self.assertAlmostEqual(res.precision, 0.0)

    def test_no_actual_positives_recall_zero(self) -> None:
        """When TP + FN == 0, recall = 0.0 by policy."""
        res = evaluate_classification([0, 0, 0], [0, 0, 0])
        self.assertAlmostEqual(res.recall, 0.0)


class TestPrecisionRecallBounds(unittest.TestCase):
    """Precision and recall must always be in [0, 1]."""

    def test_metrics_in_range(self) -> None:
        fixtures = [
            ([1, 1, 0, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 0]),
            ([1, 0, 1, 0], [1, 0, 1, 0]),
            ([0, 0, 0, 0], [1, 1, 1, 1]),
            ([1, 1, 1, 1], [1, 1, 1, 1]),
            ([0, 0], [0, 0]),
        ]
        for y_t, y_p in fixtures:
            with self.subTest(y_true=y_t, y_pred=y_p):
                res = evaluate_classification(y_t, y_p)
                self.assertGreaterEqual(res.precision, 0.0)
                self.assertLessEqual(res.precision, 1.0)
                self.assertGreaterEqual(res.recall, 0.0)
                self.assertLessEqual(res.recall, 1.0)


class TestConfusionMatrixSumInvariant(unittest.TestCase):
    """TP + FN + FP + TN must always equal the number of samples."""

    def test_sum_invariant(self) -> None:
        fixtures = [
            ([1], [0]),
            ([1, 0], [1, 0]),
            ([1, 1, 0, 0], [1, 0, 1, 0]),
            ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
        ]
        for y_t, y_p in fixtures:
            with self.subTest(y_true=y_t, y_pred=y_p):
                res = evaluate_classification(y_t, y_p)
                total = res.tp + res.fn + res.fp + res.tn
                self.assertEqual(total, res.n_samples)
                self.assertEqual(total, len(y_t))


class TestInputValidation(unittest.TestCase):
    """Reject invalid inputs with meaningful errors."""

    def test_empty_lists_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_classification([], [])

    def test_unequal_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_classification([1, 0], [1])
        with self.assertRaises(ValueError):
            evaluate_classification([1], [1, 0, 0])

    def test_invalid_label_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_classification([1, 2], [1, 0])
        with self.assertRaises(ValueError):
            evaluate_classification([1, 0], [1, -1])

    def test_non_integer_labels_rejected(self) -> None:
        with self.assertRaises(TypeError):
            evaluate_classification([1.0, 0.0], [1, 0])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            evaluate_classification([1, 0], ["1", "0"])  # type: ignore[list-item]

    def test_boolean_labels_rejected(self) -> None:
        """Booleans are not accepted even though bool is a subclass of int."""
        with self.assertRaises(TypeError):
            evaluate_classification([True, False], [True, False])  # type: ignore[list-item]

    def test_non_sequence_input_rejected(self) -> None:
        with self.assertRaises(TypeError):
            evaluate_classification("1100", [1, 0, 1, 0])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            evaluate_classification([1, 0], 10)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

