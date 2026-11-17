"""Unit tests for Single Perceptron Update with Offline Service Adapter (R38-T045)."""

from __future__ import annotations

import math
import unittest
from typing import Sequence

from example import (
    MAIN_INITIAL_BIAS,
    MAIN_INITIAL_WEIGHTS,
    MAIN_INPUTS,
    MAIN_LEARNING_RATE,
    MAIN_TARGET,
    PerceptronStepResult,
    calculate_weighted_sum,
    local_deterministic_predictor,
    perceptron_update_step,
)


class TestMainRegressionFixture(unittest.TestCase):
    """Test the mandatory regression fixture from the issue specification."""

    def setUp(self) -> None:
        self.result: PerceptronStepResult = perceptron_update_step(
            weights=MAIN_INITIAL_WEIGHTS,
            bias=MAIN_INITIAL_BIAS,
            inputs=MAIN_INPUTS,
            target=MAIN_TARGET,
            learning_rate=MAIN_LEARNING_RATE,
            predictor=local_deterministic_predictor,
        )

    def test_weighted_sum_is_zero(self) -> None:
        self.assertEqual(self.result.weighted_sum, 0.0)

    def test_initial_prediction_is_zero(self) -> None:
        self.assertEqual(self.result.initial_prediction, 0)

    def test_error_is_one(self) -> None:
        # error = target - prediction = 1 - 0 = 1
        self.assertEqual(self.result.error, 1)

    def test_updated_weights(self) -> None:
        # new_w = [0 + 1*1*1, 0 + 1*1*2] = [1.0, 2.0]
        self.assertEqual(self.result.updated_weights, [1.0, 2.0])

    def test_updated_bias(self) -> None:
        # new_b = 0 + 1*1 = 1.0
        self.assertEqual(self.result.updated_bias, 1.0)

    def test_result_structure_contains_all_initial_values(self) -> None:
        self.assertEqual(self.result.initial_weights, [0.0, 0.0])
        self.assertEqual(self.result.initial_bias, 0.0)
        self.assertEqual(self.result.inputs, [1.0, 2.0])
        self.assertEqual(self.result.target, 1)
        self.assertEqual(self.result.learning_rate, 1.0)


class TestZeroScoreTieRule(unittest.TestCase):
    """Explicitly verify the documented zero-score tie resolution rule."""

    def test_zero_weighted_sum_predicts_zero(self) -> None:
        """A weighted sum of exactly zero must predict class 0."""
        pred = local_deterministic_predictor([0.0, 0.0], 0.0, [1.0, 2.0])
        self.assertEqual(pred, 0)

    def test_canceling_terms_sum_to_zero_predicts_zero(self) -> None:
        """When positive and negative terms cancel to 0.0, predict 0."""
        # z = (1.0 * 3.0) + (-1.0 * 3.0) + 0.0 = 0.0
        z = calculate_weighted_sum([1.0, -1.0], 0.0, [3.0, 3.0])
        self.assertEqual(z, 0.0)
        pred = local_deterministic_predictor([1.0, -1.0], 0.0, [3.0, 3.0])
        self.assertEqual(pred, 0)

    def test_epsilon_positive_predicts_one(self) -> None:
        """Any score strictly greater than zero must predict class 1."""
        pred = local_deterministic_predictor([0.0], 1e-9, [1.0])
        self.assertEqual(pred, 1)

    def test_epsilon_negative_predicts_zero(self) -> None:
        """Any score strictly less than zero must predict class 0."""
        pred = local_deterministic_predictor([0.0], -1e-9, [1.0])
        self.assertEqual(pred, 0)


class TestOfflineServiceAdapter(unittest.TestCase):
    """Verify that the local deterministic adapter operates reliably offline."""

    def test_deterministic_reproducibility(self) -> None:
        """Identical inputs must always produce identical predictions."""
        for _ in range(20):
            pred = local_deterministic_predictor([0.5, -0.2], 0.1, [2.0, 1.0])
            self.assertEqual(pred, 1)

    def test_adapter_works_without_network(self) -> None:
        """Adapter requires only CPU in-memory arithmetic without network dependencies."""
        pred1 = local_deterministic_predictor([1.0], 0.0, [1.0])
        pred0 = local_deterministic_predictor([-1.0], 0.0, [1.0])
        self.assertEqual(pred1, 1)
        self.assertEqual(pred0, 0)


class TestCallableInterfaceAndDependencyInjection(unittest.TestCase):
    """Verify that perceptron_update_step accepts and delegates to injected callables."""

    def test_custom_injected_callable_is_called_with_arguments(self) -> None:
        """Verify the update function queries the injected predictor with expected arguments."""
        calls = []

        def mock_predictor(weights: Sequence[float], bias: float, inputs: Sequence[float]) -> int:
            calls.append((list(weights), bias, list(inputs)))
            return 1

        result = perceptron_update_step(
            weights=[10.0, 20.0],
            bias=5.0,
            inputs=[1.0, 2.0],
            target=1,
            learning_rate=0.1,
            predictor=mock_predictor,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ([10.0, 20.0], 5.0, [1.0, 2.0]))
        self.assertEqual(result.initial_prediction, 1)
        self.assertEqual(result.error, 0)
        self.assertEqual(result.updated_weights, [10.0, 20.0])

    def test_substitute_different_callable_changes_behaviour(self) -> None:
        """A different callable returning 0 forces an update step."""
        def always_zero(weights: Sequence[float], bias: float, inputs: Sequence[float]) -> int:
            return 0

        result = perceptron_update_step(
            weights=[0.0, 0.0],
            bias=0.0,
            inputs=[1.0, 1.0],
            target=1,
            learning_rate=0.5,
            predictor=always_zero,
        )

        self.assertEqual(result.initial_prediction, 0)
        self.assertEqual(result.error, 1)
        self.assertEqual(result.updated_weights, [0.5, 0.5])
        self.assertEqual(result.updated_bias, 0.5)


class TestImmutabilityAndNoSideEffects(unittest.TestCase):
    """Verify input collections are not mutated in-place."""

    def test_weights_list_is_not_mutated(self) -> None:
        original_weights = [0.0, 0.0]
        original_inputs = [1.0, 2.0]

        _ = perceptron_update_step(
            weights=original_weights,
            bias=0.0,
            inputs=original_inputs,
            target=1,
            learning_rate=1.0,
            predictor=local_deterministic_predictor,
        )

        self.assertEqual(original_weights, [0.0, 0.0])
        self.assertEqual(original_inputs, [1.0, 2.0])


class TestBoundaryScenarios(unittest.TestCase):
    """Test edge cases such as zero learning rate, zero inputs, and matching predictions."""

    def test_zero_learning_rate(self) -> None:
        """With learning rate 0.0, weights and bias must not change even on non-zero error."""
        result = perceptron_update_step(
            weights=[5.0, -3.0],
            bias=2.0,
            inputs=[1.0, 2.0],
            target=1,
            learning_rate=0.0,
            predictor=local_deterministic_predictor,
        )
        self.assertEqual(result.learning_rate, 0.0)
        self.assertEqual(result.updated_weights, [5.0, -3.0])
        self.assertEqual(result.updated_bias, 2.0)

    def test_zero_inputs(self) -> None:
        """When inputs are 0, weights remain unchanged, but bias is updated."""
        result = perceptron_update_step(
            weights=[3.0, 4.0],
            bias=0.0,
            inputs=[0.0, 0.0],
            target=1,
            learning_rate=1.0,
            predictor=local_deterministic_predictor,
        )
        # z = 0 -> pred = 0 -> error = 1 - 0 = 1
        # w_new = w + 1*1*0 = w
        # b_new = 0 + 1*1 = 1
        self.assertEqual(result.updated_weights, [3.0, 4.0])
        self.assertEqual(result.updated_bias, 1.0)

    def test_zero_error_no_updates(self) -> None:
        """When prediction matches target, error is 0 and no weights change."""
        result = perceptron_update_step(
            weights=[2.0, 2.0],
            bias=1.0,
            inputs=[1.0, 1.0],
            target=1,
            learning_rate=1.0,
            predictor=local_deterministic_predictor,
        )
        self.assertEqual(result.error, 0)
        self.assertEqual(result.updated_weights, [2.0, 2.0])
        self.assertEqual(result.updated_bias, 1.0)


class TestInputValidation(unittest.TestCase):
    """Test validation of all inputs, types, and dimensions."""

    def test_non_callable_predictor_rejected(self) -> None:
        with self.assertRaises(TypeError):
            perceptron_update_step([0.0], 0.0, [1.0], 1, 1.0, predictor="not_callable")  # type: ignore

    def test_dimension_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([0.0, 0.0], 0.0, [1.0], 1, 1.0, local_deterministic_predictor)

    def test_empty_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([], 0.0, [], 1, 1.0, local_deterministic_predictor)

    def test_negative_learning_rate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([0.0], 0.0, [1.0], 1, -0.5, local_deterministic_predictor)

    def test_non_numeric_learning_rate_rejected(self) -> None:
        with self.assertRaises(TypeError):
            perceptron_update_step([0.0], 0.0, [1.0], 1, "fast", local_deterministic_predictor)  # type: ignore

    def test_boolean_target_rejected(self) -> None:
        with self.assertRaises(TypeError):
            perceptron_update_step([0.0], 0.0, [1.0], True, 1.0, local_deterministic_predictor)  # type: ignore

    def test_non_binary_target_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([0.0], 0.0, [1.0], 2, 1.0, local_deterministic_predictor)

    def test_boolean_weights_rejected(self) -> None:
        with self.assertRaises(TypeError):
            perceptron_update_step([True, False], 0.0, [1.0, 2.0], 1, 1.0, local_deterministic_predictor)  # type: ignore

    def test_nan_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([float("nan"), 0.0], 0.0, [1.0, 2.0], 1, 1.0, local_deterministic_predictor)

    def test_inf_input_rejected(self) -> None:
        with self.assertRaises(ValueError):
            perceptron_update_step([0.0, 0.0], 0.0, [float("inf"), 2.0], 1, 1.0, local_deterministic_predictor)

    def test_predictor_returning_non_binary_rejected(self) -> None:
        def bad_predictor(w: Sequence[float], b: float, x: Sequence[float]) -> int:
            return 99

        with self.assertRaises(ValueError):
            perceptron_update_step([0.0], 0.0, [1.0], 1, 1.0, bad_predictor)

    def test_predictor_returning_boolean_rejected(self) -> None:
        def bool_predictor(w: Sequence[float], b: float, x: Sequence[float]) -> int:
            return True  # type: ignore

        with self.assertRaises(TypeError):
            perceptron_update_step([0.0], 0.0, [1.0], 1, 1.0, bool_predictor)


if __name__ == "__main__":
    unittest.main()

