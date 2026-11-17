"""Unit tests for Token-Budget Preflight — Failure and Recovery Example (R38-T048)."""

from __future__ import annotations

import unittest

from example import (
    CASE1_BUDGET,
    CASE1_TEXT,
    CASE2_BUDGET,
    CASE2_TEXT,
    CASE3_BUDGET,
    CASE3_TEXT,
    TokenPreflightResult,
    demonstrate_failure_and_recovery,
    preflight_text,
)


class TestNormalPreflightBehavior(unittest.TestCase):
    """Test standard preflight token truncation and boundary behaviors."""

    def test_case1_normal_truncation(self) -> None:
        """Case 1: 'one two three four' with budget 3 retains 3 tokens, 1 removed, truncated=True."""
        res = preflight_text(CASE1_TEXT, CASE1_BUDGET)
        self.assertEqual(res.original_token_count, 4)
        self.assertEqual(res.retained_tokens, ["one", "two", "three"])
        self.assertEqual(res.retained_count, 3)
        self.assertEqual(res.removed_count, 1)
        self.assertTrue(res.is_truncated)
        self.assertEqual(res.original_tokens, ["one", "two", "three", "four"])

    def test_budget_equal_to_token_count(self) -> None:
        """Budget equal to token count retains all tokens without truncation."""
        res = preflight_text("apple banana orange", 3)
        self.assertEqual(res.original_token_count, 3)
        self.assertEqual(res.retained_tokens, ["apple", "banana", "orange"])
        self.assertEqual(res.retained_count, 3)
        self.assertEqual(res.removed_count, 0)
        self.assertFalse(res.is_truncated)

    def test_budget_greater_than_token_count(self) -> None:
        """Budget greater than token count retains all tokens without truncation."""
        res = preflight_text("hello world", 10)
        self.assertEqual(res.original_token_count, 2)
        self.assertEqual(res.retained_tokens, ["hello", "world"])
        self.assertEqual(res.retained_count, 2)
        self.assertEqual(res.removed_count, 0)
        self.assertFalse(res.is_truncated)

    def test_budget_zero(self) -> None:
        """Budget 0 retains no tokens, all original tokens counted as removed."""
        res = preflight_text("first second third", 0)
        self.assertEqual(res.original_token_count, 3)
        self.assertEqual(res.retained_tokens, [])
        self.assertEqual(res.retained_count, 0)
        self.assertEqual(res.removed_count, 3)
        self.assertTrue(res.is_truncated)


class TestEmptyAndWhitespaceInputs(unittest.TestCase):
    """Test handling of empty and whitespace-only strings."""

    def test_case2_empty_string(self) -> None:
        """Case 2: Empty string has 0 tokens, 0 retained, 0 removed, truncated=False."""
        res = preflight_text(CASE2_TEXT, CASE2_BUDGET)
        self.assertEqual(res.original_token_count, 0)
        self.assertEqual(res.retained_tokens, [])
        self.assertEqual(res.retained_count, 0)
        self.assertEqual(res.removed_count, 0)
        self.assertFalse(res.is_truncated)

    def test_whitespace_only_string(self) -> None:
        """Whitespace-only string produces 0 tokens without error."""
        res = preflight_text("   \t  \n   ", 5)
        self.assertEqual(res.original_token_count, 0)
        self.assertEqual(res.retained_tokens, [])
        self.assertEqual(res.retained_count, 0)
        self.assertEqual(res.removed_count, 0)
        self.assertFalse(res.is_truncated)


class TestInvalidBudgetsAndExceptions(unittest.TestCase):
    """Test rejection of negative budgets, booleans, and non-integer types."""

    def test_case3_negative_budget_raises_value_error(self) -> None:
        """Case 3: Negative budget must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            preflight_text(CASE3_TEXT, CASE3_BUDGET)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_boolean_budget_rejected(self) -> None:
        """Booleans must be rejected with TypeError even though bool is a subclass of int."""
        with self.assertRaises(TypeError):
            preflight_text("some text", True)  # type: ignore
        with self.assertRaises(TypeError):
            preflight_text("some text", False)  # type: ignore

    def test_float_budget_rejected(self) -> None:
        """Float budget must raise TypeError."""
        with self.assertRaises(TypeError):
            preflight_text("some text", 2.5)  # type: ignore

    def test_string_budget_rejected(self) -> None:
        """String budget must raise TypeError."""
        with self.assertRaises(TypeError):
            preflight_text("some text", "3")  # type: ignore


class TestTextInputValidation(unittest.TestCase):
    """Test validation of text input types and irregular whitespace delimiters."""

    def test_non_string_text_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            preflight_text(12345, 5)  # type: ignore
        with self.assertRaises(TypeError):
            preflight_text(["one", "two"], 5)  # type: ignore
        with self.assertRaises(TypeError):
            preflight_text(None, 5)  # type: ignore

    def test_multiple_consecutive_spaces(self) -> None:
        res = preflight_text("alpha    beta     gamma", 2)
        self.assertEqual(res.original_token_count, 3)
        self.assertEqual(res.retained_tokens, ["alpha", "beta"])
        self.assertEqual(res.removed_count, 1)

    def test_leading_and_trailing_whitespace(self) -> None:
        res = preflight_text("   one two   ", 2)
        self.assertEqual(res.original_tokens, ["one", "two"])
        self.assertEqual(res.retained_tokens, ["one", "two"])

    def test_mixed_whitespace_newlines_tabs(self) -> None:
        res = preflight_text("one\ttwo\nthree\r\nfour", 2)
        self.assertEqual(res.original_token_count, 4)
        self.assertEqual(res.retained_tokens, ["one", "two"])
        self.assertEqual(res.removed_count, 2)


class TestFailureAndRecovery(unittest.TestCase):
    """Test failure on an invalid request and clean recovery on subsequent valid calls."""

    def test_demonstrate_failure_and_recovery_sequence(self) -> None:
        error_msg, recovery_res = demonstrate_failure_and_recovery()
        self.assertIn("cannot be negative", error_msg)
        self.assertEqual(recovery_res.original_token_count, 4)
        self.assertEqual(recovery_res.retained_tokens, ["one", "two", "three"])
        self.assertEqual(recovery_res.retained_count, 3)
        self.assertEqual(recovery_res.removed_count, 1)
        self.assertTrue(recovery_res.is_truncated)

    def test_consecutive_failures_do_not_corrupt_subsequent_success(self) -> None:
        # Multiple failures
        for bad_budget in [-1, -5, -100]:
            with self.assertRaises(ValueError):
                preflight_text("test string", bad_budget)

        # Subsequent success is completely unaffected
        res = preflight_text("clean recovery test", 2)
        self.assertEqual(res.retained_tokens, ["clean", "recovery"])
        self.assertEqual(res.removed_count, 1)


class TestMathematicalInvariantsAndOrdering(unittest.TestCase):
    """Test preservation of mathematical invariants and token ordering."""

    def test_sum_invariant(self) -> None:
        """retained_count + removed_count == original_token_count across all inputs."""
        test_cases = [
            ("a b c d e", 2),
            ("a b c d e", 5),
            ("a b c d e", 10),
            ("a b c d e", 0),
            ("", 3),
            ("single", 1),
        ]
        for text, budget in test_cases:
            res = preflight_text(text, budget)
            self.assertEqual(
                res.retained_count + res.removed_count,
                res.original_token_count,
                f"Sum invariant failed for text={text!r}, budget={budget}",
            )

    def test_budget_upper_bound_invariant(self) -> None:
        """retained_count must never exceed budget."""
        for budget in range(6):
            res = preflight_text("one two three four five", budget)
            self.assertLessEqual(res.retained_count, budget)

    def test_order_preservation(self) -> None:
        """Retained tokens must strictly maintain original sequential ordering."""
        text = "first second third fourth fifth"
        res = preflight_text(text, 3)
        self.assertEqual(res.retained_tokens, ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()

