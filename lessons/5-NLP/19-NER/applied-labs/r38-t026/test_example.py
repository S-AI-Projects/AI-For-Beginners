"""
Unit tests for token-budget preflight and refactoring equivalence (Issue #26 / R38-T026).

Tests:
1. Truncation and count reporting accuracy.
2. Boundary cases: empty input, zero budget, exact budget matches, under-budget.
3. Negative budget rejection (ValueError).
4. Type validation (TypeError).
5. Strict before-and-after functional equivalence between baseline and refactored versions.
"""

import unittest
from example import (
    PreflightReport,
    TokenTruncationPolicy,
    token_budget_preflight,
    token_budget_preflight_baseline,
    whitespace_tokenize,
)


class TestTokenBudgetPreflightRefactoring(unittest.TestCase):
    def test_truncation_reporting(self):
        """Truncate prompt according to budget, reporting retained tokens and truncated count."""
        prompt = "one two three four five"
        budget = 3
        report = token_budget_preflight(prompt, budget)

        self.assertEqual(report.retained_tokens, ["one", "two", "three"])
        self.assertEqual(report.truncated_count, 2)
        self.assertEqual(report.original_count, 5)
        self.assertEqual(report.budget, 3)

    def test_empty_input_handling(self):
        """Empty input yields empty retained tokens and 0 truncated count."""
        report = token_budget_preflight("", 5)
        self.assertEqual(report.retained_tokens, [])
        self.assertEqual(report.truncated_count, 0)
        self.assertEqual(report.original_count, 0)

        # Whitespace only
        report_ws = token_budget_preflight("   \n\t  ", 5)
        self.assertEqual(report_ws.retained_tokens, [])
        self.assertEqual(report_ws.truncated_count, 0)
        self.assertEqual(report_ws.original_count, 0)

    def test_negative_budget_rejection(self):
        """Negative budgets must be rejected with ValueError in both implementations."""
        with self.assertRaises(ValueError):
            token_budget_preflight("hello world", -1)

        with self.assertRaises(ValueError):
            token_budget_preflight_baseline("hello world", -1)

        with self.assertRaises(ValueError):
            TokenTruncationPolicy(-5)

    def test_zero_budget_boundary(self):
        """Zero budget retains 0 tokens and reports all tokens as truncated."""
        report = token_budget_preflight("alpha beta gamma", 0)
        self.assertEqual(report.retained_tokens, [])
        self.assertEqual(report.truncated_count, 3)
        self.assertEqual(report.original_count, 3)

    def test_exact_budget_match(self):
        """Exact budget matches retain all tokens with 0 truncated count."""
        report = token_budget_preflight("alpha beta gamma", 3)
        self.assertEqual(report.retained_tokens, ["alpha", "beta", "gamma"])
        self.assertEqual(report.truncated_count, 0)
        self.assertEqual(report.original_count, 3)

    def test_under_budget(self):
        """Prompt shorter than budget retains all tokens with 0 truncated count."""
        report = token_budget_preflight("alpha beta", 5)
        self.assertEqual(report.retained_tokens, ["alpha", "beta"])
        self.assertEqual(report.truncated_count, 0)
        self.assertEqual(report.original_count, 2)

    def test_type_validation(self):
        """Type validation for prompt and budget parameters."""
        with self.assertRaises(TypeError):
            token_budget_preflight(12345, 5)  # type: ignore

        with self.assertRaises(TypeError):
            token_budget_preflight("valid", "five")  # type: ignore

        with self.assertRaises(TypeError):
            token_budget_preflight("valid", True)  # type: ignore

        with self.assertRaises(TypeError):
            token_budget_preflight("valid", 3.14)  # type: ignore

    def test_strict_before_and_after_functional_equivalence(self):
        """
        Verify that the refactored implementation (extracting TokenTruncationPolicy)
        behaves with 100% functional equivalence to the baseline across varied scenarios.
        """
        test_cases = [
            ("single", 1),
            ("single", 0),
            ("single", 2),
            ("Tricuspid valve regurgitation and lithium carbonate toxicity", 4),
            ("Tricuspid valve regurgitation and lithium carbonate toxicity", 8),
            ("Tricuspid valve regurgitation and lithium carbonate toxicity", 0),
            ("Tricuspid valve regurgitation and lithium carbonate toxicity", 100),
            ("", 0),
            ("", 5),
            ("word " * 50, 10),
            ("word " * 50, 50),
            ("word " * 50, 60),
        ]

        for prompt, budget in test_cases:
            with self.subTest(prompt=prompt[:20], budget=budget):
                baseline_res = token_budget_preflight_baseline(prompt, budget)
                refactored_res = token_budget_preflight(prompt, budget)
                self.assertEqual(
                    baseline_res,
                    refactored_res,
                    f"Equivalence mismatch for prompt='{prompt[:20]}...', budget={budget}"
                )


if __name__ == "__main__":
    unittest.main()
