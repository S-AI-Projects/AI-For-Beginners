"""
Unit tests for token-budget preflight (Issue #22 / R38-T022).

Explicitly tests boundary conditions:
- 0-budget
- Exact budget matches
- Empty string / whitespace-only inputs
- Boundary overflows
- Negative budget rejection
- Type checking
"""

import unittest
from example import (
    PreflightResult,
    token_budget_preflight,
    whitespace_tokenizer,
)


class TestTokenBudgetPreflight(unittest.TestCase):
    def test_example_scenario_overflow(self):
        """Budget 3 with 'one two three four' retains ['one','two','three'] and reports 1 removed token."""
        result = token_budget_preflight("one two three four", 3)
        self.assertEqual(result.retained_tokens, ["one", "two", "three"])
        self.assertEqual(result.truncated_count, 1)
        self.assertEqual(result.original_count, 4)
        self.assertEqual(result.budget, 3)

    def test_empty_input_retains_none(self):
        """Empty input string retains no tokens and reports 0 truncated tokens."""
        result = token_budget_preflight("", 5)
        self.assertEqual(result.retained_tokens, [])
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(result.original_count, 0)
        self.assertEqual(result.budget, 5)

    def test_whitespace_only_input_retains_none(self):
        """Whitespace-only input string produces no tokens and reports 0 truncated tokens."""
        result = token_budget_preflight("   \t  \n  ", 5)
        self.assertEqual(result.retained_tokens, [])
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(result.original_count, 0)

    def test_negative_budget_rejected(self):
        """Negative budget is rejected with ValueError."""
        with self.assertRaises(ValueError) as ctx:
            token_budget_preflight("one two three", -1)
        self.assertIn("Negative budget is rejected", str(ctx.exception))

        with self.assertRaises(ValueError):
            token_budget_preflight("one two three", -10)

    def test_zero_budget_boundary(self):
        """Zero budget retains 0 tokens and reports all original tokens as truncated."""
        result = token_budget_preflight("one two three", 0)
        self.assertEqual(result.retained_tokens, [])
        self.assertEqual(result.truncated_count, 3)
        self.assertEqual(result.original_count, 3)
        self.assertEqual(result.budget, 0)

        # Zero budget with empty input
        result_empty = token_budget_preflight("", 0)
        self.assertEqual(result_empty.retained_tokens, [])
        self.assertEqual(result_empty.truncated_count, 0)
        self.assertEqual(result_empty.original_count, 0)

    def test_exact_budget_match(self):
        """Prompt length exactly equals budget; retains all, reports 0 truncated."""
        result = token_budget_preflight("one two three", 3)
        self.assertEqual(result.retained_tokens, ["one", "two", "three"])
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(result.original_count, 3)
        self.assertEqual(result.budget, 3)

    def test_under_budget(self):
        """Prompt length is strictly less than budget; retains all, reports 0 truncated."""
        result = token_budget_preflight("one two", 5)
        self.assertEqual(result.retained_tokens, ["one", "two"])
        self.assertEqual(result.truncated_count, 0)
        self.assertEqual(result.original_count, 2)
        self.assertEqual(result.budget, 5)

    def test_boundary_overflow_single_token(self):
        """Boundary overflow of exactly 1 token over budget."""
        result = token_budget_preflight("a b c d", 3)
        self.assertEqual(result.retained_tokens, ["a", "b", "c"])
        self.assertEqual(result.truncated_count, 1)

    def test_boundary_overflow_large(self):
        """Boundary overflow of many tokens over small budget."""
        tokens = [f"word{i}" for i in range(100)]
        prompt = " ".join(tokens)
        result = token_budget_preflight(prompt, 5)
        self.assertEqual(result.retained_tokens, [f"word{i}" for i in range(5)])
        self.assertEqual(result.truncated_count, 95)
        self.assertEqual(result.original_count, 100)

    def test_whitespace_tokenizer_teaching_approximation(self):
        """Verify the deterministic whitespace tokenizer behavior and type handling."""
        self.assertEqual(whitespace_tokenizer("hello  world\nfoo\tbar"), ["hello", "world", "foo", "bar"])
        self.assertEqual(whitespace_tokenizer(""), [])
        with self.assertRaises(TypeError):
            whitespace_tokenizer(123)  # type: ignore

    def test_type_validation(self):
        """Reject invalid types for prompt and budget."""
        with self.assertRaises(TypeError):
            token_budget_preflight(["not", "a", "string"], 3)  # type: ignore
        with self.assertRaises(TypeError):
            token_budget_preflight("valid string", "three")  # type: ignore
        with self.assertRaises(TypeError):
            token_budget_preflight("valid string", 3.5)  # type: ignore
        # In Python, bool is an instance of int, so True/False should be rejected
        with self.assertRaises(TypeError):
            token_budget_preflight("valid string", True)  # type: ignore


if __name__ == "__main__":
    unittest.main()
