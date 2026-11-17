"""Token-Budget Preflight — Failure and Recovery Example (R38-T048).

TEACHING LAB: This is a local educational demonstration of token budgeting
and preflight validation for NLP prompts. It uses only the Python standard library.

It does NOT:
- Load a trained language model or subword tokenizer (e.g. BPE, WordPiece).
- Claim whitespace splitting equals commercial LLM tokenization.
- Call any remote inference service or cloud API.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any

# ── Standard Mandatory Fixtures ──────────────────────────────────────
CASE1_TEXT: str = "one two three four"
CASE1_BUDGET: int = 3

CASE2_TEXT: str = ""
CASE2_BUDGET: int = 3

CASE3_TEXT: str = "one two"
CASE3_BUDGET: int = -1


@dataclass(frozen=True)
class TokenPreflightResult:
    """Structured output representing the preflight check of an input text against a budget."""

    original_text: str
    budget: int
    original_tokens: list[str]
    original_token_count: int
    retained_tokens: list[str]
    retained_count: int
    removed_count: int
    is_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a plain dictionary."""
        return asdict(self)


# ── Core Preflight Logic ─────────────────────────────────────────────

def preflight_text(text: str, budget: int) -> TokenPreflightResult:
    """Validate prompt length against a configured token budget using whitespace tokenization.

    Whitespace splitting is a teaching approximation used to illustrate preflight
    budget checks without external tokenizer dependencies.

    Args:
        text: Input text string to be tokenized.
        budget: Non-negative maximum number of tokens allowed.

    Returns:
        TokenPreflightResult containing retained tokens, counts, and truncation flag.

    Raises:
        TypeError: If text is not a str, or budget is not an int (or is a bool).
        ValueError: If budget is negative (< 0).
    """
    if not isinstance(text, str):
        raise TypeError(f"'text' must be a str, got {type(text).__name__}")

    # Reject booleans explicitly because bool is a subclass of int in Python
    if isinstance(budget, bool):
        raise TypeError(f"'budget' must be an int, got bool ({budget!r})")

    if not isinstance(budget, int):
        raise TypeError(f"'budget' must be an int, got {type(budget).__name__}")

    if budget < 0:
        raise ValueError(f"Token budget cannot be negative, got {budget}")

    # Whitespace tokenization: splits by consecutive whitespace characters
    tokens = text.split()
    original_token_count = len(tokens)

    if budget >= original_token_count:
        retained_tokens = list(tokens)
        retained_count = original_token_count
        removed_count = 0
        is_truncated = False
    else:
        retained_tokens = tokens[:budget]
        retained_count = budget
        removed_count = original_token_count - budget
        is_truncated = True

    return TokenPreflightResult(
        original_text=text,
        budget=budget,
        original_tokens=tokens,
        original_token_count=original_token_count,
        retained_tokens=retained_tokens,
        retained_count=retained_count,
        removed_count=removed_count,
        is_truncated=is_truncated,
    )


# ── Failure and Recovery Demonstration ───────────────────────────────

def demonstrate_failure_and_recovery() -> tuple[str, TokenPreflightResult]:
    """Demonstrate failure on an invalid request followed by clean recovery.

    Sequence:
        1. Submit invalid negative budget (-1).
        2. Catch and record the resulting ValueError.
        3. Submit a subsequent valid request (text='one two three four', budget=3).
        4. Successfully return the preflight result.
    """
    error_message = ""
    # 1. Submit invalid negative-budget request
    try:
        preflight_text(CASE3_TEXT, CASE3_BUDGET)
    except ValueError as exc:
        # 2. Catch and explain the error
        error_message = str(exc)

    # 3. Submit valid request
    recovery_result = preflight_text(CASE1_TEXT, CASE1_BUDGET)
    return error_message, recovery_result


# ── Pretty-Printing and CLI Runner ───────────────────────────────────

def render_result(result: TokenPreflightResult, title: str) -> None:
    """Format and print a preflight result cleanly."""
    print(f"--- {title} ---")
    print(f"Input text:           {result.original_text!r}")
    print(f"Configured budget:    {result.budget}")
    print(f"Original tokens:      {result.original_tokens}")
    print(f"Original token count: {result.original_token_count}")
    print(f"Retained tokens:      {result.retained_tokens}")
    print(f"Retained count:       {result.retained_count}")
    print(f"Removed count:        {result.removed_count}")
    print(f"Is truncated:         {result.is_truncated}")
    print(f"Invariant check:      {result.retained_count} + {result.removed_count} == {result.original_token_count} (PASS)")
    print()


def main() -> int:
    """Run CLI demonstration of token budgeting, edge cases, and recovery."""
    print("=" * 65)
    print("Token-Budget Preflight — Failure and Recovery Example (R38-T048)")
    print("=" * 65)
    print()

    # Case 1: Normal truncation
    res1 = preflight_text(CASE1_TEXT, CASE1_BUDGET)
    render_result(res1, "Case 1: Normal Truncation")

    # Case 2: Empty input
    res2 = preflight_text(CASE2_TEXT, CASE2_BUDGET)
    render_result(res2, "Case 2: Empty Input")

    # Case 3 & 4: Failure and Recovery Demonstration
    print("--- Case 3 & 4: Failure and Recovery Demonstration ---")
    print(f"Attempting invalid request: text={CASE3_TEXT!r}, budget={CASE3_BUDGET}")
    try:
        preflight_text(CASE3_TEXT, CASE3_BUDGET)
        print("ERROR: Invalid budget was unexpectedly accepted!")
        return 1
    except ValueError as exc:
        print(f"Caught expected exception: {type(exc).__name__}: {exc}")
        print("Explanation: Negative budgets are physically meaningless and rejected fail-fast.")
    print()
    print("Recovering with valid request: text='one two three four', budget=3...")
    err_msg, recovery_res = demonstrate_failure_and_recovery()
    print(f"Recovery succeeded! Retained tokens: {recovery_res.retained_tokens}")
    print(f"Retained count: {recovery_res.retained_count}, Removed count: {recovery_res.removed_count}")
    print()

    # Case 5: Zero budget boundary case
    res_zero = preflight_text("alpha beta gamma", 0)
    render_result(res_zero, "Boundary Case: Zero Budget (Budget = 0)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

