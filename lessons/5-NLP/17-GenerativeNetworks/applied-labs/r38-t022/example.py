"""
Token-Budget Preflight for Generative Networks (Boundary-Case Lesson Extension).

Lesson Concept Connection:
In generative language models (lessons/5-NLP/17-GenerativeNetworks), generation begins
with a prompt passed through the model (RNN/LSTM/GRU or Transformer). Every generative
model operates within a finite sequence context window or token budget. When the input
prompt exceeds the allowable context window, preflight truncation or verification is
required to avoid memory exhaustion, out-of-index errors, or degraded generation.

Note:
This module provides a deterministic whitespace tokenizer fixture clearly labeled
as a teaching approximation (as real generative models utilize BPE, WordPiece, SentencePiece,
or character tokenization).
"""

from dataclasses import dataclass
from typing import List, Optional


def whitespace_tokenizer(text: str) -> List[str]:
    """
    Deterministic local whitespace tokenizer fixture.

    TEACHING APPROXIMATION NOTICE:
    This tokenizer splits purely on standard whitespace sequences. It is a simplified
    educational fixture designed to isolate token-budget boundary mechanics without
    relying on external heavyweight vocabulary libraries (e.g. Byte-Pair Encoding or SentencePiece).

    Args:
        text: Input string prompt.

    Returns:
        List of whitespace-delimited token strings. Returns an empty list for empty or whitespace-only input.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string input for tokenizer, got {type(text).__name__}")
    return text.split()


@dataclass(frozen=True)
class PreflightResult:
    """
    Result of a token-budget preflight check.

    Attributes:
        retained_tokens: Tokens that fit within the configured budget.
        truncated_count: Number of tokens removed because they exceeded the budget.
        original_count: Total number of tokens in the prompt before budget enforcement.
        budget: The configured token budget.
    """
    retained_tokens: List[str]
    truncated_count: int
    original_count: int
    budget: int


def token_budget_preflight(
    prompt: str,
    budget: int,
    tokenizer=whitespace_tokenizer
) -> PreflightResult:
    """
    Checks prompt length against a configured token budget and truncates if necessary.

    Args:
        prompt: Raw input text prompt.
        budget: Maximum allowable number of tokens (non-negative integer).
        tokenizer: Callable that tokenizes a string into a list of strings (defaults to whitespace_tokenizer).

    Returns:
        PreflightResult containing retained tokens, truncated count, original count, and budget.

    Raises:
        ValueError: If budget is negative (< 0).
        TypeError: If prompt is not a string or budget is not an integer.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"Budget must be an integer, got {type(budget).__name__}")
    if budget < 0:
        raise ValueError(f"Negative budget is rejected: {budget}")
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

    tokens = tokenizer(prompt)
    original_count = len(tokens)

    if budget == 0:
        retained_tokens: List[str] = []
        truncated_count = original_count
    elif original_count <= budget:
        retained_tokens = list(tokens)
        truncated_count = 0
    else:
        retained_tokens = tokens[:budget]
        truncated_count = original_count - budget

    return PreflightResult(
        retained_tokens=retained_tokens,
        truncated_count=truncated_count,
        original_count=original_count,
        budget=budget,
    )


def main() -> None:
    """Demonstrate token-budget preflight on typical and boundary inputs."""
    print("=== Generative Networks: Token-Budget Preflight Demonstration ===")

    # Scenario 1: Standard overflow
    prompt1 = "one two three four"
    budget1 = 3
    res1 = token_budget_preflight(prompt1, budget1)
    print(f"Prompt: '{prompt1}', Budget: {budget1}")
    print(f"  -> Retained tokens: {res1.retained_tokens}")
    print(f"  -> Truncated count: {res1.truncated_count}")
    print(f"  -> Original count : {res1.original_count}\n")

    # Scenario 2: Exact boundary match
    prompt2 = "alpha beta gamma"
    budget2 = 3
    res2 = token_budget_preflight(prompt2, budget2)
    print(f"Prompt: '{prompt2}', Budget: {budget2}")
    print(f"  -> Retained tokens: {res2.retained_tokens}")
    print(f"  -> Truncated count: {res2.truncated_count}\n")

    # Scenario 3: Boundary zero-budget
    prompt3 = "hello world"
    budget3 = 0
    res3 = token_budget_preflight(prompt3, budget3)
    print(f"Prompt: '{prompt3}', Budget: {budget3}")
    print(f"  -> Retained tokens: {res3.retained_tokens}")
    print(f"  -> Truncated count: {res3.truncated_count}\n")

    # Scenario 4: Empty prompt
    prompt4 = ""
    budget4 = 5
    res4 = token_budget_preflight(prompt4, budget4)
    print(f"Prompt: '{prompt4}', Budget: {budget4}")
    print(f"  -> Retained tokens: {res4.retained_tokens}")
    print(f"  -> Truncated count: {res4.truncated_count}\n")


if __name__ == "__main__":
    main()
