"""
Token-Budget Preflight for NER Pipelines — Refactored Teaching Example.

Lesson Concept Connection:
In Named Entity Recognition (lessons/5-NLP/19-NER), input sequences (sentences or titles)
are classified token-by-token (e.g. BIO tagging). Recurrent networks (LSTMs) or transformer-based
token classifiers operate with a fixed maximum sequence length (token budget).
Inputs exceeding this limit require preflight token budget enforcement to truncate
excess tokens and report retained versus truncated counts.

Refactoring Structure:
1. Baseline Implementation (`token_budget_preflight_baseline`):
   A procedural monolithic implementation that tokenizes, validates budgets,
   and slices sequences in a single function.
2. Refactored Implementation (`token_budget_preflight`):
   Extracts a dedicated responsibility class, `TokenTruncationPolicy`, which encapsulates
   budget evaluation, boundary validation, and truncation slicing, promoting single-responsibility
   and reuse across token classification pipelines.
"""

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class PreflightReport:
    """
    Structured outcome of a token-budget preflight check.

    Attributes:
        retained_tokens: Tokens that fit within the budget.
        truncated_count: Number of tokens removed due to budget limit.
        original_count: Original number of tokens before truncation.
        budget: The configured token budget.
    """
    retained_tokens: List[str]
    truncated_count: int
    original_count: int
    budget: int


def whitespace_tokenize(text: str) -> List[str]:
    """
    Splits text on whitespace into tokens (educational teaching approximation).
    """
    if not isinstance(text, str):
        raise TypeError(f"Prompt must be a string, got {type(text).__name__}")
    return text.split()


# =====================================================================
# 1. Baseline Implementation (Monolithic Teaching Example)
# =====================================================================

def token_budget_preflight_baseline(prompt: str, budget: int) -> PreflightReport:
    """
    Baseline procedural implementation of token-budget preflight.
    All validation, counting, and slicing logic are coupled in one function.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError(f"Budget must be an integer, got {type(budget).__name__}")
    if budget < 0:
        raise ValueError(f"Negative budget is rejected: {budget}")
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

    tokens = whitespace_tokenize(prompt)
    original_count = len(tokens)

    if budget == 0:
        retained = []
        truncated = original_count
    elif original_count <= budget:
        retained = list(tokens)
        truncated = 0
    else:
        retained = tokens[:budget]
        truncated = original_count - budget

    return PreflightReport(
        retained_tokens=retained,
        truncated_count=truncated,
        original_count=original_count,
        budget=budget,
    )


# =====================================================================
# 2. Refactored Implementation (Extracted Responsibility: TokenTruncationPolicy)
# =====================================================================

class TokenTruncationPolicy:
    """
    Extracted responsibility: Encapsulates policy rules for token budget truncation.

    Separates the domain policy (budget limits, truncation calculation, boundary checks)
    from text tokenization and report serialization.
    """

    def __init__(self, budget: int):
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError(f"Budget must be an integer, got {type(budget).__name__}")
        if budget < 0:
            raise ValueError(f"Negative budget is rejected: {budget}")
        self._budget = budget

    @property
    def budget(self) -> int:
        return self._budget

    def truncate(self, tokens: Sequence[str]) -> Tuple[List[str], int]:
        """
        Applies the truncation policy to a sequence of tokens.

        Returns:
            Tuple of (retained_tokens, truncated_count).
        """
        total = len(tokens)
        if self._budget == 0:
            return ([], total)
        if total <= self._budget:
            return (list(tokens), 0)
        return (list(tokens[:self._budget]), total - self._budget)


def token_budget_preflight(
    prompt: str,
    budget: int,
    policy_cls=TokenTruncationPolicy
) -> PreflightReport:
    """
    Refactored preflight check delegating truncation decisions to TokenTruncationPolicy.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")

    policy = policy_cls(budget)
    tokens = whitespace_tokenize(prompt)
    retained, truncated_count = policy.truncate(tokens)

    return PreflightReport(
        retained_tokens=retained,
        truncated_count=truncated_count,
        original_count=len(tokens),
        budget=budget,
    )


def main() -> None:
    """Demonstrate both baseline and refactored implementations."""
    print("=== NER Token-Budget Preflight: Baseline vs Refactored Policy ===")
    sample_text = "Tricuspid valve regurgitation and lithium carbonate toxicity in infant"
    budget = 5

    report_baseline = token_budget_preflight_baseline(sample_text, budget)
    report_refactored = token_budget_preflight(sample_text, budget)

    print(f"Sample Input: '{sample_text}'")
    print(f"Configured Budget: {budget}")
    print(f"\n[Baseline Output]:")
    print(f"  Retained ({len(report_baseline.retained_tokens)}): {report_baseline.retained_tokens}")
    print(f"  Truncated Count: {report_baseline.truncated_count}")
    print(f"\n[Refactored Output]:")
    print(f"  Retained ({len(report_refactored.retained_tokens)}): {report_refactored.retained_tokens}")
    print(f"  Truncated Count: {report_refactored.truncated_count}")

    assert report_baseline == report_refactored, "Baseline and Refactored must be 100% equivalent!"
    print("\nVerified: Baseline and refactored TokenTruncationPolicy yield identical results.")


if __name__ == "__main__":
    main()
