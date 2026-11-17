"""Cosine-Similarity Lookup — Scenario Assessment (R38-T047).

TEACHING LAB: This is a local educational demonstration of vector cosine similarity
for word and sequence embeddings with an applied scenario assessment:
1. Predict an output (identical and orthogonal fixtures).
2. Diagnose a failure (zero-vector division by zero).
3. Choose a boundary policy (documented undefined-result sentinel: None).

It uses only the Python standard library.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# Documented undefined-result sentinel preserved from R38-T046
UNDEFINED_SENTINEL: None = None

# ── Standard Specification Fixtures ──────────────────────────────────
CASE1_A: list[float] = [1.0, 0.0]
CASE1_B: list[float] = [1.0, 0.0]

CASE2_A: list[float] = [1.0, 0.0]
CASE2_B: list[float] = [0.0, 1.0]

CASE3_A: list[float] = [0.0, 0.0]
CASE3_B: list[float] = [1.0, 0.0]


@dataclass(frozen=True)
class AppliedQuestionAssessment:
    """Structured assessment record for an applied scenario question."""

    question_id: str
    question_type: str  # "predict_output" | "diagnose_failure" | "choose_boundary_policy"
    question_text: str
    scenario: str
    expected_answer: str
    actual_result: str
    rationale: str
    check_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a plain dictionary."""
        return asdict(self)


# ── Mathematical Implementation (Preserved from R38-T046) ────────────

def _validate_numeric(val: Any, name: str) -> float:
    """Validate numeric scalar (rejecting booleans, NaN, and Inf)."""
    if isinstance(val, bool):
        raise TypeError(f"'{name}' must be numeric, got boolean {val!r}")
    if not isinstance(val, (int, float)):
        raise TypeError(f"'{name}' must be int or float, got {type(val).__name__}")
    f_val = float(val)
    if math.isnan(f_val) or math.isinf(f_val):
        raise ValueError(f"'{name}' cannot be NaN or Infinite: {val!r}")
    return f_val


def validate_vector_pair(a: Sequence[float], b: Sequence[float]) -> None:
    """Validate dimensions and types of vector pairs."""
    if not isinstance(a, Sequence) or isinstance(a, (str, bytes)):
        raise TypeError(f"Vector A must be a Sequence, got {type(a).__name__}")
    if not isinstance(b, Sequence) or isinstance(b, (str, bytes)):
        raise TypeError(f"Vector B must be a Sequence, got {type(b).__name__}")

    if len(a) == 0:
        raise ValueError("Vector A cannot be empty")
    if len(b) == 0:
        raise ValueError("Vector B cannot be empty")
    if len(a) != len(b):
        raise ValueError(
            f"Dimension mismatch: len(A)={len(a)} != len(B)={len(b)}"
        )

    for i, val in enumerate(a):
        _validate_numeric(val, f"A[{i}]")
    for i, val in enumerate(b):
        _validate_numeric(val, f"B[{i}]")


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute algebraic dot product sum(a_i * b_i)."""
    return sum(float(x) * float(y) for x, y in zip(a, b))


def vector_norm(a: Sequence[float]) -> float:
    """Compute Euclidean magnitude (L2 norm) sqrt(sum(a_i^2))."""
    return math.sqrt(sum(float(x) ** 2 for x in a))


def cosine_similarity(
    a: Sequence[float],
    b: Sequence[float],
) -> float | None:
    """Calculate the cosine similarity between two vectors.

    Formula:
        cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))

    Zero-Vector Policy:
        If norm(A) == 0.0 or norm(B) == 0.0, the angle between the vectors
        is geometrically undefined. Under this documented contract, the
        function returns the sentinel value `None` (UNDEFINED_SENTINEL).
        It does not divide by zero or emit an arbitrary numerical result.

    Returns:
        A float in [-1.0, 1.0], or None if either vector has magnitude zero.
    """
    validate_vector_pair(a, b)

    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    # Zero-Vector Policy: Return undefined sentinel (None)
    if norm_a == 0.0 or norm_b == 0.0:
        return UNDEFINED_SENTINEL

    denom = norm_a * norm_b
    raw_sim = dot_product(a, b) / denom
    return max(-1.0, min(1.0, raw_sim))


# ── Applied Scenario Assessments (Part 3) ────────────────────────────

def check_question_1_predict_output() -> AppliedQuestionAssessment:
    """Question 1: Predict an output for identical and orthogonal vectors."""
    # Scenario: A = [1.0, 0.0], B = [1.0, 0.0]
    sim_identical = cosine_similarity(CASE1_A, CASE1_B)
    sim_orthogonal = cosine_similarity(CASE2_A, CASE2_B)

    # Executable check:
    # 1. Identical direction must equal 1.0
    # 2. Orthogonal direction must equal 0.0
    passed = (
        sim_identical is not None
        and math.isclose(sim_identical, 1.0, abs_tol=1e-9)
        and sim_orthogonal is not None
        and math.isclose(sim_orthogonal, 0.0, abs_tol=1e-9)
    )

    return AppliedQuestionAssessment(
        question_id="Q1",
        question_type="predict_output",
        question_text="What cosine similarity should the example produce for identical vectors [1, 0] vs [1, 0] and orthogonal vectors [1, 0] vs [0, 1]?",
        scenario="A=[1.0, 0.0], B=[1.0, 0.0] (identical); A=[1.0, 0.0], B=[0.0, 1.0] (orthogonal)",
        expected_answer="Identical: 1.0; Orthogonal: 0.0",
        actual_result=f"Identical: {sim_identical}; Orthogonal: {sim_orthogonal}",
        rationale=(
            "Identical vectors point in the exact same direction (theta=0 deg), "
            "yielding dot=1.0, norms=1.0,1.0, and cos(0)=1.0. "
            "Orthogonal vectors are perpendicular (theta=90 deg) with dot=0.0, "
            "yielding cos(90)=0.0."
        ),
        check_passed=passed,
    )


def check_question_2_diagnose_failure() -> AppliedQuestionAssessment:
    """Question 2: Diagnose a failure on a zero-vector input."""
    # Scenario: A = [0.0, 0.0], B = [1.0, 0.0]
    # Check execution without crash:
    crashed = False
    actual_val: Any = None
    try:
        actual_val = cosine_similarity(CASE3_A, CASE3_B)
    except Exception as e:
        crashed = True
        actual_val = type(e).__name__

    # Also test second vector zero and both zero
    second_zero_val = cosine_similarity([1.0, 0.0], [0.0, 0.0])
    both_zero_val = cosine_similarity([0.0, 0.0], [0.0, 0.0])

    # Check passes if:
    # - No unhandled ZeroDivisionError exception was thrown
    # - The result is strictly None (not a float, not NaN)
    # - All zero variants return None
    passed = (
        not crashed
        and actual_val is UNDEFINED_SENTINEL
        and second_zero_val is UNDEFINED_SENTINEL
        and both_zero_val is UNDEFINED_SENTINEL
    )

    return AppliedQuestionAssessment(
        question_id="Q2",
        question_type="diagnose_failure",
        question_text="Why would a naive cosine-similarity implementation fail on input A=[0, 0], B=[1, 0]?",
        scenario="A=[0.0, 0.0], B=[1.0, 0.0] (zero vector input)",
        expected_answer=(
            "Denominator becomes zero because norm([0, 0]) = 0.0; naive direct division "
            "raises ZeroDivisionError or emits NaN."
        ),
        actual_result=(
            f"Handled safely without crash: returned {actual_val!r} (None) across all zero-vector combinations."
        ),
        rationale=(
            "Cosine similarity is dot(A, B) / (norm(A) * norm(B)). "
            "When either vector has zero magnitude, the denominator is zero. "
            "Direct division is arithmetic error. Handling must intercept zero norm before dividing."
        ),
        check_passed=passed,
    )


def check_question_3_choose_boundary_policy() -> AppliedQuestionAssessment:
    """Question 3: Choose a boundary policy for undefined zero-magnitude inputs."""
    # Preserved policy: Return UNDEFINED_SENTINEL (None)
    sentinel_used = UNDEFINED_SENTINEL

    # Executable verification:
    # 1. Zero vector inputs must return None, not 0.0 or NaN
    # 2. Normal non-zero inputs must still return valid numeric floats
    res_zero = cosine_similarity([0.0, 0.0], [1.0, 0.0])
    res_normal = cosine_similarity([1.0, 2.0], [2.0, 4.0])

    passed = (
        sentinel_used is None
        and res_zero is None
        and isinstance(res_normal, float)
        and math.isclose(res_normal, 1.0, abs_tol=1e-9)
    )

    return AppliedQuestionAssessment(
        question_id="Q3",
        question_type="choose_boundary_policy",
        question_text="What should the application do when cosine similarity is undefined because one or both vectors are zero?",
        scenario="Boundary condition: norm(A) == 0.0 or norm(B) == 0.0",
        expected_answer="Return the documented undefined-result sentinel (None) instead of an invented numeric score.",
        actual_result=f"Policy returns {sentinel_used!r}; normal vectors return valid float ({res_normal:.4f}).",
        rationale=(
            "A zero vector has no orientation or angle in geometric space. "
            "Returning 0.0 would falsely claim the vectors are perpendicular; "
            "returning 1.0 would falsely claim they are parallel. "
            "Returning None explicitly signals an undefined calculation to calling code."
        ),
        check_passed=passed,
    )


def run_all_scenario_assessments() -> list[AppliedQuestionAssessment]:
    """Execute all three applied question assessments."""
    return [
        check_question_1_predict_output(),
        check_question_2_diagnose_failure(),
        check_question_3_choose_boundary_policy(),
    ]


# ── Pretty-Printing and CLI Runner ───────────────────────────────────

def main() -> int:
    """Run CLI demonstration of the three applied scenario assessments."""
    print("=" * 70)
    print("Cosine-Similarity Lookup — Scenario Assessment (R38-T047)")
    print("=" * 70)
    print()

    assessments = run_all_scenario_assessments()
    all_passed = True

    for item in assessments:
        status_str = "PASS" if item.check_passed else "FAIL"
        if not item.check_passed:
            all_passed = False

        print(f"[{item.question_id}] Question Type:    {item.question_type}")
        print(f"     Question:         {item.question_text}")
        print(f"     Scenario:         {item.scenario}")
        print(f"     Expected Answer:  {item.expected_answer}")
        print(f"     Actual Result:    {item.actual_result}")
        print(f"     Rationale:        {item.rationale}")
        print(f"     Executable Check: [{status_str}]")
        print("-" * 70)

    print()
    if all_passed:
        print("Summary: All 3 scenario assessment checks PASSED successfully.")
        return 0
    else:
        print("Summary: One or more scenario assessment checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

