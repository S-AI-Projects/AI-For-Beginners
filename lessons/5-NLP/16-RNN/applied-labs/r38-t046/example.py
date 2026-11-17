"""Cosine-Similarity Lookup — Evaluation Experiment (R38-T046).

TEACHING LAB: This is a local educational demonstration of vector cosine similarity
for word and sequence embeddings. It uses only the Python standard library.

It does NOT:
- Train or download neural embedding models (e.g. Word2Vec, GloVe, BERT).
- Claim real-world semantic natural language processing accuracy.
- Connect to any cloud service, external API, or require GPU hardware.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# Sentinel value indicating that cosine similarity is undefined
# because at least one input vector has zero Euclidean magnitude (norm = 0).
UNDEFINED_SENTINEL: None = None

# ── Standard Mandatory Fixtures ──────────────────────────────────────
CASE1_A: list[float] = [1.0, 0.0]
CASE1_B: list[float] = [1.0, 0.0]

CASE2_A: list[float] = [1.0, 0.0]
CASE2_B: list[float] = [0.0, 1.0]

CASE3_A: list[float] = [0.0, 0.0]
CASE3_B: list[float] = [1.0, 0.0]


@dataclass(frozen=True)
class ExperimentResult:
    """Evaluation result for a controlled vector configuration."""

    name: str
    dimension: int
    vector_a: list[float]
    vector_b: list[float]
    expected_similarity: float | None
    actual_similarity: float | None
    is_correct: bool
    iterations: int
    total_time_seconds: float
    avg_time_microseconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain dictionary representation."""
        return asdict(self)


# ── Mathematical Helpers ─────────────────────────────────────────────

def _validate_numeric(val: Any, name: str) -> float:
    """Validate that an element is numeric (rejecting booleans, NaN, and Inf)."""
    if isinstance(val, bool):
        raise TypeError(f"'{name}' must be numeric, got boolean {val!r}")
    if not isinstance(val, (int, float)):
        raise TypeError(f"'{name}' must be int or float, got {type(val).__name__}")
    f_val = float(val)
    if math.isnan(f_val) or math.isinf(f_val):
        raise ValueError(f"'{name}' cannot be NaN or Infinite: {val!r}")
    return f_val


def validate_vector_pair(a: Sequence[float], b: Sequence[float]) -> None:
    """Validate two vectors for similarity comparison."""
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


# ── Cosine Similarity Function ───────────────────────────────────────

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

    Args:
        a: First numeric vector sequence.
        b: Second numeric vector sequence of equal dimension.

    Returns:
        A float in [-1.0, 1.0], or None if either vector has magnitude zero.

    Raises:
        TypeError: If inputs are not sequences or contain non-numeric/bool values.
        ValueError: If vectors have mismatched lengths, are empty, or contain NaN/Inf.
    """
    validate_vector_pair(a, b)

    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    # Explicit Zero-Vector Policy: Return undefined sentinel (None)
    if norm_a == 0.0 or norm_b == 0.0:
        return UNDEFINED_SENTINEL

    denom = norm_a * norm_b
    raw_sim = dot_product(a, b) / denom

    # Clamp floating-point inaccuracies to valid theoretical range [-1.0, 1.0]
    return max(-1.0, min(1.0, raw_sim))


# ── Controlled Evaluation Experiments ────────────────────────────────

def run_experiment(
    name: str,
    a: Sequence[float],
    b: Sequence[float],
    expected: float | None,
    iterations: int = 10_000,
) -> ExperimentResult:
    """Execute a controlled evaluation configuration and measure execution time."""
    dim = len(a)

    # 1. Verification of correctness
    actual = cosine_similarity(a, b)
    if expected is None:
        is_correct = actual is None
    else:
        is_correct = actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)

    # 2. Local timing observation using time.perf_counter
    start = time.perf_counter()
    for _ in range(iterations):
        _ = cosine_similarity(a, b)
    elapsed = time.perf_counter() - start

    avg_microsec = (elapsed / iterations) * 1_000_000

    return ExperimentResult(
        name=name,
        dimension=dim,
        vector_a=[float(x) for x in a],
        vector_b=[float(x) for x in b],
        expected_similarity=expected,
        actual_similarity=actual,
        is_correct=is_correct,
        iterations=iterations,
        total_time_seconds=elapsed,
        avg_time_microseconds=avg_microsec,
    )


def run_all_experiments(iterations: int = 10_000) -> list[ExperimentResult]:
    """Execute the three required controlled configurations."""
    experiments = [
        # Configuration 1: Small vectors (2D)
        # A = [3.0, 4.0], B = [4.0, 3.0]
        # dot = 12 + 12 = 24, norm(A)=5, norm(B)=5, sim = 24 / 25 = 0.96
        (
            "Config 1: 2D Non-trivial Acute Vectors",
            [3.0, 4.0],
            [4.0, 3.0],
            24.0 / 25.0,  # 0.96
        ),
        # Configuration 2: Medium vectors (4D)
        # A = [1.0, 2.0, 3.0, 4.0], B = [4.0, 3.0, 2.0, 1.0]
        # dot = 4 + 6 + 6 + 4 = 20, norm(A)=sqrt(30), norm(B)=sqrt(30)
        # sim = 20 / 30 = 2/3
        (
            "Config 2: 4D Symmetric Mixed Vectors",
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            20.0 / 30.0,  # 0.666666...
        ),
        # Configuration 3: Larger local vectors (8D)
        # A has first 4 components non-zero, B has last 4 components non-zero (Orthogonal partitions)
        # dot = 0.0, norm(A)=2, norm(B)=2, sim = 0.0
        (
            "Config 3: 8D Orthogonal Block Vectors",
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            0.0,
        ),
    ]

    results = []
    for name, a, b, expected in experiments:
        res = run_experiment(name, a, b, expected, iterations=iterations)
        results.append(res)
    return results


# ── Pretty-Printing and CLI Runner ───────────────────────────────────

def main() -> int:
    """Run CLI demonstration of mandatory fixtures and controlled experiments."""
    print("=" * 65)
    print("Cosine-Similarity Lookup — Evaluation Experiment (R38-T046)")
    print("=" * 65)
    print()

    print("--- Mandatory Specification Fixtures ---")
    c1 = cosine_similarity(CASE1_A, CASE1_B)
    print(f"Case 1 (Identical):  A={CASE1_A}, B={CASE1_B} -> Result: {c1} (Expected: 1.0)")

    c2 = cosine_similarity(CASE2_A, CASE2_B)
    print(f"Case 2 (Orthogonal): A={CASE2_A}, B={CASE2_B} -> Result: {c2} (Expected: 0.0)")

    c3 = cosine_similarity(CASE3_A, CASE3_B)
    print(f"Case 3 (Zero vector): A={CASE3_A}, B={CASE3_B} -> Result: {c3!r} (Expected: None)")

    opp = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
    print(f"Opposite Direction:  A=[1.0, 0.0], B=[-1.0, 0.0] -> Result: {opp} (Expected: -1.0)")
    print()

    print("--- Controlled Evaluation Experiments ---")
    results = run_all_experiments(iterations=10_000)
    for res in results:
        print(f"Configuration: {res.name}")
        print(f"  Dimension:           {res.dimension}")
        print(f"  Vector A:            {res.vector_a}")
        print(f"  Vector B:            {res.vector_b}")
        print(f"  Expected Similarity: {res.expected_similarity}")
        print(f"  Actual Similarity:   {res.actual_similarity:.6f}" if res.actual_similarity is not None else f"  Actual Similarity:   {res.actual_similarity}")
        print(f"  Correct:             {res.is_correct}")
        print(f"  Benchmark:           {res.iterations:,} iterations in {res.total_time_seconds:.4f}s ({res.avg_time_microseconds:.2f} µs/op)")
        print()

    print("Experiment Trade-off Summary:")
    print("  - As vector dimension grows (2D -> 4D -> 8D), arithmetic operations scale linearly O(D).")
    print("  - All configurations produce exact mathematical results adhering to theoretical bounds [-1.0, 1.0].")
    print("  - Timing measurements reflect local CPU execution time, not a general cloud latency guarantee.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

