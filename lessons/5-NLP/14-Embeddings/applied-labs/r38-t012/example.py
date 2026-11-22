"""
Cosine Similarity Lookup — Boundary-Case Lesson Extension
Applied lab for lessons/5-NLP/14-Embeddings (Task R38-T012)

Focus:
Emphasize the smallest valid, boundary, and invalid cases in the fixture,
and explain the exact boundary decision.

In word embeddings (Word2Vec, GloVe), semantic similarity is measured by the
cosine of the angle between two embedding vectors:
    cosine_similarity(u, v) = (u . v) / (||u||_2 * ||v||_2)

Boundary Decisions:
1. Smallest valid vectors: 1-dimensional non-zero vectors, e.g., [1.0] and [1.0] -> 1.0
2. Orthogonal boundary: [1, 0] vs [0, 1] -> 0.0 (angle = 90 deg)
3. Opposite boundary: [1, 0] vs [-1, 0] -> -1.0 (angle = 180 deg)
4. Zero-norm boundary: When ||u|| <= tolerance or ||v|| <= tolerance, the direction
   vector is undefined. Normalization requires division by zero. This implementation
   explicitly catches zero-norm vectors and returns UNDEFINED_SENTINEL (float('nan')).
5. Invalid cases:
   - Dimension mismatch (len(a) != len(b)) -> raises ValueError
   - Empty input vectors -> raises ValueError
"""

import math
from typing import Sequence, Union

# Documented undefined-result sentinel for zero-norm vectors
UNDEFINED_SENTINEL: float = float("nan")


def vector_norm(vec: Sequence[Union[int, float]]) -> float:
    """Compute Euclidean (L2) norm of a numeric vector."""
    return math.sqrt(sum(x * x for x in vec))


def dot_product(vec_a: Sequence[Union[int, float]], vec_b: Sequence[Union[int, float]]) -> float:
    """Compute dot product between two equal-length vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(vec_a)} != len(b)={len(vec_b)}")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12,
) -> float:
    """
    Compute cosine similarity with explicit boundary decisions.

    Contract:
      - [1, 0] vs [1, 0] -> 1.0 (identical direction)
      - [1, 0] vs [0, 1] -> 0.0 (orthogonal)
      - Zero vector (norm <= tolerance) -> UNDEFINED_SENTINEL (float('nan'))

    Args:
        vec_a: Sequence of int or float
        vec_b: Sequence of int or float
        tolerance: Threshold below which vector is considered zero-norm

    Returns:
        float in [-1.0, 1.0] or UNDEFINED_SENTINEL
    """
    if not vec_a or not vec_b:
        raise ValueError("Vectors must not be empty.")

    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(vec_a)} != len(b)={len(vec_b)}")

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    # Boundary decision: zero norm has no defined direction
    if norm_a <= tolerance or norm_b <= tolerance:
        return UNDEFINED_SENTINEL

    dot = dot_product(vec_a, vec_b)
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


def is_undefined(val: float) -> bool:
    """Check whether a cosine similarity result is the undefined sentinel."""
    return math.isnan(val)


def run_demonstration():
    """Run educational demonstration fixtures emphasizing boundary decisions."""
    fixtures = [
        ("Smallest valid (1D identical)", [1.0], [1.0]),
        ("Smallest valid (1D opposite)", [1.0], [-1.0]),
        ("2D Identical (full alignment)", [1, 0], [1, 0]),
        ("2D Orthogonal (boundary: zero similarity)", [1, 0], [0, 1]),
        ("2D Opposite (boundary: -1 similarity)", [1, 0], [-1, 0]),
        ("Zero-norm boundary (vector a is 0)", [0, 0], [1, 0]),
        ("Zero-norm boundary (vector b is 0)", [1, 0], [0, 0]),
        ("Zero-norm boundary (both 0)", [0, 0], [0, 0]),
    ]

    print("=== Cosine Similarity Boundary-Case Extension (R38-T012) ===")
    print(f"Undefined sentinel: {UNDEFINED_SENTINEL}\n")
    for desc, va, vb in fixtures:
        res = cosine_similarity(va, vb)
        res_str = "UNDEFINED_SENTINEL (nan)" if is_undefined(res) else f"{res:.4f}"
        print(f"[{desc}]")
        print(f"  a={va}, b={vb} -> result: {res_str}")


if __name__ == "__main__":
    run_demonstration()
