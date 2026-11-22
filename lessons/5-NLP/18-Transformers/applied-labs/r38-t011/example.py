"""
Cosine Similarity Lookup Concept Example
Applied lab for lessons/5-NLP/18-Transformers (Task R38-T011)

This module demonstrates vector cosine similarity and its direct relationship to
dot-product attention in Transformer architectures.

In attention mechanisms:
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
The unscaled compatibility score between query and key is given by their dot product.
When query and key representations are normalized to unit Euclidean length (L2 norm = 1),
the dot product equals the cosine similarity:
    cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)

Boundary Case:
When either vector is a zero vector (norm == 0), the direction is undefined and division
by zero would occur. This implementation explicitly detects zero-norm vectors and returns
the documented sentinel value UNDEFINED_SENTINEL (float('nan')).
"""

import math
from typing import Sequence, Union

# Documented undefined-result sentinel for zero-norm vectors
UNDEFINED_SENTINEL: float = float("nan")


def vector_norm(vec: Sequence[Union[int, float]]) -> float:
    """Compute the Euclidean (L2) norm of a sequence of numbers."""
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
    Compute cosine similarity between two numeric vectors.

    Contract:
      - [1, 0] vs [1, 0] -> 1.0 (identical direction)
      - [1, 0] vs [0, 1] -> 0.0 (orthogonal direction)
      - [1, 0] vs [-1, 0] -> -1.0 (opposite direction)
      - Any zero vector (norm <= tolerance) -> UNDEFINED_SENTINEL (float('nan'))

    Args:
        vec_a: Sequence of int or float
        vec_b: Sequence of int or float
        tolerance: Epsilon threshold to detect zero norm

    Returns:
        float: Cosine similarity in range [-1.0, 1.0], or UNDEFINED_SENTINEL (float('nan'))

    Raises:
        ValueError: If vector lengths do not match or vectors are empty.
        TypeError: If non-numeric elements are encountered.
    """
    if not vec_a or not vec_b:
        raise ValueError("Vectors must not be empty.")

    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: len(a)={len(vec_a)} != len(b)={len(vec_b)}")

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    # Zero-norm detection: direction is undefined
    if norm_a <= tolerance or norm_b <= tolerance:
        return UNDEFINED_SENTINEL

    dot = dot_product(vec_a, vec_b)
    sim = dot / (norm_a * norm_b)

    # Clamp rounding errors into [-1.0, 1.0]
    return max(-1.0, min(1.0, sim))


def is_undefined(val: float) -> bool:
    """Check whether a cosine similarity result is the undefined sentinel."""
    return math.isnan(val)


def run_demonstration():
    """Run educational demonstration fixtures and print their interpretation."""
    fixtures = [
        ("Identical vectors (max similarity / full attention alignment)", [1, 0], [1, 0]),
        ("Orthogonal vectors (no alignment / zero attention weight)", [1, 0], [0, 1]),
        ("Opposite vectors (negative alignment)", [1, 0], [-1, 0]),
        ("Arbitrary 3D vectors", [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
        ("Zero vector input (undefined direction sentinel)", [0, 0], [1, 0]),
        ("Both zero vectors (undefined direction sentinel)", [0, 0], [0, 0]),
    ]

    print("=== Cosine Similarity & Attention Lookup Demonstration ===")
    print(f"Undefined sentinel representation: {UNDEFINED_SENTINEL}\n")

    for desc, va, vb in fixtures:
        result = cosine_similarity(va, vb)
        res_str = "UNDEFINED_SENTINEL (nan)" if is_undefined(result) else f"{result:.4f}"
        print(f"Scenario: {desc}")
        print(f"  Input a: {va}")
        print(f"  Input b: {vb}")
        print(f"  Cosine Similarity: {res_str}\n")


if __name__ == "__main__":
    run_demonstration()
