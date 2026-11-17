"""
Cosine-Similarity Lookup — Worked Example with Checks.

Lesson Concept Connection:
In Word Embeddings (lessons/5-NLP/14-Embeddings), words and documents are mapped
to dense vector spaces where semantic similarity is measured geometrically.
Rather than Euclidean distance (which is sensitive to vector magnitude/length),
Cosine Similarity measures the cosine of the angle between two non-zero vectors:

    cos_sim(u, v) = (u . v) / (||u|| * ||v||)

Key Specifications:
1. Identical direction: [1, 0] vs [1, 0] = 1.0.
2. Orthogonal direction: [1, 0] vs [0, 1] = 0.0.
3. Opposite direction: [1, 0] vs [-1, 0] = -1.0.
4. Zero-Vector Sentinel:
   When either vector has zero magnitude (||u|| == 0 or ||v|| == 0), the angle
   is geometrically undefined. It explicitly returns the documented sentinel `None`.
5. Validation:
   Vectors must be non-empty sequences of equal dimension containing numeric values.
"""

import math
from typing import List, Optional, Sequence, Union


def vector_norm(v: Sequence[Union[int, float]]) -> float:
    """Computes the Euclidean norm (L2 norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))


def dot_product(u: Sequence[Union[int, float]], v: Sequence[Union[int, float]]) -> float:
    """Computes the dot product of two vectors of equal dimension."""
    return sum(a * b for a, b in zip(u, v))


def cosine_similarity(
    u: Sequence[Union[int, float]],
    v: Sequence[Union[int, float]]
) -> Optional[float]:
    """
    Computes cosine similarity between two numeric vectors.

    Args:
        u: First vector (sequence of numbers).
        v: Second vector (sequence of numbers).

    Returns:
        float in [-1.0, 1.0] representing the cosine similarity, or `None` if either vector is a zero vector.

    Raises:
        ValueError: If vectors are empty or have mismatched dimensions.
        TypeError: If inputs are not sequences or contain non-numeric types.
    """
    if not isinstance(u, (list, tuple)) or not isinstance(v, (list, tuple)):
        raise TypeError(f"Vectors must be sequences (list or tuple), got {type(u).__name__} and {type(v).__name__}")

    if len(u) == 0 or len(v) == 0:
        raise ValueError("Vectors must be non-empty.")

    if len(u) != len(v):
        raise ValueError(f"Vector dimension mismatch: len(u)={len(u)} != len(v)={len(v)}.")

    # Validate elements
    for idx, val in enumerate(u):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Non-numeric element in u at index {idx}: {val!r}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite element in u at index {idx}: {val}")

    for idx, val in enumerate(v):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"Non-numeric element in v at index {idx}: {val!r}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite element in v at index {idx}: {val}")

    norm_u = vector_norm(u)
    norm_v = vector_norm(v)

    # Sentinel check: zero vector has undefined cosine similarity
    if norm_u == 0.0 or norm_v == 0.0:
        return None

    similarity = dot_product(u, v) / (norm_u * norm_v)

    # Clamp floating-point inaccuracies to [-1.0, 1.0]
    return max(-1.0, min(1.0, similarity))


def find_most_similar(
    query: Sequence[Union[int, float]],
    candidates: Sequence[Sequence[Union[int, float]]]
) -> List[tuple[int, Optional[float]]]:
    """
    Computes cosine similarity of `query` against a collection of candidates.
    Returns list of tuples `(candidate_index, similarity)`.
    """
    return [(idx, cosine_similarity(query, cand)) for idx, cand in enumerate(candidates)]


def main() -> None:
    """Demonstrate cosine similarity lookup and worked arithmetic checks."""
    print("=== Word Embeddings: Cosine Similarity Lookup ===")

    # Worked Example 1: Standard vectors
    # u = [3, 4], v = [4, 3]
    # u . v = 12 + 12 = 24
    # ||u|| = sqrt(9 + 16) = 5
    # ||v|| = sqrt(16 + 9) = 5
    # cos_sim = 24 / (5 * 5) = 24 / 25 = 0.96
    u_worked = [3, 4]
    v_worked = [4, 3]
    sim_worked = cosine_similarity(u_worked, v_worked)
    print(f"Worked Example: u={u_worked}, v={v_worked}")
    print(f"  -> Cosine Similarity: {sim_worked} (Expected: 0.96)")

    # Key specifications:
    # [1, 0] vs [1, 0] = 1.0
    print(f"[1, 0] vs [1, 0]: {cosine_similarity([1, 0], [1, 0])}")
    # [1, 0] vs [0, 1] = 0.0
    print(f"[1, 0] vs [0, 1]: {cosine_similarity([1, 0], [0, 1])}")
    # [1, 0] vs [0, 0] = None (sentinel)
    print(f"[1, 0] vs [0, 0]: {cosine_similarity([1, 0], [0, 0])} (Zero-vector sentinel None)")


if __name__ == "__main__":
    main()
