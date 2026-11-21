"""
R38-T031: Cosine-similarity lookup — runnable concept example
Lesson: lessons/5-NLP/14-Embeddings (Word Embeddings)

Concept Connection:
In word and text embeddings, words or tokens are mapped to dense continuous vectors.
Semantic similarity between embedding vectors is typically measured using Cosine Similarity
rather than Euclidean distance, because Euclidean distance is sensitive to vector magnitude
(e.g., word frequency or token counts), whereas cosine similarity measures angular alignment:

    cos_sim(u, v) = (u . v) / (||u|| * ||v||)

Where:
- (u . v) is the inner (dot) product of vectors u and v.
- ||u|| and ||v|| are the Euclidean L2 norms of u and v.
- When vectors are L2-normalized (unit length), cosine similarity simplifies directly to their dot product.

Zero-Norm Boundary Condition:
If either vector is a zero vector (e.g., an uninitialized embedding, all-zero padding, or
out-of-vocabulary representation with norm 0), the cosine angle is geometrically undefined.
Dividing by zero is prevented by explicitly detecting zero norms and returning the documented
UNDEFINED_SENTINEL (None).

Scope and Simplifications:
This implementation is a minimal, complete, self-contained educational example using only the
Python standard library. It operates on bounded 2D/ND local fixtures rather than downloading
external heavy neural network models or pre-trained embedding matrices.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Documented sentinel returned when cosine similarity is undefined (e.g. zero-norm vectors)
UNDEFINED_SENTINEL: None = None


def validate_vector(vec: Sequence[Union[int, float]], name: str = "vector") -> None:
    """Validate that input is a non-empty sequence of finite numeric values."""
    if not isinstance(vec, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of numeric values, got {type(vec).__name__}")
    if len(vec) == 0:
        raise ValueError(f"{name} cannot be empty")
    for idx, val in enumerate(vec):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name}[{idx}] must be a numeric int/float, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name}[{idx}] must be a finite real number, got {val}")


def vector_norm(vec: Sequence[Union[int, float]]) -> float:
    """Compute the Euclidean L2 norm of a numeric vector: sqrt(sum(x_i^2))."""
    validate_vector(vec, name="vec")
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def dot_product(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
) -> float:
    """Compute the dot product (inner product) between two equal-length vectors."""
    validate_vector(vec_a, name="vec_a")
    validate_vector(vec_b, name="vec_b")
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} != {len(vec_b)}")
    return sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))


def cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12,
) -> Optional[float]:
    """
    Compute cosine similarity between two numeric vectors.

    cos_sim(vec_a, vec_b) = (vec_a . vec_b) / (||vec_a|| * ||vec_b||)

    Parameters:
    - vec_a: First numeric vector.
    - vec_b: Second numeric vector of identical dimension.
    - tolerance: Threshold below which vector norm is considered zero.

    Returns:
    - float in range [-1.0, 1.0] for valid non-zero vectors.
    - UNDEFINED_SENTINEL (None) if either vector has Euclidean norm <= tolerance.

    Raises:
    - ValueError: If dimensions mismatch, vectors are empty, or values are non-finite.
    - TypeError: If inputs or elements are not numeric.
    """
    validate_vector(vec_a, name="vec_a")
    validate_vector(vec_b, name="vec_b")
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} != {len(vec_b)}")

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    # Explicit handling of zero vectors: geometrically undefined angle
    if norm_a <= tolerance or norm_b <= tolerance:
        return UNDEFINED_SENTINEL

    dot = dot_product(vec_a, vec_b)
    similarity = dot / (norm_a * norm_b)

    # Clamp to [-1.0, 1.0] to account for minor floating-point inaccuracies
    return max(-1.0, min(1.0, similarity))


def find_most_similar(
    query_vector: Sequence[Union[int, float]],
    candidate_embeddings: Dict[str, Sequence[Union[int, float]]],
    top_k: Optional[int] = None,
) -> List[Tuple[str, Optional[float]]]:
    """
    Score a query vector against a dictionary of named candidate embeddings.

    Returns candidate labels sorted by descending cosine similarity.
    Candidates with undefined similarity (e.g. zero vectors) appear at the end.
    """
    validate_vector(query_vector, name="query_vector")
    scored: List[Tuple[str, Optional[float]]] = []

    for label, candidate in candidate_embeddings.items():
        sim = cosine_similarity(query_vector, candidate)
        scored.append((label, sim))

    # Sort descending by score, placing None (undefined sentinel) at the end
    scored.sort(
        key=lambda item: (-2.0 if item[1] is None else item[1]),
        reverse=True,
    )

    if top_k is not None:
        return scored[:top_k]
    return scored


def main() -> None:
    """Demonstrate cosine similarity contract on local fixtures."""
    print("=" * 60)
    print("R38-T031: Cosine-Similarity Lookup Runnable Concept Example")
    print("=" * 60)

    # Stated contract fixtures
    fixtures = [
        ("Identical vectors ([1, 0] vs [1, 0])", [1, 0], [1, 0]),
        ("Orthogonal vectors ([1, 0] vs [0, 1])", [1, 0], [0, 1]),
        ("Opposite vectors ([1, 0] vs [-1, 0])", [1, 0], [-1, 0]),
        ("Zero-norm vector vs [1, 0]", [1, 0], [0, 0]),
        ("Both zero-norm vectors", [0, 0], [0, 0]),
        ("Arbitrary vectors ([3, 4] vs [4, 3])", [3, 4], [4, 3]),
    ]

    for desc, u, v in fixtures:
        result = cosine_similarity(u, v)
        print(f"{desc}:")
        print(f"  u = {u}, v = {v} -> cosine_similarity = {result}")

    print("\nDemonstrating Vocabulary Embedding Lookup:")
    toy_vocab = {
        "king": [0.9, 0.1],
        "queen": [0.85, 0.2],
        "apple": [0.0, 0.95],
        "zero_token": [0.0, 0.0],
    }
    query_word = "monarch"
    query_vector = [0.95, 0.05]

    print(f"Query: '{query_word}' with vector {query_vector}")
    matches = find_most_similar(query_vector, toy_vocab)
    for rank, (word, score) in enumerate(matches, 1):
        score_str = f"{score:.4f}" if score is not None else "UNDEFINED (sentinel None)"
        print(f"  {rank}. {word:12s} : {score_str}")


if __name__ == "__main__":
    main()
