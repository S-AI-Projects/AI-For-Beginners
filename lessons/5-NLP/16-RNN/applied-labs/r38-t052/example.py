"""
R38-T052: Cosine-similarity lookup — boundary-case lesson extension
Extend “Recurrent Neural Networks” at lessons/5-NLP/16-RNN/README.md.
Compare small local embedding vectors using cosine similarity and handle zero vectors explicitly.
[1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns the documented undefined-result sentinel.
"""
import math
from typing import Sequence, Optional, Union

UNDEFINED_SENTINEL = None

def cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12
) -> Optional[float]:
    """
    Compute cosine similarity between two numeric vectors.

    Returns:
    - Cosine similarity in range [-1.0, 1.0] for valid non-zero vectors.
    - UNDEFINED_SENTINEL (None) if either vector has norm 0 (documented sentinel).

    Raises:
    - ValueError if vector lengths differ or vectors are empty.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}")
    if len(vec_a) == 0:
        raise ValueError("Vectors cannot be empty")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a <= tolerance or norm_b <= tolerance:
        return UNDEFINED_SENTINEL

    sim = dot_product / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))

def main():
    print("[1, 0] vs [1, 0]:", cosine_similarity([1, 0], [1, 0]))
    print("[1, 0] vs [0, 1]:", cosine_similarity([1, 0], [0, 1]))
    print("[1, 0] vs [0, 0]:", cosine_similarity([1, 0], [0, 0]))

if __name__ == "__main__":
    main()
