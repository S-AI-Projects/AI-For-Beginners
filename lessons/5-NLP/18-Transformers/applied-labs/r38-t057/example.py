"""
R38-T057: Cosine-similarity lookup — worked example with checks
Extend “Attention Mechanisms and Transformers” at lessons/5-NLP/18-Transformers/README.md.
Compare small local embedding vectors using cosine similarity and handle zero vectors explicitly.
[1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns the documented undefined-result sentinel.
Focus: Publish a worked input-to-output explanation and execute the exact example as a regression test.
"""
import math
from typing import Sequence, Optional, Union, Dict, Any

UNDEFINED_SENTINEL = None

def compute_cosine_similarity_worked(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12
) -> Dict[str, Any]:
    """
    Step-by-step worked cosine similarity calculator returning all intermediate values.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}")
    if len(vec_a) == 0:
        raise ValueError("Vectors cannot be empty")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    is_zero = norm_a <= tolerance or norm_b <= tolerance
    similarity = None if is_zero else max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    return {
        "vec_a": list(vec_a),
        "vec_b": list(vec_b),
        "dot_product": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "is_zero_vector": is_zero,
        "similarity": similarity
    }

def main():
    res = compute_cosine_similarity_worked([1, 0], [1, 0])
    print("Worked result:", res)

if __name__ == "__main__":
    main()
