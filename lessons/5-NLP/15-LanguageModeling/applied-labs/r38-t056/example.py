"""
R38-T056: Cosine-similarity lookup — refactor the teaching example
Extend “Language Modeling” at lessons/5-NLP/15-LanguageModeling/README.md.
Compare small local embedding vectors using cosine similarity and handle zero vectors explicitly.
[1,0] versus [1,0] gives 1; versus [0,1] gives 0; a zero vector returns the documented undefined-result sentinel.
Focus: Separate the core algorithm from presentation code and eliminate side effects from pure functions.
"""
import math
from typing import Sequence, Optional, Union, Dict, Any

UNDEFINED_SENTINEL = None

def compute_cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12
) -> Optional[float]:
    """
    Pure computation function: returns cosine similarity or UNDEFINED_SENTINEL for zero vectors.
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

    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))

def format_similarity_report(
    label_a: str,
    label_b: str,
    similarity: Optional[float]
) -> Dict[str, Any]:
    """
    Presentation layer: creates a clean presentation record without altering the pure computation.
    """
    return {
        "comparison": f"{label_a} vs {label_b}",
        "score": similarity,
        "status": "VALID" if similarity is not None else "UNDEFINED_ZERO_VECTOR"
    }

def main():
    score = compute_cosine_similarity([1, 0], [1, 0])
    report = format_similarity_report("TokenA", "TokenB", score)
    print("Report:", report)

if __name__ == "__main__":
    main()
