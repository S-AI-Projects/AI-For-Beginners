"""
Cosine-Similarity Lookup — Worked Example with Checks
Applied lab for lessons/5-NLP/15-LanguageModeling (Task R38-T017)

Concept & Contract:
In statistical and neural Language Modeling (n-grams, RNNs, transformer LMs),
vocabulary tokens or context representations are represented as continuous vectors.
Evaluating semantic relatedness and word-vector alignments relies on vector cosine similarity:
    cosine_similarity(u, v) = (u . v) / (||u||_2 * ||v||_2)

Contract:
- [1, 0] versus [1, 0] gives 1.0 (exact direction alignment).
- [1, 0] versus [0, 1] gives 0.0 (orthogonal / uncorrelated directions).
- A zero vector returns the documented undefined-result sentinel `UNDEFINED_SENTINEL` (float('nan')).

Focus:
Publish a worked input-to-output explanation tracing normalization, dot product,
and quotient calculation, and execute the exact example as a regression test.
"""

import math
from typing import Dict, Sequence, Union

UNDEFINED_SENTINEL: float = float("nan")


def vector_norm(vec: Sequence[Union[int, float]]) -> float:
    """Compute Euclidean (L2) norm of a numeric sequence."""
    return math.sqrt(sum(x * x for x in vec))


def dot_product(vec_a: Sequence[Union[int, float]], vec_b: Sequence[Union[int, float]]) -> float:
    """Compute dot product of two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} != {len(vec_b)}")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def explain_cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12,
) -> Dict[str, Union[float, str]]:
    """
    Compute cosine similarity while returning a worked step-by-step trace.
    """
    if not vec_a or not vec_b:
        raise ValueError("Vectors must not be empty.")
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} != {len(vec_b)}")

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    if norm_a <= tolerance or norm_b <= tolerance:
        return {
            "norm_a": norm_a,
            "norm_b": norm_b,
            "dot_product": 0.0,
            "result": UNDEFINED_SENTINEL,
            "explanation": "Zero norm encountered: division by zero prevented, returning UNDEFINED_SENTINEL."
        }

    dot = dot_product(vec_a, vec_b)
    raw_sim = dot / (norm_a * norm_b)
    clamped_sim = max(-1.0, min(1.0, raw_sim))

    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "dot_product": dot,
        "result": clamped_sim,
        "explanation": f"dot({dot}) / (norm_a({norm_a:.4f}) * norm_b({norm_b:.4f})) = {clamped_sim:.4f}"
    }


def cosine_similarity(
    vec_a: Sequence[Union[int, float]],
    vec_b: Sequence[Union[int, float]],
    tolerance: float = 1e-12,
) -> float:
    """Standard contract entrypoint."""
    trace = explain_cosine_similarity(vec_a, vec_b, tolerance=tolerance)
    return float(trace["result"])


def is_undefined(val: float) -> bool:
    """Check whether a similarity value is the undefined sentinel."""
    return math.isnan(val)


def run_demonstration():
    fixtures = [
        ("Identical", [1, 0], [1, 0]),
        ("Orthogonal", [1, 0], [0, 1]),
        ("Zero Vector", [0, 0], [1, 0]),
    ]
    print("=== Cosine Similarity: Worked Example with Checks (R38-T017) ===")
    for name, va, vb in fixtures:
        trace = explain_cosine_similarity(va, vb)
        res = trace["result"]
        res_str = "UNDEFINED_SENTINEL (nan)" if is_undefined(res) else f"{res:.4f}"
        print(f"[{name}] {va} vs {vb} -> {res_str}")
        print(f"  Step trace: {trace['explanation']}")


if __name__ == "__main__":
    run_demonstration()
