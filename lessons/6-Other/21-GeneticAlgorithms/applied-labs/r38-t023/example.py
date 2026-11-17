"""
Deterministic fitness ranking and elite selection for Genetic Algorithms.

Lesson Concept Connection:
In Genetic Algorithms (lessons/6-Other/21-GeneticAlgorithms), selection is used to choose
the most optimal solutions from a candidate population based on their fitness scores.
In elite selection (elitism), a specified number of top-performing individuals (elites)
are preserved into the next generation without modification, preventing loss of the best
solutions discovered so far.

This lab demonstrates:
1. Immutability: Population candidates remain strictly unmodified during selection.
2. Determinism: High fitness is preferred. Ties are broken deterministically by candidate ID.
3. Two independent approaches:
   - Direct sorting approach (`select_elites_sorting`)
   - Declarative/lookup-based approach (`select_elites_declarative`)
   Both produce 100% equivalent selections across all valid inputs.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    """
    Immutable representation of an individual in a genetic population.

    Attributes:
        id: Unique identifier for deterministic tie-breaking (e.g. 'A', 'B', 'C').
        fitness: Numeric fitness evaluation score (higher represents better solution).
        genome: The underlying chromosome/solution representation (optional).
    """
    id: str
    fitness: float
    genome: Any = None


def select_elites_sorting(
    population: Sequence[Candidate],
    elite_count: int
) -> List[Candidate]:
    """
    Approach 1: Direct sorting approach for elite selection.

    Sorts candidates descending by fitness, and ascending by ID for deterministic tie-breaking.
    Returns the top `elite_count` candidates.

    Args:
        population: Sequence of Candidate objects (immutable).
        elite_count: Number of elite candidates to select (0 <= elite_count <= len(population)).

    Returns:
        List of selected elite Candidates in ranked order.

    Raises:
        TypeError: If elite_count is not an integer or candidate types are invalid.
        ValueError: If elite_count < 0 or elite_count > len(population).
    """
    if isinstance(elite_count, bool) or not isinstance(elite_count, int):
        raise TypeError(f"elite_count must be an integer, got {type(elite_count).__name__}")
    if elite_count < 0:
        raise ValueError(f"elite_count cannot be negative: {elite_count}")
    if elite_count > len(population):
        raise ValueError(
            f"elite_count ({elite_count}) cannot exceed population size ({len(population)})"
        )

    for c in population:
        if not isinstance(c, Candidate):
            raise TypeError(f"All population items must be Candidate instances, got {type(c).__name__}")

    # Sorting key: (-fitness, id)
    # Higher fitness comes first (-fitness is smaller); tie-breaker is candidate ID ascending
    sorted_population = sorted(
        population,
        key=lambda cand: (-cand.fitness, cand.id)
    )

    return sorted_population[:elite_count]


def select_elites_declarative(
    population: Sequence[Candidate],
    elite_count: int
) -> List[Candidate]:
    """
    Approach 2: Declarative/lookup-based approach for elite selection.

    Groups candidates into a dictionary of distinct fitness levels, sorts fitness levels
    descending, and within each bucket resolves ties by sorting candidate IDs ascending.
    Flattens the ordered buckets until `elite_count` candidates are collected.

    Args:
        population: Sequence of Candidate objects (immutable).
        elite_count: Number of elite candidates to select (0 <= elite_count <= len(population)).

    Returns:
        List of selected elite Candidates in ranked order.

    Raises:
        TypeError: If elite_count is not an integer or candidate types are invalid.
        ValueError: If elite_count < 0 or elite_count > len(population).
    """
    if isinstance(elite_count, bool) or not isinstance(elite_count, int):
        raise TypeError(f"elite_count must be an integer, got {type(elite_count).__name__}")
    if elite_count < 0:
        raise ValueError(f"elite_count cannot be negative: {elite_count}")
    if elite_count > len(population):
        raise ValueError(
            f"elite_count ({elite_count}) cannot exceed population size ({len(population)})"
        )

    for c in population:
        if not isinstance(c, Candidate):
            raise TypeError(f"All population items must be Candidate instances, got {type(c).__name__}")

    if elite_count == 0:
        return []

    # Declarative grouping: map fitness score -> list of candidates
    fitness_map: dict[float, List[Candidate]] = {}
    for cand in population:
        fitness_map.setdefault(cand.fitness, []).append(cand)

    # Sort unique fitness levels descending
    sorted_fitness_levels = sorted(fitness_map.keys(), reverse=True)

    selected: List[Candidate] = []
    for level in sorted_fitness_levels:
        # Sort candidates at the same fitness level ascending by ID (deterministic tie-break)
        bucket = sorted(fitness_map[level], key=lambda c: c.id)
        needed = elite_count - len(selected)
        if len(bucket) <= needed:
            selected.extend(bucket)
        else:
            selected.extend(bucket[:needed])
        if len(selected) == elite_count:
            break

    return selected


def main() -> None:
    """Demonstrate deterministic elite selection and compare both approaches."""
    print("=== Genetic Algorithms: One Selection Round (Elitism) ===")

    # Scenario: Fitness A=3, B=8, C=5 with elite_count=2
    candidates = (
        Candidate(id="A", fitness=3.0, genome=[0, 1, 0]),
        Candidate(id="B", fitness=8.0, genome=[1, 1, 1]),
        Candidate(id="C", fitness=5.0, genome=[1, 0, 1]),
    )
    elite_count = 2

    elites_sort = select_elites_sorting(candidates, elite_count)
    elites_decl = select_elites_declarative(candidates, elite_count)

    print(f"Population: {[(c.id, c.fitness) for c in candidates]}")
    print(f"Target elite count: {elite_count}")
    print(f"Selected by Direct Sorting    : {[c.id for c in elites_sort]}")
    print(f"Selected by Declarative Lookup: {[c.id for c in elites_decl]}")

    assert [c.id for c in elites_sort] == ["B", "C"], "Expected ['B', 'C']"
    assert elites_sort == elites_decl, "Both approaches must yield 100% equivalent results!"
    print("Verification successful: [B, C] selected, both approaches match 100%!")


if __name__ == "__main__":
    main()
