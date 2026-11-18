"""
R38-T051: Deterministic fitness ranking and elite selection for a small population.
Extend “Genetic Algorithms” at lessons/6-Other/21-GeneticAlgorithms/README.md.
Fitness A=3,B=8,C=5 with elite_count 2 selects B,C; ties use candidate ID; count beyond the population is rejected.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple, List

@dataclass(frozen=True)
class Candidate:
    id: str
    fitness: float

def select_elites(population: Sequence[Candidate], elite_count: int) -> Tuple[Candidate, ...]:
    """
    Deterministically rank candidates and select top `elite_count` individuals.

    Ranking Criteria:
    1. Fitness descending (higher fitness is ranked higher)
    2. Candidate ID ascending (alphabetical order for deterministic tie breaking)

    Validation:
    - If elite_count < 0 or elite_count > len(population), raises ValueError.
    """
    if elite_count < 0 or elite_count > len(population):
        raise ValueError(f"elite_count {elite_count} is out of bounds for population of size {len(population)}")

    # Deterministic sort: higher fitness first (-c.fitness), then ID ascending
    ranked = sorted(population, key=lambda c: (-c.fitness, c.id))
    return tuple(ranked[:elite_count])

def main():
    population = [
        Candidate(id="A", fitness=3.0),
        Candidate(id="B", fitness=8.0),
        Candidate(id="C", fitness=5.0),
    ]
    elites = select_elites(population, 2)
    print("Selected elites:", [c.id for c in elites])

if __name__ == "__main__":
    main()
