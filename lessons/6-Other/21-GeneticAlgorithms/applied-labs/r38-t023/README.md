# Applied Lab R38-T023: Deterministic Fitness Ranking and Elite Selection (Comparing Two Approaches)

## Overview & Lesson Concept Connection

In **Genetic Algorithms** ([Lesson 21: Genetic Algorithms](../../README.md)), evolutionary optimization progresses across discrete generations. One fundamental genetic operator is **Selection**: identifying the highest-performing candidate solutions according to their evaluation by a fitness function.

A widely used strategy is **Elitism**:
- The top $K$ candidates (the "elites") are carried over into the succeeding generation intact, without being altered or degraded by crossover and mutation.
- **Candidate Immutability:** Selected candidates must remain unaltered during ranking and selection rounds.
- **Deterministic Tie-Breaking:** When multiple individuals share the same fitness score, tie-breaking must be strictly deterministic (e.g., ordered by unique candidate ID) to maintain reproducibility in algorithmic benchmarks and test pipelines.

---

## Comparing Two Approaches

This lab implements and validates two distinct architectural approaches to elite selection:

1. **Direct Sorting (`select_elites_sorting`)**:
   - Sorts candidates directly via multi-key sorting (`(-fitness, id)`), achieving descending fitness ranking with ascending alphabetical tie-breaking, followed by slicing the top $K$ items.
   - Time complexity: $O(N \log N)$.
2. **Declarative / Lookup Approach (`select_elites_declarative`)**:
   - Groups candidates by discrete fitness scores in a lookup map, identifies sorted unique fitness levels, and sorts within each bucket by candidate ID before flattening to retrieve the top $K$ candidates.
   - Highly suitable for sparse or discretized fitness spaces.

Both approaches are verified for **100% output equivalence** across arbitrary populations, ties, and varying elite counts.

---

## Specifications

- **Candidate Representation:** Immutable frozen dataclass `Candidate(id: str, fitness: float, genome: Any)`.
- **Ranking Rules:** Higher fitness ranks higher; equal fitness scores break ties by ascending candidate ID.
  - Example: Candidates $A$ (fitness 3), $B$ (fitness 8), $C$ (fitness 5) with `elite_count=2` selects $[B, C]$.
- **Validation:**
  - `elite_count < 0` is rejected with `ValueError`.
  - `elite_count > len(population)` is rejected with `ValueError`.
  - Non-integer or invalid input types are rejected with `TypeError`.
  - `elite_count = 0` cleanly returns an empty list `[]`.

---

## Lab Files

- `example.py`: Implementations of `select_elites_sorting` and `select_elites_declarative`, along with a comparison demonstration CLI.
- `test_example.py`: Comprehensive test suite testing immutability, tie-breaking, error conditions, and 100% equivalence over randomized simulations.
- `README.md`: Lab documentation and conceptual explanation.

---

## Running the Code

### Execute Runnable Demonstration
```bash
python3 lessons/6-Other/21-GeneticAlgorithms/applied-labs/r38-t023/example.py
```

### Run Unit Tests
```bash
python3 -m unittest discover -s lessons/6-Other/21-GeneticAlgorithms/applied-labs/r38-t023 -p "test_*.py"
```
