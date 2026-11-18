# Applied Lab R38-T051: Genetic Selection and Deterministic Ranking

## Overview
This applied lab extends the "Genetic Algorithms" lesson by demonstrating deterministic fitness ranking and elite selection on a discrete population of candidate solutions, preserving immutable candidate records.

## Problem Formulation & Contract
- **Input**: A sequence of immutable `Candidate(id: str, fitness: float)` records and an integer `elite_count`.
- **Ranking Criteria**:
  1. Descending fitness score (higher fitness is superior).
  2. Ascending candidate ID for deterministic tie-breaking.
- **Output**: A tuple of `elite_count` selected candidate records.
- **Boundary & Validation Rules**:
  - `elite_count > len(population)` or `elite_count < 0` raises `ValueError`.
  - Candidate records are frozen/immutable.

## Example Scenario
- Input Population: `[Candidate(id='A', fitness=3.0), Candidate(id='B', fitness=8.0), Candidate(id='C', fitness=5.0)]`
- Selection Parameter: `elite_count = 2`
- Selected Output: `('B', 'C')` with fitnesses `[8.0, 5.0]`

## Scope & Limitations
- This is a minimal, self-contained educational demonstration using the Python standard library.
- No GPU, external libraries, cloud services, or real-world hardware are required.
- Candidate fitness values are provided directly rather than evaluated through domain-specific objective functions.

## Running Tests
Run the unit test suite:
```bash
python3 -m unittest discover -s lessons/6-Other/21-GeneticAlgorithms/applied-labs/r38-t051 -p "test_*.py"
```
