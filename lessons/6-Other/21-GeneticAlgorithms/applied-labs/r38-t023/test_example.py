"""
Unit tests for deterministic fitness ranking and elite selection (Issue #23 / R38-T023).

Tests:
1. Immutability of candidate population.
2. Selection correctness: Fitness A=3, B=8, C=5 with elite_count=2 selects [B, C].
3. Deterministic tie-breaking by candidate ID.
4. Error handling: elite_count > population size, negative elite_count, invalid types.
5. 100% equivalence comparison between direct sorting and declarative/lookup approaches.
"""

import unittest
from example import (
    Candidate,
    select_elites_declarative,
    select_elites_sorting,
)


class TestGeneticSelection(unittest.TestCase):
    def test_example_scenario(self):
        """Fitness A=3, B=8, C=5 with elite_count=2 selects [B, C]."""
        population = [
            Candidate(id="A", fitness=3.0),
            Candidate(id="B", fitness=8.0),
            Candidate(id="C", fitness=5.0),
        ]
        selected_sort = select_elites_sorting(population, 2)
        selected_decl = select_elites_declarative(population, 2)

        expected_ids = ["B", "C"]
        self.assertEqual([c.id for c in selected_sort], expected_ids)
        self.assertEqual([c.id for c in selected_decl], expected_ids)
        self.assertEqual(selected_sort, selected_decl)

    def test_population_immutability(self):
        """Population candidates must remain immutable and original list untouched."""
        c_a = Candidate(id="A", fitness=3.0, genome=[1, 2, 3])
        c_b = Candidate(id="B", fitness=8.0, genome=[4, 5, 6])
        population = (c_a, c_b)

        # Attempt to modify frozen dataclass should raise error
        with self.assertRaises((AttributeError, TypeError)):
            c_a.fitness = 10.0  # type: ignore

        # Run selection
        res_sort = select_elites_sorting(population, 1)
        res_decl = select_elites_declarative(population, 1)

        # Ensure candidates retain original properties
        self.assertEqual(c_a.fitness, 3.0)
        self.assertEqual(c_b.fitness, 8.0)
        self.assertEqual(c_a.genome, [1, 2, 3])
        self.assertEqual(population[0], c_a)
        self.assertEqual(population[1], c_b)

    def test_deterministic_tie_breaking(self):
        """Ties in fitness broken deterministically by candidate ID ascending."""
        population = [
            Candidate(id="Z", fitness=10.0),
            Candidate(id="A", fitness=10.0),
            Candidate(id="M", fitness=10.0),
            Candidate(id="B", fitness=5.0),
        ]
        # Top 2 should be 'A' and 'M' (both fitness 10.0, 'A' < 'M' < 'Z')
        res_sort = select_elites_sorting(population, 2)
        res_decl = select_elites_declarative(population, 2)

        self.assertEqual([c.id for c in res_sort], ["A", "M"])
        self.assertEqual([c.id for c in res_decl], ["A", "M"])
        self.assertEqual(res_sort, res_decl)

    def test_rejection_negative_elite_count(self):
        """Negative elite_count must be rejected with ValueError."""
        population = [Candidate(id="A", fitness=1.0)]
        with self.assertRaises(ValueError) as ctx:
            select_elites_sorting(population, -1)
        self.assertIn("cannot be negative", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            select_elites_declarative(population, -1)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_rejection_elite_count_exceeds_population(self):
        """elite_count > population size must be rejected with ValueError."""
        population = [Candidate(id="A", fitness=1.0), Candidate(id="B", fitness=2.0)]
        with self.assertRaises(ValueError) as ctx:
            select_elites_sorting(population, 3)
        self.assertIn("cannot exceed population size", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            select_elites_declarative(population, 3)
        self.assertIn("cannot exceed population size", str(ctx.exception))

    def test_zero_elite_count(self):
        """elite_count = 0 returns empty list."""
        population = [Candidate(id="A", fitness=1.0), Candidate(id="B", fitness=2.0)]
        self.assertEqual(select_elites_sorting(population, 0), [])
        self.assertEqual(select_elites_declarative(population, 0), [])

    def test_select_all_candidates(self):
        """elite_count = len(population) returns all candidates ranked."""
        population = [
            Candidate(id="A", fitness=3.0),
            Candidate(id="B", fitness=8.0),
            Candidate(id="C", fitness=5.0),
        ]
        res_sort = select_elites_sorting(population, 3)
        res_decl = select_elites_declarative(population, 3)
        expected = ["B", "C", "A"]
        self.assertEqual([c.id for c in res_sort], expected)
        self.assertEqual([c.id for c in res_decl], expected)
        self.assertEqual(res_sort, res_decl)

    def test_type_validation(self):
        """Validate type checks for elite_count and candidates."""
        population = [Candidate(id="A", fitness=1.0)]
        with self.assertRaises(TypeError):
            select_elites_sorting(population, "1")  # type: ignore
        with self.assertRaises(TypeError):
            select_elites_declarative(population, "1")  # type: ignore
        with self.assertRaises(TypeError):
            select_elites_sorting(population, 1.5)  # type: ignore
        with self.assertRaises(TypeError):
            select_elites_sorting(population, True)  # type: ignore
        with self.assertRaises(TypeError):
            select_elites_sorting(["not_a_candidate"], 1)  # type: ignore
        with self.assertRaises(TypeError):
            select_elites_declarative(["not_a_candidate"], 1)  # type: ignore

    def test_comprehensive_100_percent_equivalence_across_randomized_scenarios(self):
        """
        Compare the direct sorting approach and the declarative/lookup approach
        across diverse configurations (ties, reverse orders, duplicates, varying elite counts).
        Verifies 100% equivalence.
        """
        import random
        rng = random.Random(42)

        for _ in range(50):
            pop_size = rng.randint(1, 25)
            elite_count = rng.randint(0, pop_size)
            # Create candidates with deliberate duplicate fitness scores to trigger tie-breaking
            population = [
                Candidate(
                    id=f"Cand_{i:03d}",
                    fitness=float(rng.randint(0, 10)),
                    genome=f"gene_{i}"
                )
                for i in range(pop_size)
            ]
            # Shuffle order to test stability and tie-break
            rng.shuffle(population)

            res_sort = select_elites_sorting(population, elite_count)
            res_decl = select_elites_declarative(population, elite_count)

            self.assertEqual(
                res_sort,
                res_decl,
                f"Mismatch found between approaches for pop_size={pop_size}, elite_count={elite_count}"
            )


if __name__ == "__main__":
    unittest.main()
