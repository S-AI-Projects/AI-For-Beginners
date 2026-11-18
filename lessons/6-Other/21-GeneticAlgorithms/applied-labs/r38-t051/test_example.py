import unittest
import sys
import os

# Support running directly or via unittest discover
sys.path.insert(0, os.path.dirname(__file__))
from example import Candidate, select_elites

class TestGeneticSelection(unittest.TestCase):
    def test_stated_scenario(self):
        # Fitness A=3,B=8,C=5 with elite_count 2 selects B,C
        population = [
            Candidate(id="A", fitness=3.0),
            Candidate(id="B", fitness=8.0),
            Candidate(id="C", fitness=5.0),
        ]
        elites = select_elites(population, 2)
        self.assertEqual([c.id for c in elites], ["B", "C"])
        self.assertEqual([c.fitness for c in elites], [8.0, 5.0])

    def test_tie_breaking_by_candidate_id(self):
        # Ties use candidate ID in alphabetical order
        population = [
            Candidate(id="Z", fitness=10.0),
            Candidate(id="A", fitness=10.0),
            Candidate(id="M", fitness=10.0),
        ]
        elites = select_elites(population, 2)
        self.assertEqual([c.id for c in elites], ["A", "M"])

    def test_count_beyond_population_rejected(self):
        population = [
            Candidate(id="A", fitness=3.0),
            Candidate(id="B", fitness=8.0),
        ]
        with self.assertRaises(ValueError):
            select_elites(population, 3)
        with self.assertRaises(ValueError):
            select_elites(population, -1)

    def test_immutability(self):
        candidate = Candidate(id="A", fitness=3.0)
        population = [candidate]
        elites = select_elites(population, 1)
        self.assertIs(elites[0], candidate)
        with self.assertRaises(Exception):
            candidate.fitness = 9.0  # type: ignore

    def test_deliberate_failure_check(self):
        # Deliberate wrong expectation verification
        population = [
            Candidate(id="A", fitness=3.0),
            Candidate(id="B", fitness=8.0),
        ]
        elites = select_elites(population, 1)
        self.assertNotEqual([c.id for c in elites], ["A"])

if __name__ == "__main__":
    unittest.main()
