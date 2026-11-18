import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import (
    topological_sort_kahn,
    topological_sort_table,
    MissingDependencyError,
    CycleDetectedError,
)

class TestDependencyPlan(unittest.TestCase):
    def test_stated_scenario(self):
        # A has no dependencies, B depends on A, C on B: order A,B,C
        plan = {"A": [], "B": ["A"], "C": ["B"]}
        order1 = topological_sort_kahn(plan)
        order2 = topological_sort_table(plan)
        self.assertEqual(order1, ["A", "B", "C"])
        self.assertEqual(order2, ["A", "B", "C"])

    def test_cycle_rejected(self):
        # A -> B -> A is rejected
        plan = {"A": ["B"], "B": ["A"]}
        with self.assertRaises(CycleDetectedError):
            topological_sort_kahn(plan)
        with self.assertRaises(CycleDetectedError):
            topological_sort_table(plan)

    def test_unknown_dependency_reported_by_id(self):
        plan = {"A": ["UNKNOWN_X"]}
        with self.assertRaises(MissingDependencyError) as ctx:
            topological_sort_kahn(plan)
        self.assertEqual(ctx.exception.missing_dep, "UNKNOWN_X")
        self.assertEqual(ctx.exception.task_id, "A")

    def test_deliberate_failure_check(self):
        plan = {"A": [], "B": ["A"]}
        order = topological_sort_kahn(plan)
        self.assertNotEqual(order, ["B", "A"])

if __name__ == "__main__":
    unittest.main()
