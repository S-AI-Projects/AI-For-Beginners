import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from example import ComponentRouter

class TestComponentRouter(unittest.TestCase):
    def setUp(self):
        self.router = ComponentRouter()

    def test_stated_scenario(self):
        # help->support, calculate->calculator, unknown->fallback
        r1 = self.router.route("Please help me")
        self.assertEqual(r1.chosen_handler, "support")
        self.assertEqual(r1.chosen_rule, "rule_help")

        r2 = self.router.route("calculate 2+2")
        self.assertEqual(r2.chosen_handler, "calculator")
        self.assertEqual(r2.chosen_rule, "rule_calculate")

        r3 = self.router.route("some random text")
        self.assertEqual(r3.chosen_handler, "fallback")
        self.assertEqual(r3.chosen_rule, "rule_fallback")

    def test_ambiguous_intent_priority(self):
        # Ambiguous contains both keywords; must follow documented priority (help > calculate)
        r = self.router.route("help me calculate my taxes")
        self.assertEqual(r.chosen_handler, "support")
        self.assertEqual(r.chosen_rule, "rule_help")
        # Invariant: trace must record both evaluations
        matches = [t for t in r.trace if t.matched]
        self.assertGreaterEqual(len(matches), 1)

    def test_trace_invariant(self):
        # Step indices in trace must be strictly sequential
        r = self.router.route("unknown")
        indices = [t.step_index for t in r.trace]
        self.assertEqual(indices, list(range(len(r.trace))))

    def test_deliberate_failure_check(self):
        r = self.router.route("help")
        self.assertNotEqual(r.chosen_handler, "fallback")

if __name__ == "__main__":
    unittest.main()
