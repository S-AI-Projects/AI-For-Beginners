import unittest

import example


class TestPreflight(unittest.TestCase):
    def test_normal_budget_3(self):
        r = example.preflight("one two three four", 3)
        self.assertEqual(r["retained"], ["one", "two", "three"])
        self.assertEqual(r["removed"], ["four"])
        example.check_invariant(r)

    def test_boundary_exact_fit(self):
        r = example.preflight("one two", 2)
        self.assertEqual(r["retained"], ["one", "two"])
        self.assertEqual(r["removed"], [])

    def test_boundary_zero_budget(self):
        r = example.preflight("one two", 0)
        self.assertEqual(r["retained"], [])
        self.assertEqual(r["removed"], ["one", "two"])

    def test_empty_input_retains_none(self):
        r = example.preflight("", 3)
        self.assertEqual(r["retained"], [])
        self.assertEqual(r["removed"], [])
        example.check_invariant(r)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            example.preflight("one two", -1)

    def test_non_integer_budget_rejected(self):
        with self.assertRaises(TypeError):
            example.preflight("one two", "3")

    def test_state_recorded_after_each_step(self):
        r = example.preflight("one two three four", 3)
        self.assertEqual([s["step"] for s in r["trace"]], ["tokenize", "retain", "report"])
        self.assertEqual(r["trace"][0]["tokens"], ["one", "two", "three", "four"])
        self.assertEqual(r["trace"][1]["retained"], ["one", "two", "three"])
        self.assertEqual(r["trace"][2]["removed"], ["four"])

    def test_invariant_rejects_deliberate_wrong_result(self):
        r = example.preflight("one two three four", 3)
        r["retained"] = ["one", "two", "three", "four"]  # deliberately wrong
        with self.assertRaises(AssertionError):
            example.check_invariant(r)

    def test_render_matches_readme(self):
        out = example.render(example.preflight("one two three four", 3))
        self.assertEqual(out.splitlines()[2], "Retained: [one,two,three]")
        self.assertEqual(out.splitlines()[3], "Removed: 1 token(s) [four]")

    def test_label_says_teaching_approximation(self):
        self.assertIn("teaching approximation", example.TOKENIZER_LABEL)


if __name__ == "__main__":
    unittest.main()
