"""
Unit tests for prompt-template escaping engine with invariant and state trace (Issue #24 / R38-T024).
"""

import unittest
from example import (
    MissingKeyError,
    RenderResult,
    RenderStepTrace,
    TemplateError,
    UnknownKeyError,
    extract_placeholders,
    render_prompt_template,
)


class TestPromptTemplateEngine(unittest.TestCase):
    def test_render_with_declared_variables(self):
        """Render templates from declared variables."""
        template = "Hello {name}, your role is {role}."
        variables = {"name": "Alice", "role": "Engineer"}
        result = render_prompt_template(template, variables)

        self.assertEqual(result.rendered_prompt, "Hello Alice, your role is Engineer.")
        self.assertEqual(len(result.state_trace), 4)

    def test_reject_missing_keys(self):
        """Reject template execution when required keys are missing."""
        template = "Hello {name}, your role is {role}."
        variables = {"name": "Alice"}
        with self.assertRaises(MissingKeyError) as ctx:
            render_prompt_template(template, variables)
        self.assertIn("Missing required template variable", str(ctx.exception))
        self.assertIn("role", str(ctx.exception))

    def test_reject_unknown_keys(self):
        """Reject template execution when unknown extraneous keys are supplied."""
        template = "Hello {name}."
        variables = {"name": "Alice", "extra": "bonus"}
        with self.assertRaises(UnknownKeyError) as ctx:
            render_prompt_template(template, variables)
        self.assertIn("Unknown variable(s) provided", str(ctx.exception))
        self.assertIn("extra", str(ctx.exception))

    def test_user_data_braces_remain_literal_no_secondary_injection(self):
        """Braces inside user data remain literal and are not recursively substituted."""
        template = "System: Analyze query.\nQuery: {user_input}\nContext: {context}"
        # Adversarial attempt: user input contains '{context}' and '{injected}'
        user_input_with_braces = "Check {context} and {injected} and function() { return 1; }"
        variables = {
            "user_input": user_input_with_braces,
            "context": "Production DB",
        }
        result = render_prompt_template(template, variables)

        expected = (
            "System: Analyze query.\n"
            "Query: Check {context} and {injected} and function() { return 1; }\n"
            "Context: Production DB"
        )
        self.assertEqual(result.rendered_prompt, expected)
        # Verify that the literal braces '{context}' in user_input were NOT replaced by 'Production DB'
        self.assertIn("Check {context} and {injected}", result.rendered_prompt)

    def test_template_escaped_braces(self):
        """Template literal escaped braces {{ and }} are unescaped to { and }."""
        template = "JSON snippet: {{ \"key\": \"{val}\" }}"
        variables = {"val": "123"}
        result = render_prompt_template(template, variables)
        self.assertEqual(result.rendered_prompt, 'JSON snippet: { "key": "123" }')

    def test_state_trace_and_invariants(self):
        """Test that state trace is recorded and satisfies step invariants."""
        template = "Prefix: {greeting} {target}!"
        variables = {"greeting": "Hi", "target": "World"}
        result = render_prompt_template(template, variables)

        step_names = [trace.step_name for trace in result.state_trace]
        self.assertEqual(
            step_names,
            ["parse_placeholders", "validate_keys", "bind_literals", "substitute"]
        )

        # Invariant 1: Parsed placeholders match variables keys
        trace_step1 = result.state_trace[0]
        self.assertEqual(trace_step1.state_data["placeholders"], ["greeting", "target"])

        # Invariant 2: Validate keys matched exactly
        trace_step2 = result.state_trace[1]
        self.assertEqual(trace_step2.state_data["provided_keys"], ["greeting", "target"])

        # Invariant 3: Literals bound
        trace_step3 = result.state_trace[2]
        self.assertEqual(trace_step3.state_data["string_values"], variables)

        # Invariant 4: Final rendered output links directly to step 4 state
        trace_step4 = result.state_trace[3]
        self.assertEqual(trace_step4.state_data["rendered_prompt"], result.rendered_prompt)
        self.assertEqual(trace_step4.state_data["output_length"], len(result.rendered_prompt))

    def test_empty_template_and_variables(self):
        """Empty template with empty variables dict succeeds cleanly."""
        result = render_prompt_template("", {})
        self.assertEqual(result.rendered_prompt, "")
        self.assertEqual(len(result.state_trace), 4)

    def test_type_rejection(self):
        """Reject invalid template or variable types."""
        with self.assertRaises(TypeError):
            render_prompt_template(123, {})  # type: ignore
        with self.assertRaises(TypeError):
            render_prompt_template("Hello {name}", ["Alice"])  # type: ignore
        with self.assertRaises(TypeError):
            render_prompt_template("Hello {data}", {"data": [1, 2, 3]})  # type: ignore


if __name__ == "__main__":
    unittest.main()
