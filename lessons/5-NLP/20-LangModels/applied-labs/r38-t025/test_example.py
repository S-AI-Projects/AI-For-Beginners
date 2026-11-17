"""
Unit tests for prompt-template escaping and offline service adapter (Issue #25 / R38-T025).
"""

import unittest
from example import (
    MissingKeyError,
    ModelRequest,
    ModelResponse,
    OfflineModelAdapter,
    TemplateError,
    TokenUsage,
    extract_placeholders,
    render_template,
)


class TestPromptTemplateAndOfflineAdapter(unittest.TestCase):
    def test_safe_template_rendering(self):
        """Render variables into template placeholders without remote network access."""
        template = "Instruction: {instruction}. Target: {target}."
        variables = {"instruction": "Summarize", "target": "Quarterly Report"}
        rendered = render_template(template, variables)
        self.assertEqual(rendered, "Instruction: Summarize. Target: Quarterly Report.")

    def test_rejection_missing_keys(self):
        """Reject template rendering when a required variable is missing."""
        template = "Query: {query}. Max results: {limit}."
        with self.assertRaises(MissingKeyError) as ctx:
            render_template(template, {"query": "Find cats"})
        self.assertIn("Missing required template variable", str(ctx.exception))
        self.assertIn("limit", str(ctx.exception))

    def test_literal_braces_in_variables_preserved(self):
        """Variables containing braces remain literal (no secondary template injection)."""
        template = "Analyze the payload: {user_payload}"
        payload_with_braces = "{ \"command\": \"format\", \"options\": { \"all\": true } }"
        variables = {"user_payload": payload_with_braces}
        rendered = render_template(template, variables)

        expected = "Analyze the payload: { \"command\": \"format\", \"options\": { \"all\": true } }"
        self.assertEqual(rendered, expected)

    def test_template_escaped_braces(self):
        """Double braces {{ and }} are unescaped to literal single braces."""
        template = "Config: {{ \"mode\": \"{mode}\" }}"
        rendered = render_template(template, {"mode": "test"})
        self.assertEqual(rendered, 'Config: { "mode": "test" }')

    def test_offline_adapter_dispatch_payload_and_return_structure(self):
        """Offline adapter validates payload and produces expected response structure."""
        adapter = OfflineModelAdapter()
        req = ModelRequest(
            prompt="Classify sentiment: Great product!",
            model="offline-gpt-3.5-mock",
            temperature=0.5,
            max_tokens=60
        )
        response = adapter.dispatch(req)

        # Verify response structure
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.model, "offline-gpt-3.5-mock")
        self.assertEqual(response.finish_reason, "stop")
        self.assertIsInstance(response.usage, TokenUsage)
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.total_tokens, 0)
        self.assertEqual(response.request_payload, req)
        self.assertEqual(len(adapter.call_history), 1)
        self.assertEqual(adapter.call_history[0], req)

    def test_offline_adapter_canned_response(self):
        """Offline adapter returns canned response when provided."""
        canned_text = '{"sentiment": "positive", "confidence": 0.98}'
        adapter = OfflineModelAdapter(canned_response=canned_text)
        req = ModelRequest(prompt="Classify sentiment: Excellent service!")
        response = adapter.dispatch(req)

        self.assertEqual(response.output_text, canned_text)
        self.assertEqual(response.usage.completion_tokens, len(canned_text.split()))

    def test_offline_adapter_payload_validation_errors(self):
        """Offline adapter strictly validates invalid payload attributes."""
        adapter = OfflineModelAdapter()

        # Empty prompt
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="   "))

        # Temperature out of bounds
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="Valid prompt", temperature=-0.1))
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="Valid prompt", temperature=2.5))

        # max_tokens <= 0
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="Valid prompt", max_tokens=0))
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="Valid prompt", max_tokens=-5))

        # Empty model name
        with self.assertRaises(ValueError):
            adapter.dispatch(ModelRequest(prompt="Valid prompt", model=""))

    def test_end_to_end_rendering_to_adapter(self):
        """End-to-end integration: template rendering feeding offline adapter."""
        adapter = OfflineModelAdapter()
        template = "Summarize text for {audience}: {content}"
        variables = {
            "audience": "executives",
            "content": "Q3 revenue grew by 15% due to product launch.",
        }
        rendered_prompt = render_template(template, variables)
        req = ModelRequest(prompt=rendered_prompt, model="offline-gpt-4-local", max_tokens=50)
        response = adapter.dispatch(req)

        self.assertEqual(response.request_payload.prompt, rendered_prompt)
        self.assertIn("offline-gpt-4-local", response.output_text)
        self.assertEqual(len(adapter.call_history), 1)


if __name__ == "__main__":
    unittest.main()
