"""
Unit tests for R38-T030: Capability Negotiation — Gated Dispatch & Evaluation Experiment.

Test classes:
  TestNegotiationResult        — NegotiationResult.build(): intersection, ratio, validation
  TestCapabilityGate           — CapabilityGate: negotiate, register, dispatch, error paths
  TestGatedDispatchRejection   — Rejection for un-negotiated capabilities
  TestHandlerRegistration      — Duplicate handler and un-negotiated handler errors
  TestEvaluationExperiment     — run_evaluation_experiment() with all three configurations
"""

import unittest

from example import (
    CapabilityGate,
    CapabilityNotNegotiatedError,
    DuplicateCapabilityHandlerError,
    EmptyCapabilitySetError,
    ExperimentConfig,
    NegotiationResult,
    run_evaluation_experiment,
)


# ---------------------------------------------------------------------------
# NegotiationResult tests
# ---------------------------------------------------------------------------

class TestNegotiationResult(unittest.TestCase):

    def test_canonical_example_intersection(self):
        """client={tools,resources} ∩ server={tools,prompts} = {tools}."""
        r = NegotiationResult.build({"tools", "resources"}, {"tools", "prompts"})
        self.assertEqual(r.negotiated, frozenset({"tools"}))

    def test_canonical_ratio(self):
        """negotiation_ratio = 1 negotiated / 2 client caps = 0.5."""
        r = NegotiationResult.build({"tools", "resources"}, {"tools", "prompts"})
        self.assertAlmostEqual(r.negotiation_ratio, 0.5)

    def test_disjoint_sets_empty_negotiation(self):
        r = NegotiationResult.build({"resources", "prompts"}, {"tools", "sampling"})
        self.assertEqual(r.negotiated, frozenset())
        self.assertAlmostEqual(r.negotiation_ratio, 0.0)

    def test_identical_sets_full_negotiation(self):
        r = NegotiationResult.build({"tools", "resources"}, {"tools", "resources"})
        self.assertEqual(r.negotiated, frozenset({"tools", "resources"}))
        self.assertAlmostEqual(r.negotiation_ratio, 1.0)

    def test_client_caps_stored(self):
        r = NegotiationResult.build({"tools", "resources"}, {"tools", "prompts"})
        self.assertEqual(r.client_caps, frozenset({"tools", "resources"}))

    def test_server_caps_stored(self):
        r = NegotiationResult.build({"tools", "resources"}, {"tools", "prompts"})
        self.assertEqual(r.server_caps, frozenset({"tools", "prompts"}))

    def test_empty_client_raises(self):
        with self.assertRaises(EmptyCapabilitySetError):
            NegotiationResult.build(set(), {"tools"})

    def test_empty_server_raises(self):
        with self.assertRaises(EmptyCapabilitySetError):
            NegotiationResult.build({"tools"}, set())

    def test_non_string_capability_raises(self):
        with self.assertRaises(TypeError):
            NegotiationResult.build({1, "tools"}, {"tools"})


# ---------------------------------------------------------------------------
# CapabilityGate — negotiation and properties
# ---------------------------------------------------------------------------

class TestCapabilityGate(unittest.TestCase):

    def setUp(self):
        self.gate = CapabilityGate(
            client_caps={"tools", "resources"},
            server_caps={"tools", "prompts"},
        )

    def test_negotiated_set(self):
        self.assertEqual(self.gate.negotiated_caps, frozenset({"tools"}))

    def test_negotiation_ratio(self):
        self.assertAlmostEqual(self.gate.negotiation_ratio, 0.5)

    def test_dispatch_negotiated_capability(self):
        """Dispatching a negotiated capability invokes the handler."""
        self.gate.register_handler("tools", lambda p: f"result:{p}")
        dr = self.gate.dispatch("tools", "my_tool")
        self.assertEqual(dr.capability, "tools")
        self.assertEqual(dr.handler_return, "result:my_tool")
        self.assertGreaterEqual(dr.latency_seconds, 0.0)

    def test_dispatch_returns_handler_value(self):
        self.gate.register_handler("tools", lambda p: 42)
        dr = self.gate.dispatch("tools", None)
        self.assertEqual(dr.handler_return, 42)

    def test_dispatch_latency_non_negative(self):
        self.gate.register_handler("tools", lambda p: p)
        dr = self.gate.dispatch("tools", "x")
        self.assertGreaterEqual(dr.latency_seconds, 0.0)


# ---------------------------------------------------------------------------
# Gated dispatch — rejection for un-negotiated capabilities
# ---------------------------------------------------------------------------

class TestGatedDispatchRejection(unittest.TestCase):

    def setUp(self):
        self.gate = CapabilityGate(
            client_caps={"tools", "resources"},
            server_caps={"tools", "prompts"},
        )
        self.gate.register_handler("tools", lambda p: p)

    def test_resources_rejected(self):
        """'resources' is in client set but not negotiated → rejected."""
        with self.assertRaises(CapabilityNotNegotiatedError):
            self.gate.dispatch("resources", "some_resource")

    def test_prompts_rejected(self):
        """'prompts' is in server set but not negotiated → rejected."""
        with self.assertRaises(CapabilityNotNegotiatedError):
            self.gate.dispatch("prompts", "my_prompt")

    def test_unknown_capability_rejected(self):
        """Entirely unknown capability → rejected."""
        with self.assertRaises(CapabilityNotNegotiatedError):
            self.gate.dispatch("sampling", None)

    def test_state_unchanged_after_rejection(self):
        """Gate's negotiated set remains intact after a rejected dispatch."""
        try:
            self.gate.dispatch("resources", "x")
        except CapabilityNotNegotiatedError:
            pass
        # Can still dispatch the valid one
        dr = self.gate.dispatch("tools", "ok")
        self.assertEqual(dr.handler_return, "ok")

    def test_rejection_error_message_contains_capability(self):
        """Error message mentions the rejected capability name."""
        with self.assertRaises(CapabilityNotNegotiatedError) as ctx:
            self.gate.dispatch("resources", None)
        self.assertIn("resources", str(ctx.exception))


# ---------------------------------------------------------------------------
# Handler registration errors
# ---------------------------------------------------------------------------

class TestHandlerRegistration(unittest.TestCase):

    def test_duplicate_handler_raises(self):
        gate = CapabilityGate({"tools"}, {"tools"})
        gate.register_handler("tools", lambda p: p)
        with self.assertRaises(DuplicateCapabilityHandlerError):
            gate.register_handler("tools", lambda p: p)

    def test_register_un_negotiated_raises(self):
        gate = CapabilityGate({"tools", "resources"}, {"tools", "prompts"})
        with self.assertRaises(CapabilityNotNegotiatedError):
            gate.register_handler("resources", lambda p: p)

    def test_no_handler_registered_raises_key_error(self):
        gate = CapabilityGate({"tools"}, {"tools"})
        # No handler registered → KeyError
        with self.assertRaises(KeyError):
            gate.dispatch("tools", None)


# ---------------------------------------------------------------------------
# Evaluation experiment
# ---------------------------------------------------------------------------

class TestEvaluationExperiment(unittest.TestCase):

    def setUp(self):
        self.results = run_evaluation_experiment(trials=10)
        self.by_name = {r.config_name: r for r in self.results}

    def test_three_results_returned(self):
        self.assertEqual(len(self.results), 3)

    def test_config_a_disjoint_no_negotiation(self):
        r = self.by_name["A_disjoint"]
        self.assertEqual(r.negotiated, frozenset())
        self.assertAlmostEqual(r.negotiation_ratio, 0.0)
        self.assertFalse(r.dispatch_succeeded)

    def test_config_b_partial_negotiates_tools(self):
        r = self.by_name["B_partial"]
        self.assertIn("tools", r.negotiated)
        self.assertAlmostEqual(r.negotiation_ratio, 0.5)
        self.assertTrue(r.dispatch_succeeded)

    def test_config_c_identical_full_negotiation(self):
        r = self.by_name["C_identical"]
        self.assertEqual(r.negotiated, frozenset({"tools", "resources"}))
        self.assertAlmostEqual(r.negotiation_ratio, 1.0)
        self.assertTrue(r.dispatch_succeeded)

    def test_latency_non_negative_for_successful_dispatches(self):
        for r in self.results:
            if r.dispatch_succeeded:
                self.assertGreaterEqual(r.latency_seconds, 0.0)

    def test_disjoint_latency_zero(self):
        r = self.by_name["A_disjoint"]
        self.assertAlmostEqual(r.latency_seconds, 0.0)

    def test_custom_config(self):
        """Custom single config with identical sets."""
        custom = [
            ExperimentConfig(
                name="X_custom",
                description="Single custom config",
                client_caps={"alpha", "beta"},
                server_caps={"alpha", "gamma"},
            )
        ]
        results = run_evaluation_experiment(configs=custom, trials=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].negotiated, frozenset({"alpha"}))
        self.assertAlmostEqual(results[0].negotiation_ratio, 0.5)
        self.assertTrue(results[0].dispatch_succeeded)

    def test_negotiation_ratio_bounded(self):
        for r in self.results:
            self.assertGreaterEqual(r.negotiation_ratio, 0.0)
            self.assertLessEqual(r.negotiation_ratio, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
