"""Automated tests for Capability Negotiation (R38-T042)."""

from __future__ import annotations

import unittest

from example import (
    CLIENT_CAPABILITIES,
    SERVER_CAPABILITIES,
    UnsupportedCapabilityError,
    get_dispatch_call_log,
    invoke_capability,
    invoke_capability_baseline,
    negotiate_capabilities,
    reset_dispatch_call_log,
    validate_capability,
)


class TestCapabilityNegotiation(unittest.TestCase):
    """Tests verifying capability intersection logic."""

    def test_default_capabilities_intersection(self) -> None:
        """Client {'tools', 'resources'} and Server {'tools', 'prompts'} negotiate to {'tools'}."""
        negotiated = negotiate_capabilities(CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        self.assertEqual(negotiated, {"tools"})

    def test_disjoint_capabilities_yields_empty_set(self) -> None:
        """Disjoint capability sets negotiate to an empty set."""
        client_caps = {"tools"}
        server_caps = {"prompts"}
        negotiated = negotiate_capabilities(client_caps, server_caps)
        self.assertEqual(negotiated, set())

    def test_identical_capabilities(self) -> None:
        """Identical capability sets negotiate to the exact same set."""
        caps = {"tools", "resources", "prompts"}
        negotiated = negotiate_capabilities(caps, set(caps))
        self.assertEqual(negotiated, caps)

    def test_empty_capability_sets(self) -> None:
        """An empty set from client, server, or both always negotiates to empty set."""
        self.assertEqual(negotiate_capabilities(set(), {"tools"}), set())
        self.assertEqual(negotiate_capabilities({"tools"}, set()), set())
        self.assertEqual(negotiate_capabilities(set(), set()), set())


class TestCapabilityValidationAndDispatch(unittest.TestCase):
    """Tests verifying invocation, pre-dispatch validation, and dispatch isolation."""

    def setUp(self) -> None:
        reset_dispatch_call_log()

    def tearDown(self) -> None:
        reset_dispatch_call_log()

    def test_successful_invocation_of_agreed_capability(self) -> None:
        """Agreed capability 'tools' must be allowed and dispatched."""
        result = invoke_capability("tools", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["capability"], "tools")
        self.assertIn("Executed tool", result["result"])

        # Dispatch log must show exactly 1 call
        calls = get_dispatch_call_log()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["capability"], "tools")

    def test_rejection_of_resources(self) -> None:
        """Capability 'resources' is supported by client but not server; must be rejected."""
        with self.assertRaises(UnsupportedCapabilityError) as ctx:
            invoke_capability("resources", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        self.assertIn("resources", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    def test_rejection_of_prompts(self) -> None:
        """Capability 'prompts' is supported by server but not client; must be rejected."""
        with self.assertRaises(UnsupportedCapabilityError) as ctx:
            invoke_capability("prompts", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        self.assertIn("prompts", str(ctx.exception))
        self.assertIn("not supported", str(ctx.exception))

    def test_rejected_capabilities_never_reach_dispatch(self) -> None:
        """Ensure no dispatch handler is called when an unsupported capability is requested."""
        unsupported_calls = ["resources", "prompts", "unadvertised_cap"]
        for cap in unsupported_calls:
            with self.subTest(capability=cap):
                reset_dispatch_call_log()
                with self.assertRaises(UnsupportedCapabilityError):
                    invoke_capability(cap, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
                # Dispatch log must remain completely empty!
                self.assertEqual(
                    len(get_dispatch_call_log()),
                    0,
                    f"Dispatch was reached for rejected capability '{cap}'!",
                )

    def test_empty_capabilities_reject_all_invocations(self) -> None:
        """When negotiated set is empty, calling any capability is rejected before dispatch."""
        client_empty: set[str] = set()
        server_empty: set[str] = set()

        for c_caps, s_caps in [
            (client_empty, SERVER_CAPABILITIES),
            (CLIENT_CAPABILITIES, server_empty),
            (client_empty, server_empty),
        ]:
            reset_dispatch_call_log()
            with self.assertRaises(UnsupportedCapabilityError):
                invoke_capability("tools", c_caps, s_caps)
            self.assertEqual(len(get_dispatch_call_log()), 0)

    def test_boundary_payload_forwarding(self) -> None:
        """Test with None payload and structured dictionary payload."""
        payload = {"action": "resize", "width": 256, "height": 256}
        result = invoke_capability("tools", CLIENT_CAPABILITIES, SERVER_CAPABILITIES, payload=payload)
        self.assertEqual(result["status"], "success")
        calls = get_dispatch_call_log()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["payload"], payload)

    def test_boundary_case_sensitivity(self) -> None:
        """Capability names must match casing exactly ('Tools' != 'tools')."""
        with self.assertRaises(UnsupportedCapabilityError):
            invoke_capability("Tools", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        self.assertEqual(len(get_dispatch_call_log()), 0)


class TestExtractedValidationFunction(unittest.TestCase):
    """Tests verifying the isolated validate_capability guard function."""

    def test_validate_capability_passes_for_agreed_capability(self) -> None:
        """Valid capability in negotiated set passes without exception."""
        # Should execute silently without raising
        validate_capability("tools", {"tools"})
        validate_capability("resources", {"tools", "resources"})

    def test_validate_capability_raises_for_missing_capability(self) -> None:
        """Missing capability raises UnsupportedCapabilityError."""
        with self.assertRaises(UnsupportedCapabilityError):
            validate_capability("resources", {"tools"})
        with self.assertRaises(UnsupportedCapabilityError):
            validate_capability("prompts", {"tools"})
        with self.assertRaises(UnsupportedCapabilityError):
            validate_capability("tools", set())


class TestInvalidInputHandling(unittest.TestCase):
    """Tests verifying validation on malformed inputs and data types."""

    def test_non_string_capability_rejected(self) -> None:
        """Non-string capability argument raises TypeError."""
        for invalid in [123, None, ["tools"], {"cap": "tools"}]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    invoke_capability(invalid, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)  # type: ignore[arg-type]

    def test_non_set_capabilities_rejected(self) -> None:
        """Providing lists or non-sets as capability collections raises TypeError."""
        with self.assertRaises(TypeError):
            negotiate_capabilities(["tools"], SERVER_CAPABILITIES)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            negotiate_capabilities(CLIENT_CAPABILITIES, ("tools", "prompts"))  # type: ignore[arg-type]

    def test_non_string_element_in_sets_rejected(self) -> None:
        """Sets containing non-string members raise TypeError."""
        with self.assertRaises(TypeError):
            negotiate_capabilities({1, 2}, SERVER_CAPABILITIES)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            validate_capability("tools", {123})  # type: ignore[arg-type]


class TestBaselineAndRefactoredEquivalence(unittest.TestCase):
    """Tests proving identical behavior between baseline and refactored implementations."""

    def setUp(self) -> None:
        reset_dispatch_call_log()

    def tearDown(self) -> None:
        reset_dispatch_call_log()

    def test_equivalence_on_agreed_capability(self) -> None:
        """Both baseline and refactored return equivalent results for agreed capability."""
        reset_dispatch_call_log()
        res_baseline = invoke_capability_baseline("tools", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        calls_baseline = get_dispatch_call_log()

        reset_dispatch_call_log()
        res_refactored = invoke_capability("tools", CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        calls_refactored = get_dispatch_call_log()

        self.assertEqual(res_baseline["status"], res_refactored["status"])
        self.assertEqual(res_baseline["capability"], res_refactored["capability"])
        self.assertEqual(res_baseline["result"], res_refactored["result"])
        self.assertEqual(len(calls_baseline), 1)
        self.assertEqual(len(calls_refactored), 1)

    def test_equivalence_on_unsupported_capabilities(self) -> None:
        """Both baseline and refactored raise UnsupportedCapabilityError and never reach dispatch."""
        test_caps = ["resources", "prompts", "unknown_service"]
        for cap in test_caps:
            with self.subTest(capability=cap):
                # Baseline
                reset_dispatch_call_log()
                with self.assertRaises(UnsupportedCapabilityError) as ctx_base:
                    invoke_capability_baseline(cap, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
                self.assertEqual(len(get_dispatch_call_log()), 0)

                # Refactored
                reset_dispatch_call_log()
                with self.assertRaises(UnsupportedCapabilityError) as ctx_ref:
                    invoke_capability(cap, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
                self.assertEqual(len(get_dispatch_call_log()), 0)

                # Error messages must convey the same unsupported capability
                self.assertIn(cap, str(ctx_base.exception))
                self.assertIn(cap, str(ctx_ref.exception))

    def test_equivalence_on_type_errors(self) -> None:
        """Both baseline and refactored raise TypeError on invalid input types."""
        with self.assertRaises(TypeError):
            invoke_capability_baseline(999, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            invoke_capability(999, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

