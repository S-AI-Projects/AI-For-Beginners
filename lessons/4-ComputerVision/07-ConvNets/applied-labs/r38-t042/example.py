"""Capability Negotiation & Pre-Dispatch Validation (R38-T042).

TEACHING APPROXIMATION: This is a local educational simulation of capability
negotiation and pre-dispatch feature gating. It uses pure Python standard
library to demonstrate how client and server feature sets intersect, and why
unsupported capabilities must be rejected before reaching internal dispatch handlers.
It does not implement an actual network protocol or production MCP client/server.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

# Default capability sets for the educational scenario
CLIENT_CAPABILITIES: set[str] = {"tools", "resources"}
SERVER_CAPABILITIES: set[str] = {"tools", "prompts"}

# Tracking dispatch executions to prove whether dispatch was reached
_dispatch_call_log: list[dict[str, Any]] = []


class UnsupportedCapabilityError(Exception):
    """Raised when an operation is requested for a capability not agreed upon during negotiation."""
    pass


def reset_dispatch_call_log() -> None:
    """Clear the record of dispatched calls (useful for test isolation)."""
    _dispatch_call_log.clear()


def get_dispatch_call_log() -> list[dict[str, Any]]:
    """Return a copy of the recorded dispatch calls."""
    return list(_dispatch_call_log)


# --- Dispatch Handlers (Simulated Server-Side Implementations) ---

def handle_tools(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the 'tools' capability."""
    return {
        "status": "success",
        "capability": "tools",
        "result": f"Executed tool with payload: {payload}",
    }


def handle_resources(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Simulated handler for 'resources' (should never execute if unnegotiated)."""
    return {
        "status": "success",
        "capability": "resources",
        "result": f"Read resource with payload: {payload}",
    }


def handle_prompts(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Simulated handler for 'prompts' (should never execute if unnegotiated)."""
    return {
        "status": "success",
        "capability": "prompts",
        "result": f"Rendered prompt with payload: {payload}",
    }


DISPATCH_TABLE: dict[str, Callable[[dict[str, Any] | None], dict[str, Any]]] = {
    "tools": handle_tools,
    "resources": handle_resources,
    "prompts": handle_prompts,
}


# =====================================================================
# Phase A: Baseline Implementation
# =====================================================================
def invoke_capability_baseline(
    capability: str,
    client_caps: set[str],
    server_caps: set[str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Baseline monolithic implementation.

    In this baseline version, capability negotiation (set intersection),
    guard validation, and dispatch routing are coupled together inside a single
    function.
    """
    if not isinstance(capability, str):
        raise TypeError(f"Capability must be a string, got {type(capability).__name__}")
    if not isinstance(client_caps, set) or not isinstance(server_caps, set):
        raise TypeError("client_caps and server_caps must be sets of strings")
    for item in client_caps | server_caps:
        if not isinstance(item, str):
            raise TypeError(f"All capability elements must be strings, got {type(item).__name__}")

    # 1. Inline negotiation via set intersection
    negotiated = client_caps & server_caps

    # 2. Inline validation check
    if capability not in negotiated:
        raise UnsupportedCapabilityError(
            f"Capability '{capability}' is not supported. "
            f"Agreed negotiated capabilities: {sorted(list(negotiated))}"
        )

    # 3. Dispatch to handler
    _dispatch_call_log.append({"capability": capability, "payload": payload, "version": "baseline"})
    handler = DISPATCH_TABLE.get(capability)
    if handler is None:
        raise KeyError(f"No dispatch handler registered for capability '{capability}'")
    return handler(payload)


# =====================================================================
# Phase B: Refactored Implementation (Extracted Responsibilities)
# =====================================================================

# Extracted Responsibility 1: Capability Negotiation
def negotiate_capabilities(client_caps: set[str], server_caps: set[str]) -> set[str]:
    """Compute the mutually agreed capabilities via set intersection.

    Args:
        client_caps: Set of capability strings supported by the client.
        server_caps: Set of capability strings supported by the server.

    Returns:
        A set of mutually supported capability strings.

    Raises:
        TypeError: If client_caps or server_caps is not a set, or contains non-strings.
    """
    if not isinstance(client_caps, set) or not isinstance(server_caps, set):
        raise TypeError("client_caps and server_caps must be sets of strings")
    for item in client_caps | server_caps:
        if not isinstance(item, str):
            raise TypeError(f"All capability elements must be strings, got {type(item).__name__}")
    return client_caps & server_caps


# Extracted Responsibility 2: Pre-Dispatch Capability Validation
# (This is the primary responsibility extracted from the monolithic baseline:
# a standalone guard that enforces access control before any dispatch action)
def validate_capability(capability: str, negotiated_capabilities: set[str]) -> None:
    """Validate that the requested capability was agreed upon during negotiation.

    This function acts as a pre-dispatch firewall/guard. If the capability
    is not in negotiated_capabilities, it raises UnsupportedCapabilityError immediately,
    preventing any dispatch handler from running.

    Args:
        capability: The capability name requested by the caller.
        negotiated_capabilities: The set of agreed capabilities from negotiation.

    Raises:
        TypeError: If capability is not a string or negotiated_capabilities is not a set.
        UnsupportedCapabilityError: If capability is not in negotiated_capabilities.
    """
    if not isinstance(capability, str):
        raise TypeError(f"Capability must be a string, got {type(capability).__name__}")
    if not isinstance(negotiated_capabilities, set):
        raise TypeError("negotiated_capabilities must be a set of strings")
    for item in negotiated_capabilities:
        if not isinstance(item, str):
            raise TypeError(f"All negotiated capability elements must be strings, got {type(item).__name__}")

    if capability not in negotiated_capabilities:
        raise UnsupportedCapabilityError(
            f"Capability '{capability}' is not supported. "
            f"Agreed negotiated capabilities: {sorted(list(negotiated_capabilities))}"
        )


# Extracted Responsibility 3: Dispatch Routing
def dispatch_capability(capability: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route a validated capability to its registered execution handler.

    Args:
        capability: The name of the validated capability.
        payload: Optional dictionary payload passed to the handler.

    Returns:
        The dictionary response produced by the capability handler.

    Raises:
        KeyError: If no handler is registered for the given capability.
    """
    _dispatch_call_log.append({"capability": capability, "payload": payload, "version": "refactored"})
    handler = DISPATCH_TABLE.get(capability)
    if handler is None:
        raise KeyError(f"No dispatch handler registered for capability '{capability}'")
    return handler(payload)


def invoke_capability(
    capability: str,
    client_caps: set[str],
    server_caps: set[str],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate capability negotiation, pre-dispatch validation, and dispatch.

    1. Negotiates capabilities via set intersection.
    2. Validates that the requested capability is supported BEFORE dispatch.
    3. If valid, dispatches the request to the appropriate handler.
    """
    # Step 1: Capability negotiation
    negotiated = negotiate_capabilities(client_caps, server_caps)

    # Step 2: Pre-dispatch validation (fails fast before dispatch if unsupported)
    validate_capability(capability, negotiated)

    # Step 3: Dispatch (only reached if validation succeeds)
    return dispatch_capability(capability, payload)


# --- Helper Output and CLI Execution ---

def format_set(s: set[str]) -> str:
    """Format a set of strings into a sorted comma-separated string."""
    return ", ".join(sorted(list(s))) if s else "(none)"


def run_single(capability: str) -> int:
    """Execute a single capability request against the default client/server capabilities."""
    agreed = negotiate_capabilities(CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
    print(f"Client capabilities: {format_set(CLIENT_CAPABILITIES)}")
    print(f"Server capabilities: {format_set(SERVER_CAPABILITIES)}")
    print(f"Negotiated capabilities: {format_set(agreed)}")
    print(f"Requesting capability: {capability}")
    try:
        result = invoke_capability(capability, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
        print("Validation: PASSED (pre-dispatch check)")
        print(f"Dispatch result: {result}")
        return 0
    except UnsupportedCapabilityError as err:
        print("Validation: REJECTED (blocked before dispatch)")
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error: {err}", file=sys.stderr)
        return 2


def run_full_demo() -> int:
    """Run an educational walkthrough demonstrating both allowed and rejected capabilities."""
    agreed = negotiate_capabilities(CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
    print("=" * 65)
    print("Capability Negotiation & Pre-Dispatch Validation Demo")
    print("=" * 65)
    print(f"Client capabilities: {format_set(CLIENT_CAPABILITIES)}")
    print(f"Server capabilities: {format_set(SERVER_CAPABILITIES)}")
    print(f"Negotiated capabilities: {format_set(agreed)}")
    print("-" * 65)

    test_cases = ["tools", "resources", "prompts"]
    for cap in test_cases:
        reset_dispatch_call_log()
        print(f"\n[Invoking capability: '{cap}']")
        try:
            result = invoke_capability(cap, CLIENT_CAPABILITIES, SERVER_CAPABILITIES)
            calls = get_dispatch_call_log()
            print(f"-> Allowed: True")
            print(f"-> Dispatch executed: {len(calls)} call(s)")
            print(f"-> Result: {result['result']}")
        except UnsupportedCapabilityError as err:
            calls = get_dispatch_call_log()
            print(f"-> Allowed: False")
            print(f"-> Dispatch executed: {len(calls)} call(s) (blocked before dispatch)")
            print(f"-> Reason: {err}")

    print("\n" + "=" * 65)
    print("Demo complete: only mutually agreed capabilities reached dispatch.")
    print("=" * 65)
    return 0


def main(argv: list[str]) -> int:
    """CLI entrypoint."""
    if len(argv) == 2:
        return run_single(argv[1])
    elif len(argv) == 1:
        return run_full_demo()
    else:
        print("usage: python3 example.py [capability]", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

