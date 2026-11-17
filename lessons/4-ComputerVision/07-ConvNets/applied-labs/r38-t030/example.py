"""
Capability Negotiation — Gated Dispatch with Evaluation Experiment.

Lesson Concept Connection:
In lessons/4-ComputerVision/07-ConvNets, convolutional neural networks process
inputs through layers that each have fixed capabilities (e.g., a pooling layer
cannot perform convolution). Real distributed systems face the same challenge:
a client and server must agree on a shared protocol subset before exchanging
data, just as a CNN must route feature maps only through compatible layer types.

This module implements a lightweight capability negotiation protocol:
  1. Client advertises its capability set.
  2. Server advertises its capability set.
  3. The negotiated set is the INTERSECTION of both sets.
  4. Any dispatch request is gated: requests for capabilities outside the
     negotiated set are rejected with CapabilityNotNegotiatedError before
     reaching the handler.

Specification (canonical example):
  Client capabilities : {tools, resources}
  Server capabilities : {tools, prompts}
  Negotiated          : {tools}            (intersection)
  Dispatch 'tools'    : allowed  → handler called, result returned
  Dispatch 'resources': rejected → CapabilityNotNegotiatedError raised

Evaluation Experiment:
  Three controlled configurations are compared, measuring:
    - negotiation_ratio  = |negotiated| / |client|
    - execution_latency  = wall-clock time for a full negotiate+dispatch cycle (seconds)
  Configurations:
    A. Disjoint sets    : client={resources,prompts}  server={tools,sampling}
    B. Partial overlap  : client={tools,resources}    server={tools,prompts}
    C. Identical sets   : client={tools,resources}    server={tools,resources}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class NegotiationError(Exception):
    """Base class for capability-negotiation errors."""


class CapabilityNotNegotiatedError(NegotiationError):
    """
    Raised when a capability is requested that was not included in the
    negotiated intersection set.
    """


class DuplicateCapabilityHandlerError(NegotiationError):
    """Raised when a handler is registered for a capability already registered."""


class EmptyCapabilitySetError(NegotiationError):
    """Raised when a capability set is provided as an empty collection."""


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegotiationResult:
    """
    Immutable outcome of a capability negotiation handshake.

    Attributes:
        client_caps:    Frozenset of capabilities the client declared.
        server_caps:    Frozenset of capabilities the server declared.
        negotiated:     Intersection — the agreed, dispatchable capabilities.
        negotiation_ratio: |negotiated| / |client_caps|  (0.0 when client is empty)
    """
    client_caps: FrozenSet[str]
    server_caps: FrozenSet[str]
    negotiated: FrozenSet[str]
    negotiation_ratio: float

    @classmethod
    def build(
        cls,
        client_caps: Set[str],
        server_caps: Set[str],
    ) -> "NegotiationResult":
        """
        Factory: validates inputs, computes intersection and ratio.

        Args:
            client_caps: Capability names the client supports.
            server_caps: Capability names the server supports.

        Returns:
            NegotiationResult with negotiated intersection.

        Raises:
            EmptyCapabilitySetError: If either set is empty.
            TypeError: If capability names are not strings.
        """
        for name, caps in [("client", client_caps), ("server", server_caps)]:
            if not caps:
                raise EmptyCapabilitySetError(
                    f"The {name} capability set must not be empty."
                )
            for cap in caps:
                if not isinstance(cap, str):
                    raise TypeError(
                        f"All capability names must be strings; got {type(cap)!r} in {name} set."
                    )

        fc = frozenset(client_caps)
        fs = frozenset(server_caps)
        negotiated = fc & fs
        ratio = len(negotiated) / len(fc) if fc else 0.0
        return cls(
            client_caps=fc,
            server_caps=fs,
            negotiated=negotiated,
            negotiation_ratio=ratio,
        )


@dataclass
class DispatchResult:
    """Result of a successful capability dispatch."""
    capability: str
    handler_return: Any
    latency_seconds: float


# ---------------------------------------------------------------------------
# CapabilityGate — negotiation + gated dispatch
# ---------------------------------------------------------------------------

class CapabilityGate:
    """
    Negotiates a shared capability set and gates all dispatch calls.

    Usage:
        gate = CapabilityGate(client_caps={"tools", "resources"},
                              server_caps={"tools", "prompts"})
        gate.register_handler("tools", lambda payload: f"tool:{payload}")
        result = gate.dispatch("tools", "my_tool")
        # gate.dispatch("resources", ...)  → raises CapabilityNotNegotiatedError
    """

    def __init__(
        self,
        client_caps: Set[str],
        server_caps: Set[str],
    ) -> None:
        self._negotiation = NegotiationResult.build(client_caps, server_caps)
        self._handlers: Dict[str, Callable[[Any], Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def negotiation(self) -> NegotiationResult:
        return self._negotiation

    @property
    def negotiated_caps(self) -> FrozenSet[str]:
        return self._negotiation.negotiated

    @property
    def negotiation_ratio(self) -> float:
        return self._negotiation.negotiation_ratio

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(
        self,
        capability: str,
        handler: Callable[[Any], Any],
    ) -> None:
        """
        Register a handler function for a negotiated capability.

        Args:
            capability: Capability name (must be in negotiated set).
            handler:    Callable accepting one payload argument.

        Raises:
            CapabilityNotNegotiatedError: If capability is not negotiated.
            DuplicateCapabilityHandlerError: If a handler already exists.
        """
        if capability not in self._negotiation.negotiated:
            raise CapabilityNotNegotiatedError(
                f"Cannot register handler for '{capability}': not in negotiated set "
                f"{set(self._negotiation.negotiated)}."
            )
        if capability in self._handlers:
            raise DuplicateCapabilityHandlerError(
                f"Handler already registered for '{capability}'."
            )
        self._handlers[capability] = handler

    # ------------------------------------------------------------------
    # Gated dispatch
    # ------------------------------------------------------------------

    def dispatch(self, capability: str, payload: Any = None) -> DispatchResult:
        """
        Gate-check and dispatch a capability request.

        The gate rejects the request BEFORE calling any handler if the
        capability was not included in the negotiated intersection.

        Args:
            capability: Capability name to invoke.
            payload:    Argument forwarded to the registered handler.

        Returns:
            DispatchResult with the handler's return value and wall-clock latency.

        Raises:
            CapabilityNotNegotiatedError: Capability not in negotiated set.
            KeyError: Capability is negotiated but no handler was registered.
        """
        # Gate check — rejects before any handler is called
        if capability not in self._negotiation.negotiated:
            raise CapabilityNotNegotiatedError(
                f"Capability '{capability}' was not negotiated. "
                f"Negotiated set: {set(self._negotiation.negotiated)}. "
                f"Client declared: {set(self._negotiation.client_caps)}. "
                f"Server declared: {set(self._negotiation.server_caps)}."
            )

        if capability not in self._handlers:
            raise KeyError(
                f"No handler registered for negotiated capability '{capability}'."
            )

        t0 = time.perf_counter()
        result = self._handlers[capability](payload)
        latency = time.perf_counter() - t0

        return DispatchResult(
            capability=capability,
            handler_return=result,
            latency_seconds=latency,
        )


# ---------------------------------------------------------------------------
# Evaluation Experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """One controlled configuration for the evaluation experiment."""
    name: str
    description: str
    client_caps: Set[str]
    server_caps: Set[str]


@dataclass
class ExperimentResult:
    """Measured outcome for one experiment configuration."""
    config_name: str
    negotiated: FrozenSet[str]
    negotiation_ratio: float
    latency_seconds: float  # full negotiate + dispatch cycle (best of trials)
    dispatch_attempted: str
    dispatch_succeeded: bool


def run_evaluation_experiment(
    configs: Optional[List[ExperimentConfig]] = None,
    trials: int = 100,
) -> List[ExperimentResult]:
    """
    Run three controlled configurations and measure negotiation_ratio and latency.

    Default configurations:
        A. Disjoint   : client={resources, prompts}  server={tools, sampling}
        B. Partial    : client={tools, resources}    server={tools, prompts}
        C. Identical  : client={tools, resources}    server={tools, resources}

    For each config:
      - Negotiate the gate.
      - If the negotiated set is non-empty, register a no-op handler for the
        first negotiated capability and dispatch it, measuring latency over
        `trials` iterations (minimum latency reported).
      - If the negotiated set is empty, record latency=0, dispatch_succeeded=False.

    Args:
        configs: Override the default three configs (optional).
        trials:  Number of dispatch iterations for latency measurement.

    Returns:
        List of ExperimentResult, one per config.
    """
    if configs is None:
        configs = [
            ExperimentConfig(
                name="A_disjoint",
                description="Disjoint sets — no shared capabilities",
                client_caps={"resources", "prompts"},
                server_caps={"tools", "sampling"},
            ),
            ExperimentConfig(
                name="B_partial",
                description="Partial overlap — one shared capability",
                client_caps={"tools", "resources"},
                server_caps={"tools", "prompts"},
            ),
            ExperimentConfig(
                name="C_identical",
                description="Identical sets — full overlap",
                client_caps={"tools", "resources"},
                server_caps={"tools", "resources"},
            ),
        ]

    results: List[ExperimentResult] = []

    for cfg in configs:
        gate = CapabilityGate(cfg.client_caps, cfg.server_caps)
        negotiated = gate.negotiated_caps
        ratio = gate.negotiation_ratio

        if not negotiated:
            results.append(ExperimentResult(
                config_name=cfg.name,
                negotiated=negotiated,
                negotiation_ratio=ratio,
                latency_seconds=0.0,
                dispatch_attempted="<none>",
                dispatch_succeeded=False,
            ))
            continue

        # Pick a stable capability to dispatch (sorted for determinism)
        cap = sorted(negotiated)[0]
        gate.register_handler(cap, lambda payload: f"handled:{payload}")

        # Measure latency over `trials` iterations (report minimum)
        latencies: List[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            gate.dispatch(cap, "eval_payload")
            latencies.append(time.perf_counter() - t0)

        results.append(ExperimentResult(
            config_name=cfg.name,
            negotiated=negotiated,
            negotiation_ratio=ratio,
            latency_seconds=min(latencies),
            dispatch_attempted=cap,
            dispatch_succeeded=True,
        ))

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Capability Negotiation — Gated Dispatch & Evaluation Experiment")
    print("=" * 65)

    # -----------------------------------------------------------
    # Canonical example (from spec)
    # -----------------------------------------------------------
    print("\n--- Canonical Example ---")
    print("Client: {tools, resources}  |  Server: {tools, prompts}")
    gate = CapabilityGate(
        client_caps={"tools", "resources"},
        server_caps={"tools", "prompts"},
    )
    print(f"Negotiated set : {set(gate.negotiated_caps)}")
    print(f"Negotiation ratio: {gate.negotiation_ratio:.2f}")

    gate.register_handler("tools", lambda p: f"tool_result:{p}")

    result = gate.dispatch("tools", "image_classifier")
    print(f"\nDispatch 'tools'     → SUCCESS: {result.handler_return}  "
          f"({result.latency_seconds*1e6:.2f} µs)")

    try:
        gate.dispatch("resources", "some_resource")
    except CapabilityNotNegotiatedError as e:
        print(f"Dispatch 'resources' → REJECTED (as expected):")
        print(f"  {e}")

    # -----------------------------------------------------------
    # Evaluation Experiment
    # -----------------------------------------------------------
    print("\n--- Evaluation Experiment (3 Configurations) ---")
    exp_results = run_evaluation_experiment(trials=200)

    header = f"{'Config':<14} {'Negotiated':<20} {'Ratio':>7}  {'Latency (µs)':>14}  {'Dispatch':>14}  {'OK?':>5}"
    print(header)
    print("-" * len(header))
    for r in exp_results:
        lat = f"{r.latency_seconds * 1e6:.3f}" if r.dispatch_succeeded else "N/A"
        print(
            f"{r.config_name:<14} {str(set(r.negotiated)):<20} "
            f"{r.negotiation_ratio:>7.2f}  {lat:>14}  "
            f"{r.dispatch_attempted:>14}  {'✅' if r.dispatch_succeeded else '❌':>5}"
        )


if __name__ == "__main__":
    main()
