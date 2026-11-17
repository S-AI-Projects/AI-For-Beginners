# R38-T030: Capability Negotiation — Gated Dispatch & Evaluation Experiment

**Lesson:** [Convolutional Neural Networks](../../README.md) | `lessons/4-ComputerVision/07-ConvNets`

---

## Overview

This lab implements a **capability negotiation protocol** with gated dispatch, then runs a controlled evaluation experiment across three configurations.

**Lesson connection:** Just as CNN layers can only process feature maps with the right shape (e.g., a pooling layer cannot directly receive a raw image with unsupported dimensions), real distributed systems require a *handshake* where client and server agree on a shared protocol subset before exchanging data. This lab models that handshake as a set intersection and gates all subsequent calls against it.

---

## Protocol Specification

| Step | Description |
|------|-------------|
| 1. Advertise | Client declares its capability set; server declares its capability set |
| 2. Negotiate | Negotiated set = **client ∩ server** (intersection) |
| 3. Gate | Any dispatch for a capability **outside** the negotiated set is rejected with `CapabilityNotNegotiatedError` *before* any handler is called |
| 4. Dispatch | Requests for negotiated capabilities are forwarded to registered handlers |

### Canonical Example (spec)

```
Client capabilities : {tools, resources}
Server capabilities : {tools, prompts}
─────────────────────────────────────────
Negotiated          : {tools}            ← intersection

dispatch("tools",     payload)  → ✅  handler called, result returned
dispatch("resources", payload)  → ❌  CapabilityNotNegotiatedError
dispatch("prompts",   payload)  → ❌  CapabilityNotNegotiatedError
```

**Negotiation ratio** = |negotiated| / |client| = 1 / 2 = **0.50**

---

## Measurement Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| `negotiation_ratio` | \|negotiated\| / \|client\| | Fraction of client capabilities accepted by server |
| `latency_seconds` | wall-clock time (min over N trials) | Cost of a full negotiate + dispatch cycle |

---

## Evaluation Experiment — 3 Controlled Configurations

| Config | Client Caps | Server Caps | Negotiated | Ratio | Dispatch OK? |
|--------|-------------|-------------|------------|-------|--------------|
| **A — Disjoint** | {resources, prompts} | {tools, sampling} | {} | 0.00 | ❌ |
| **B — Partial** | {tools, resources} | {tools, prompts} | {tools} | 0.50 | ✅ |
| **C — Identical** | {tools, resources} | {tools, resources} | {tools, resources} | 1.00 | ✅ |

**Key findings:**

- **Disjoint sets** → no shared capabilities → gate blocks all dispatches. Any call attempt raises `CapabilityNotNegotiatedError` without touching any handler.
- **Partial overlap** → subset negotiated; latency is bounded by a single handler call.
- **Identical sets** → all client capabilities accepted; negotiation ratio = 1.0 (maximum).

> [!NOTE]
> The gate checks the negotiated set **before** invoking the handler. This means invalid dispatch calls add no latency beyond the set-membership lookup O(1) — an important property for high-frequency systems.

---

## Custom Exceptions

| Exception | When raised |
|-----------|-------------|
| `CapabilityNotNegotiatedError` | Dispatch or handler registration for a capability not in the negotiated set |
| `DuplicateCapabilityHandlerError` | A handler is registered for the same capability twice |
| `EmptyCapabilitySetError` | Either client or server advertises an empty capability set |

---

## Files

| File | Purpose |
|------|---------|
| `example.py` | `NegotiationResult`, `CapabilityGate`, `run_evaluation_experiment()`, demo `main()` |
| `test_example.py` | 30 unit tests across negotiation, dispatch, rejection, handler registration, experiment |
| `README.md` | This document |

---

## Usage

```bash
# Run the demonstration
python3 lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t030/example.py

# Run the test suite
python3 -m unittest discover \
  -s lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t030 \
  -p "test_*.py" -v
```

### Expected demo output (abbreviated)

```
=================================================================
Capability Negotiation — Gated Dispatch & Evaluation Experiment
=================================================================

--- Canonical Example ---
Client: {tools, resources}  |  Server: {tools, prompts}
Negotiated set : {'tools'}
Negotiation ratio: 0.50

Dispatch 'tools'     → SUCCESS: tool_result:image_classifier  (1.23 µs)
Dispatch 'resources' → REJECTED (as expected):
  Capability 'resources' was not negotiated. ...

--- Evaluation Experiment (3 Configurations) ---
Config         Negotiated           Ratio   Latency (µs)       Dispatch    OK?
A_disjoint     set()                 0.00            N/A          <none>     ❌
B_partial      {'tools'}             0.50          0.312           tools     ✅
C_identical    {'resources','tools'} 1.00          0.287        resources   ✅
```

---

## Key Learning Points

1. **Capability negotiation as set intersection:** The simplest correct protocol is the mathematical intersection — it guarantees both sides can handle every agreed-upon capability.
2. **Gate before dispatch:** Rejecting calls at the gate layer (before handler invocation) keeps handlers simple and prevents partial execution on unsupported capabilities.
3. **Negotiation ratio as a compatibility metric:** A ratio near 0 signals major version mismatch; near 1 signals full compatibility.
4. **Separation of mechanism from policy:** `CapabilityGate` enforces *who can communicate*; registered handlers enforce *what happens* — these are independent concerns.
5. **CNN analogy:** A convolutional layer negotiates its output shape with the next layer's expected input shape. If they don't match, the forward pass is "rejected" (a shape error) before any computation occurs.
