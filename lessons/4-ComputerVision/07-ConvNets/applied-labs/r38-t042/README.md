# Applied Lab R38-T042: Capability Negotiation & Pre-Dispatch Validation

## 1. Task Objective

In distributed systems, microservices, and AI tool platforms (such as the Model Context Protocol / MCP), clients and servers must agree on which features both parties support before exchanging commands. This process is called **capability negotiation**.

The objective of this lab is to:
1. Demonstrate capability discovery and intersection between a client and a server.
2. Enforce pre-dispatch validation guards so that unsupported capabilities are rejected *before* any dispatch logic executes.
3. Contrast a monolithic baseline implementation with a refactored implementation where pre-dispatch validation is extracted into a dedicated responsibility.
4. Verify system invariants using automated `unittest` test suites.

> **Simulation Disclaimer:** This is a local educational simulation and mock using only Python's standard library. It does not implement a network transport, wire protocol, cloud service, or production MCP client/server.

---

## 2. Explanation of Capability Negotiation

When a client initiates communication with a server:
1. **Capability Discovery:** The client advertises the set of features it supports. The server advertises the set of features it supports.
2. **Capability Intersection:** The mutually available feature set is computed as the mathematical set intersection ($\text{Client} \cap \text{Server}$).
3. **Agreed Session Capabilities:** Only features present in this intersection may be invoked during the session.

### Educational Client/Server Example

In this lab:
* **Client supports:** `{"tools", "resources"}`
* **Server supports:** `{"tools", "prompts"}`

### How Intersection Works

Using Python set operations:
```python
negotiated = client_capabilities & server_capabilities
# {'tools', 'resources'} & {'tools', 'prompts'} == {'tools'}
```

The resulting negotiated capability set is:
```python
{"tools"}
```

* **`tools`**: Supported by both client and server $\rightarrow$ **Allowed**.
* **`resources`**: Supported by the client, but unsupported by the server $\rightarrow$ **Rejected**.
* **`prompts`**: Supported by the server, but unsupported by the client $\rightarrow$ **Rejected**.

---

## 3. How Invocation is Validated & Why Validation Must Precede Dispatch

### The Invariant: Validation Before Dispatch

A critical software engineering rule in service design is **fail-fast pre-dispatch validation**:

```
Caller Request (e.g., 'resources')
       │
       ▼
┌──────────────────────────────────────────────┐
│  Step 1: Pre-Dispatch Validation Guard       │
│  Is requested capability in negotiated set?   │
└──────────────────────────────────────────────┘
       │
       ├──── No ───► RAISE UnsupportedCapabilityError
       │             (Halt immediately! Dispatch is NEVER reached)
       ▼ Yes
┌──────────────────────────────────────────────┐
│  Step 2: Dispatch Routing                    │
│  Look up handler in DISPATCH_TABLE & execute │
└──────────────────────────────────────────────┘
```

### Why Validation Must Happen Before Dispatch

1. **Security & Least Privilege:** If a capability is not agreed upon, internal server handlers for that capability must never be touched, preventing unexpected side effects or unauthorized subsystem execution.
2. **Resource Protection:** Dispatching handlers may trigger memory allocations, file access, or GPU compute. Rejecting early prevents wasted computational resources.
3. **Defensive Isolation:** The dispatch table may contain handlers that the client is not certified to invoke. Validation acts as a firewall before the router.

If a capability is rejected, **zero** dispatch handlers are executed, and no entry is appended to the dispatch call log.

---

## 4. Baseline vs. Refactored Implementation

To follow clean software engineering practices, this lab implements a baseline and then refactors it:

### Phase A: Baseline (`invoke_capability_baseline`)
The baseline implementation couples capability negotiation, pre-dispatch validation, and dispatch routing inside a single monolithic function. While functional, the validation logic is intertwined with execution mechanics.

### Phase B: Refactored (`invoke_capability`)
The refactored design extracts **Pre-Dispatch Capability Validation** into its own standalone function:
* `validate_capability(capability: str, negotiated_capabilities: set[str]) -> None`: A dedicated guard function that checks capability membership and raises `UnsupportedCapabilityError` if unagreed.
* `negotiate_capabilities(client_caps: set[str], server_caps: set[str]) -> set[str]`: Pure mathematical set intersection.
* `dispatch_capability(capability: str, payload: dict | None) -> dict`: Dispatches to registered handlers only after validation has succeeded.

**Why Extract This Responsibility?**
- Isolates access-control policy from execution routing.
- Enables unit testing the validation firewall in isolation without triggering dispatch handlers.
- Preserves identical public behavior and contracts, proven through automated equivalence tests (`TestBaselineAndRefactoredEquivalence`).

---

## 5. Prerequisites

- **Python:** Python 3.8+ (uses standard library only: `sys`, `typing`, `unittest`).
- **Dependencies:** None. No third-party packages, GPU drivers, network access, or model weights are required.

---

## 6. Execution Commands & Expected Output

### Run Specific Capability Invocation

Invoke the agreed capability `tools`:
```bash
python3 lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t042/example.py tools
```

**Expected Terminal Output:**
```
Client capabilities: resources, tools
Server capabilities: prompts, tools
Negotiated capabilities: tools
Requesting capability: tools
Validation: PASSED (pre-dispatch check)
Dispatch result: {'status': 'success', 'capability': 'tools', 'result': 'Executed tool with payload: None'}
```

Invoke an un-negotiated capability `resources`:
```bash
python3 lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t042/example.py resources
```

**Expected Terminal Output:**
```
Client capabilities: resources, tools
Server capabilities: prompts, tools
Negotiated capabilities: tools
Requesting capability: resources
Validation: REJECTED (blocked before dispatch)
Error: Capability 'resources' is not supported. Agreed negotiated capabilities: ['tools']
```
*(Process exits with non-zero exit code 1).*

### Run the Full Interactive Demo

```bash
python3 lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t042/example.py
```

**Expected Terminal Output:**
```
=================================================================
Capability Negotiation & Pre-Dispatch Validation Demo
=================================================================
Client capabilities: resources, tools
Server capabilities: prompts, tools
Negotiated capabilities: tools
-----------------------------------------------------------------

[Invoking capability: 'tools']
-> Allowed: True
-> Dispatch executed: 1 call(s)
-> Result: Executed tool with payload: None

[Invoking capability: 'resources']
-> Allowed: False
-> Dispatch executed: 0 call(s) (blocked before dispatch)
-> Reason: Capability 'resources' is not supported. Agreed negotiated capabilities: ['tools']

[Invoking capability: 'prompts']
-> Allowed: False
-> Dispatch executed: 0 call(s) (blocked before dispatch)
-> Reason: Capability 'prompts' is not supported. Agreed negotiated capabilities: ['tools']

=================================================================
Demo complete: only mutually agreed capabilities reached dispatch.
=================================================================
```

---

## 7. Automated Test Execution

Run the complete test suite using Python's standard `unittest` framework:

```bash
python3 -m unittest discover -s lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t042 -p "test_*.py"
```

To run with verbose output:
```bash
python3 -m unittest discover -v -s lessons/4-ComputerVision/07-ConvNets/applied-labs/r38-t042 -p "test_*.py"
```

The test suite covers:
- Set intersection logic (default sets, disjoint sets, identical sets, empty sets).
- Successful invocation of agreed capability `tools`.
- Proper rejection of `resources` and `prompts`.
- Verification that dispatch log remains empty for rejected capabilities.
- Type errors and boundary input handling.
- Equivalence between baseline and refactored implementations.

---

## 8. Limitations & Simplifications

1. **Local In-Memory Simulation:** Capabilities are simulated in memory within a single Python process. A real-world protocol uses JSON-RPC over stdio, HTTP/SSE, or WebSockets.
2. **Static Handlers:** Handlers return simple mock status dictionaries rather than performing long-running computations.
3. **No Dynamic Session State:** Capabilities are evaluated per invocation; production protocols maintain multi-round stateful handshakes.

