# R38-T028: Finite-State Reward Trace — Failure & Recovery Example

**Lesson:** [Deep Reinforcement Learning](../../README.md) | `lessons/6-Other/22-DeepRL`

---

## Overview

This lab implements a minimal **3-state discrete RL environment** that demonstrates:

- Reward tracing across a finite-horizon episode.
- How an environment must **reject invalid inputs without corrupting its state**, enabling clean recovery.
- The standard `step()` / `reset()` / `trace` interface used in real RL frameworks (e.g., OpenAI Gym).

---

## Environment Specification

| Property | Value |
|---|---|
| States | {0, 1, 2} |
| Initial state | 0 |
| Terminal state | 2 |
| Actions | `'left'`, `'right'` |
| Step cap | 5 |
| Reward | +1 strictly on entering state 2; 0 otherwise |

### Transition Table

| From state | Action | To state |
|---|---|---|
| 0 | `right` | 1 |
| 0 | `left` | 0 (boundary clamp) |
| 1 | `right` | 2 (terminal, reward +1) |
| 1 | `left` | 0 |
| 2 | — | Terminal (no further steps) |

---

## Step-by-Step Arithmetic Derivation (Worked Example)

### Optimal path: `[right, right]`

```
Step 1: state=0, action='right' → state=1, reward=0, done=False
Step 2: state=1, action='right' → state=2, reward=1, done=True
Total reward = 0 + 1 = 1.0
```

### Step-cap path: `[left, left, left, left, left]`

```
Step 1: state=0, action='left' → state=0, reward=0, done=False
Step 2: state=0, action='left' → state=0, reward=0, done=False
Step 3: state=0, action='left' → state=0, reward=0, done=False
Step 4: state=0, action='left' → state=0, reward=0, done=False
Step 5: state=0, action='left' → state=0, reward=0, done=True  ← step cap
Total reward = 0.0
```

### Failure & Recovery: invalid action followed by valid action

```
reset() → state=0
step('right')      → state=1, step_count=1  (valid)
step('jump')       → InvalidActionError raised; state=1, step_count=1 (unchanged!)
step('right')      → state=2, reward=1, done=True  (clean recovery)
Total reward = 1.0
```

The key insight: **invalid inputs raise an exception before modifying any internal state**, so `step_count`, `state`, and `total_reward` are all preserved exactly as they were before the bad call.

---

## Custom Exceptions

| Exception | When raised |
|---|---|
| `InvalidActionError` | Action is not in `{'left', 'right'}` |
| `InvalidStateError` | Constructed or reset with state ∉ {0, 1, 2} |
| `EpisodeTerminatedError` | `step()` called after episode is done |

---

## Files

| File | Purpose |
|---|---|
| `example.py` | `ThreeStateEnv` class, `StepResult`, `StepTraceEntry`, demo `main()` |
| `test_example.py` | Unit tests: optimal path, step-cap, zig-zag, failure & recovery, trace |
| `README.md` | This document |

---

## Usage

```bash
# Run the demonstration
python3 example.py

# Run the test suite
python3 -m unittest discover -s lessons/6-Other/22-DeepRL/applied-labs/r38-t028 -p "test_*.py"
```

### Expected demo output (abbreviated)

```
=== Discrete 3-State RL Environment: Reward Trace & Recovery ===

--- Scenario 1: Optimal Trajectory ['right', 'right'] ---
Step 1: State 0 --(right)--> State 1 | Reward: 0.0 | Done: False
Step 2: State 1 --(right)--> State 2 | Reward: 1.0 | Done: True
Total Reward: 1.0 (Expected: 1.0), Final State: 2

--- Scenario 2: Step Cap (Repeated 'left') ---
Step 1: State 0 --(left)--> State 0 | Reward: 0.0 | Done: False
...
Step 5: State 0 --(left)--> State 0 | Reward: 0.0 | Done: True
Total Reward: 0.0 (Expected: 0.0), Steps taken: 5

--- Scenario 3: Failure & Recovery Handling ---
After valid 'right': state=1, step=1
Dispatching invalid action 'jump'...
Caught expected error: Invalid action 'jump'. Must be one of ['left', 'right']
State preserved after failure: state=1, step=1
Recovered with valid 'right': state=2, reward=1.0, done=True
Final Total Reward: 1.0
```

---

## Key Learning Points

1. **Reward shaping:** Reward signals must be precisely gated (here, only on entering state 2) to correctly guide learning.
2. **Episode horizon:** Step caps prevent infinite episodes without a termination condition.
3. **Robust environment design:** Invalid inputs should fail loudly (exception) without silently corrupting state, so the agent can retry.
4. **Trace logging:** Recording every `(step, from_state, action, to_state, reward)` tuple is essential for debugging RL training.
