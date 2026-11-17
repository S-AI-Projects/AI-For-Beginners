# Applied Lab R38-T043: Finite-State Reward Trace & Approach Comparison

## 1. Task Objective

The objective of this lab is to model and simulate a small, deterministic, 3-state Reinforcement Learning (RL) environment to understand fundamental Markov Decision Process (MDP) concepts:
- **States & Actions:** Discrete states $\{0, 1, 2\}$ and discrete action space $\{\text{"left"}, \text{"right"}\}$.
- **State Transitions:** Deterministic state evolution according to a transition graph.
- **Terminal States:** Reaching state $2$ terminates the episode immediately; no further actions execute.
- **Rewards:** A reward of $+1$ is granted strictly upon transition into terminal state $2$; all other transitions yield $0$.
- **Bounded Horizons (Truncation):** A finite step budget of $5$ steps caps the episode if the terminal state is not reached.
- **Comparison of Two Approaches:** Validating the environment by comparing a step-by-step direct simulator (Approach 1) against an independent declarative calculation (Approach 2) on identical inputs.

> **Simulation Disclaimer:** This is a deterministic local teaching simulation using Python standard library. It does not train a neural network, run an agent policy gradient, or interface with external environments like OpenAI Gym/Gymnasium.

---

## 2. Core Reinforcement Learning Concepts

In Reinforcement Learning (RL):
1. **Agent & Environment:** An agent interacts with an environment at discrete time steps $t = 0, 1, 2, \dots$
2. **State ($S_t \in \mathcal{S}$):** The situation of the agent in the environment. Here, $\mathcal{S} = \{0, 1, 2\}$.
3. **Action ($A_t \in \mathcal{A}$):** The choice made by the agent. Here, $\mathcal{A} = \{\text{"left"}, \text{"right"}\}$.
4. **Transition Function ($P(s' | s, a)$):** The rule determining the next state $s'$ given current state $s$ and action $a$. In this deterministic environment, $s' = \delta(s, a)$.
5. **Reward Function ($R(s, a, s')$):** Numerical feedback received after a transition. Here:
   $$R(s, a, s') = \begin{cases} 1 & \text{if } s' = 2 \\ 0 & \text{otherwise} \end{cases}$$
6. **Terminal State:** A state which, once entered, cannot be transitioned out of. Entering state $2$ ends the episode.
7. **Episode Horizon & Truncation:** To prevent infinite loops (e.g. repeated left actions), the environment enforces a maximum step limit ($5$). Reaching the step limit triggers truncation (`truncated = True`).

---

## 3. Transition Rules & Transition Table

The environment consists of 3 states: $0, 1, 2$, starting at initial state $0$.

| Current State | Action | Next State | Reward | Terminal? | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `right` | **1** | 0 | No | Moves toward goal |
| **0** | `left` | **0** | 0 | No | Boundary condition (bounces against wall) |
| **1** | `right` | **2** | 1 | **Yes** | Reaches terminal goal state; reward granted |
| **1** | `left` | **0** | 0 | No | Returns to initial state |
| **2** | *any* | — | — | **Yes** | Terminal; no actions executed once state 2 is reached |

*Note on Rewards:* The reward is granted **only on entry** into state $2$. Remaining in a state never generates a reward.

---

## 4. Required Scenarios

### Scenario A: Successful Path
- **Initial State:** $0$
- **Actions:** `["right", "right"]`
- **Trajectory:**
  - Step 0: State = $0$, Reward = $0$
  - Step 1: Action = `right` $\rightarrow$ State = $1$, Reward = $0$
  - Step 2: Action = `right` $\rightarrow$ State = $2$, Reward = $1$
- **Outcome:**
  - Final State: $2$
  - Total Reward: $1$
  - Steps Executed: $2$
  - Terminated: `True`
  - Truncated: `False`

### Scenario B: Repeated Left
- **Initial State:** $0$
- **Actions:** `["left", "left", "left", "left", "left"]`
- **Trajectory:**
  - Steps 1 through 5: Action = `left` $\rightarrow$ State = $0$, Reward = $0$
- **Outcome:**
  - Final State: $0$
  - Total Reward: $0$
  - Steps Executed: $5$
  - Terminated: `False`
  - Truncated: `True` (hit 5-step cap)

---

## 5. Comparison of Two Approaches

This assignment compares two independent implementations:

1. **Approach 1: Direct Step Simulation (`simulate_direct`)**
   - Implements dynamic step transitions via `step_direct(s, a)`.
   - Records an explicit step-by-step trace including intermediate state transitions, individual step rewards, and cumulative reward tracking.
2. **Approach 2: Independent Declarative Calculation (`calculate_independent`)**
   - Uses an explicit transition graph lookup table:
     ```python
     INDEPENDENT_TRANSITION_MAP = {
         (0, "left"): (0, 0),
         (0, "right"): (1, 0),
         (1, "left"): (0, 0),
         (1, "right"): (2, 1),
     }
     ```
   - Does not call `simulate_direct` or rely on its mutable step objects.
   - Evaluates the path directly to compute final state, cumulative reward, step count, and termination/truncation flags.

### Comparison Results
Both approaches are evaluated on identical inputs (`compare_approaches`). Across Scenario A, Scenario B, empty actions, boundary bounces, and random action trajectories, both implementations produce identical results on all five outcome metrics:
- `final_state`
- `total_reward`
- `steps_executed`
- `terminated`
- `truncated`

---

## 6. Prerequisites

- **Python:** Python 3.8+ (uses Python standard library only: `sys`, `dataclasses`, `typing`, `unittest`).
- **Dependencies:** None. No PyTorch, TensorFlow, Gymnasium, or external simulators needed.

---

## 7. Execution Commands & Expected Terminal Output

### Run Full Demonstration (Both Scenarios & Comparison)

```bash
python3 lessons/6-Other/22-DeepRL/applied-labs/r38-t043/example.py
```

**Expected Output:**
```
====================================================================
Scenario: Scenario A: Successful Path
Input Actions: ['right', 'right']
--------------------------------------------------------------------
Approach 1 (Direct Simulation Trace):
  Step 0: Initial State  | State = 0 | Reward = 0
  Step 1: Action = right | State = 1 | Reward = 0
  Step 2: Action = right | State = 2 | Reward = 1

Approach 1 Summary:
  Final State:      2
  Total Reward:     1
  Steps Executed:   2
  Terminated:       True
  Truncated:        False

Approach 2 Summary (Independent Calculation):
  Final State:      2
  Total Reward:     1
  Steps Executed:   2
  Terminated:       True
  Truncated:        False

Comparison Result:
  Approaches Agree: True
====================================================================

====================================================================
Scenario: Scenario B: Repeated Left
Input Actions: ['left', 'left', 'left', 'left', 'left']
--------------------------------------------------------------------
Approach 1 (Direct Simulation Trace):
  Step 0: Initial State  | State = 0 | Reward = 0
  Step 1: Action = left  | State = 0 | Reward = 0
  Step 2: Action = left  | State = 0 | Reward = 0
  Step 3: Action = left  | State = 0 | Reward = 0
  Step 4: Action = left  | State = 0 | Reward = 0
  Step 5: Action = left  | State = 0 | Reward = 0

Approach 1 Summary:
  Final State:      0
  Total Reward:     0
  Steps Executed:   5
  Terminated:       False
  Truncated:        True

Approach 2 Summary (Independent Calculation):
  Final State:      0
  Total Reward:     0
  Steps Executed:   5
  Terminated:       False
  Truncated:        True

Comparison Result:
  Approaches Agree: True
====================================================================
```

### Run a Specific Scenario

```bash
python3 lessons/6-Other/22-DeepRL/applied-labs/r38-t043/example.py scenario_a
python3 lessons/6-Other/22-DeepRL/applied-labs/r38-t043/example.py scenario_b
```

---

## 8. Automated Test Execution

Run the tests using standard `unittest`:

```bash
python3 -m unittest discover -s lessons/6-Other/22-DeepRL/applied-labs/r38-t043 -p "test_*.py"
```

Verbose mode:
```bash
python3 -m unittest discover -s lessons/6-Other/22-DeepRL/applied-labs/r38-t043 -p "test_*.py" -v
```

The test suite verifies:
- Scenario A transitions, step counts, total reward, and termination.
- Scenario B repeated left actions, 5-step cap, and truncation.
- Reward timing (granted strictly on entry to state 2).
- Boundary bounces and terminal halting.
- Invalid input validation (`ValueError` / `TypeError`).
- Cumulative reward invariant ($\sum \text{step rewards} = \text{total\_reward}$).
- 100% agreement between Approach 1 and Approach 2.

---

## 9. Limitations & Simplifications

1. **Deterministic vs. Stochastic:** Real RL environments often have stochastic state transitions $P(s' | s, a) < 1.0$. This simulation is strictly deterministic.
2. **Fixed Action Policy:** Actions are passed as fixed sequences to observe the resulting trajectory, rather than chosen dynamically by an active policy $\pi(a | s)$ (e.g., Q-learning or Policy Gradients).
3. **Discrete 3-State Chain:** The state space is discrete $\{0, 1, 2\}$, unlike continuous state spaces (e.g. CartPole angles and velocities).

