"""Finite-State Reward Trace & Approach Comparison (R38-T043).

TEACHING APPROXIMATION: This is a local educational simulation of a 3-state
finite reinforcement learning environment. It illustrates deterministic state
transitions, terminal states, reward accumulation, and bounded episode horizons
using pure Python standard library. It does not train a neural network, learn
an optimal policy, or claim trained-model or cloud service performance.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

# Environment specification
STATES: set[int] = {0, 1, 2}
INITIAL_STATE: int = 0
TERMINAL_STATE: int = 2
MAX_EPISODE_STEPS: int = 5
ACTIONS: set[str] = {"left", "right"}

# Standard Scenario Action Sequences
SCENARIO_A_ACTIONS: list[str] = ["right", "right"]
SCENARIO_B_ACTIONS: list[str] = ["left", "left", "left", "left", "left"]


@dataclass(frozen=True)
class EpisodeResult:
    """Structured result of an executed reinforcement learning episode."""

    initial_state: int
    final_state: int
    total_reward: int
    steps_executed: int
    terminated: bool
    truncated: bool
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


# =====================================================================
# Approach 1: Direct State-Transition & Reward Simulation
# =====================================================================

def step_direct(current_state: int, action: str) -> tuple[int, int, bool]:
    """Execute a single deterministic transition in the 3-state environment.

    Transition rules:
    - State 0 + right -> State 1, reward = 0
    - State 1 + right -> State 2, reward = 1 (terminal)
    - State 0 + left  -> State 0, reward = 0 (boundary bounce)
    - State 1 + left  -> State 0, reward = 0
    - State 2 is terminal; no further actions should be executed.

    Args:
        current_state: Current state index (0, 1, or 2).
        action: 'left' or 'right'.

    Returns:
        tuple of (next_state, reward, is_terminal).

    Raises:
        TypeError: If state is not int or action is not str.
        ValueError: If state is not in {0, 1, 2} or action is not in {'left', 'right'}.
        RuntimeError: If transition is attempted from the terminal state 2.
    """
    if not isinstance(current_state, int):
        raise TypeError(f"State must be an integer, got {type(current_state).__name__}")
    if not isinstance(action, str):
        raise TypeError(f"Action must be a string, got {type(action).__name__}")

    if current_state not in STATES:
        raise ValueError(f"Invalid state {current_state}. Valid states are {sorted(STATES)}")
    if action not in ACTIONS:
        raise ValueError(f"Invalid action '{action}'. Valid actions are {sorted(ACTIONS)}")

    if current_state == TERMINAL_STATE:
        raise RuntimeError("Cannot step from terminal state 2")

    # Transition logic
    if current_state == 0:
        if action == "right":
            next_state = 1
            reward = 0
            is_terminal = False
        else:  # left boundary transition: stays in state 0
            next_state = 0
            reward = 0
            is_terminal = False
    elif current_state == 1:
        if action == "right":
            next_state = 2
            reward = 1  # Reward granted ONLY upon entering terminal state 2
            is_terminal = True
        else:  # left transition: returns to state 0
            next_state = 0
            reward = 0
            is_terminal = False
    else:
        raise RuntimeError("Unexpected state")

    return next_state, reward, is_terminal


def simulate_direct(
    actions: Sequence[str],
    initial_state: int = INITIAL_STATE,
    max_steps: int = MAX_EPISODE_STEPS,
) -> EpisodeResult:
    """Simulate an episode step-by-step using direct state transitions.

    Args:
        actions: Sequence of action strings ('left' or 'right').
        initial_state: Initial state (default 0).
        max_steps: Maximum allowable steps in an episode (default 5).

    Returns:
        EpisodeResult containing final state, total reward, step counts,
        termination/truncation flags, and complete transition trace.
    """
    if not isinstance(initial_state, int):
        raise TypeError(f"initial_state must be an integer, got {type(initial_state).__name__}")
    if initial_state not in STATES:
        raise ValueError(f"Invalid initial_state {initial_state}. Valid states: {sorted(STATES)}")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")

    current_state = initial_state
    total_reward = 0
    steps_executed = 0
    terminated = (current_state == TERMINAL_STATE)
    truncated = False

    # Trace begins with initial state snapshot (Step 0)
    trace: list[dict[str, Any]] = [
        {"step": 0, "action": None, "state": current_state, "reward": 0}
    ]

    for action in actions:
        # 1. Terminal condition: stop if terminal state is reached
        if terminated:
            break

        # 2. Horizon cap: stop if max_steps is reached
        if steps_executed >= max_steps:
            truncated = True
            break

        # Execute direct step
        next_state, reward, is_terminal = step_direct(current_state, action)
        steps_executed += 1
        total_reward += reward
        current_state = next_state
        terminated = is_terminal

        # Record step in trace
        trace.append({
            "step": steps_executed,
            "action": action,
            "state": current_state,
            "reward": reward,
        })

        if terminated:
            break

        if steps_executed >= max_steps:
            truncated = True
            break

    return EpisodeResult(
        initial_state=initial_state,
        final_state=current_state,
        total_reward=total_reward,
        steps_executed=steps_executed,
        terminated=terminated,
        truncated=truncated,
        trace=trace,
    )


# =====================================================================
# Approach 2: Independent Declarative Calculation
# =====================================================================

# Explicit declarative transition graph lookup table
INDEPENDENT_TRANSITION_MAP: dict[tuple[int, str], tuple[int, int]] = {
    (0, "left"): (0, 0),
    (0, "right"): (1, 0),
    (1, "left"): (0, 0),
    (1, "right"): (2, 1),
}


def calculate_independent(
    actions: Sequence[str],
    initial_state: int = INITIAL_STATE,
    max_steps: int = MAX_EPISODE_STEPS,
) -> dict[str, Any]:
    """Calculate episode outcomes independently using an explicit transition map.

    This function does NOT call simulate_direct or reuse its internal tracking.
    Instead, it processes the action sequence declaratively and independently
    computes the final state, cumulative reward, step count, and flags.

    Args:
        actions: Sequence of action strings.
        initial_state: Starting state index.
        max_steps: Step budget ceiling.

    Returns:
        Dictionary summarizing final state, total reward, steps, terminated, truncated.
    """
    if not isinstance(initial_state, int):
        raise TypeError(f"initial_state must be an integer, got {type(initial_state).__name__}")
    if initial_state not in STATES:
        raise ValueError(f"Invalid initial_state {initial_state}")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")

    curr = initial_state
    cum_reward = 0
    step_count = 0
    is_term = (curr == TERMINAL_STATE)
    is_trunc = False

    for act in actions:
        if is_term:
            break
        if step_count >= max_steps:
            is_trunc = True
            break

        if not isinstance(act, str) or act not in ACTIONS:
            raise ValueError(f"Invalid action '{act}' in independent calculation")

        # Lookup in declarative table
        next_s, r = INDEPENDENT_TRANSITION_MAP[(curr, act)]
        step_count += 1
        cum_reward += r
        curr = next_s

        if curr == TERMINAL_STATE:
            is_term = True
            break

        if step_count >= max_steps:
            is_trunc = True
            break

    return {
        "initial_state": initial_state,
        "final_state": curr,
        "total_reward": cum_reward,
        "steps_executed": step_count,
        "terminated": is_term,
        "truncated": is_trunc,
    }


# =====================================================================
# Approach Comparison & Validation
# =====================================================================

def compare_approaches(
    actions: Sequence[str],
    initial_state: int = INITIAL_STATE,
    max_steps: int = MAX_EPISODE_STEPS,
) -> dict[str, Any]:
    """Execute both approaches on identical inputs and compare their outcomes."""
    res_direct = simulate_direct(actions, initial_state=initial_state, max_steps=max_steps)
    res_indep = calculate_independent(actions, initial_state=initial_state, max_steps=max_steps)

    comparison_fields = [
        "final_state",
        "total_reward",
        "steps_executed",
        "terminated",
        "truncated",
    ]

    direct_dict = res_direct.to_dict()
    diffs = []
    for field in comparison_fields:
        if direct_dict[field] != res_indep[field]:
            diffs.append(f"{field}: direct={direct_dict[field]} != independent={res_indep[field]}")

    return {
        "actions": list(actions),
        "direct": {k: direct_dict[k] for k in comparison_fields},
        "independent": res_indep,
        "match": len(diffs) == 0,
        "diffs": diffs,
    }


# =====================================================================
# CLI Runner and Demonstrations
# =====================================================================

def render_scenario_summary(name: str, actions: list[str]) -> None:
    """Print a clean summary of a scenario using both approaches."""
    print("=" * 68)
    print(f"Scenario: {name}")
    print(f"Input Actions: {actions}")
    print("-" * 68)

    direct_res = simulate_direct(actions)
    print("Approach 1 (Direct Simulation Trace):")
    for row in direct_res.trace:
        act_str = f"Action = {row['action']:<5}" if row['action'] else "Initial State "
        print(f"  Step {row['step']}: {act_str} | State = {row['state']} | Reward = {row['reward']}")

    print("\nApproach 1 Summary:")
    print(f"  Final State:      {direct_res.final_state}")
    print(f"  Total Reward:     {direct_res.total_reward}")
    print(f"  Steps Executed:   {direct_res.steps_executed}")
    print(f"  Terminated:       {direct_res.terminated}")
    print(f"  Truncated:        {direct_res.truncated}")

    indep_res = calculate_independent(actions)
    print("\nApproach 2 Summary (Independent Calculation):")
    print(f"  Final State:      {indep_res['final_state']}")
    print(f"  Total Reward:     {indep_res['total_reward']}")
    print(f"  Steps Executed:   {indep_res['steps_executed']}")
    print(f"  Terminated:       {indep_res['terminated']}")
    print(f"  Truncated:        {indep_res['truncated']}")

    comp = compare_approaches(actions)
    print("\nComparison Result:")
    print(f"  Approaches Agree: {comp['match']}")
    if not comp['match']:
        print(f"  Differences:      {comp['diffs']}")
    print("=" * 68 + "\n")


def main(argv: list[str]) -> int:
    """CLI entrypoint."""
    if len(argv) == 1:
        render_scenario_summary("Scenario A: Successful Path", SCENARIO_A_ACTIONS)
        render_scenario_summary("Scenario B: Repeated Left", SCENARIO_B_ACTIONS)
        return 0
    elif len(argv) == 2:
        arg = argv[1].lower()
        if arg in ("scenario_a", "a"):
            render_scenario_summary("Scenario A: Successful Path", SCENARIO_A_ACTIONS)
            return 0
        elif arg in ("scenario_b", "b"):
            render_scenario_summary("Scenario B: Repeated Left", SCENARIO_B_ACTIONS)
            return 0
        else:
            print(f"Unknown scenario '{argv[1]}'. Use 'scenario_a' or 'scenario_b'.", file=sys.stderr)
            return 2
    else:
        print("usage: python3 example.py [scenario_a|scenario_b]", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

