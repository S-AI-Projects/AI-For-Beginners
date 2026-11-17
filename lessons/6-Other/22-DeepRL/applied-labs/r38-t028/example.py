"""
Finite-State Reward Trace — 3-State RL Environment with Failure & Recovery.

Lesson Concept Connection:
In Reinforcement Learning (lessons/6-Other/22-DeepRL), an agent interacts with an
environment across discrete time steps through actions, transitioning between states
and receiving scalar rewards. In finite-horizon environments:
- Step caps terminate episodes to prevent infinite loops.
- Terminal states signal goal achievement with designated rewards.
- Robust environments must reject invalid actions/states without corrupting their
  internal state, allowing clean recovery on subsequent valid inputs.

Specifications:
- States: {0, 1, 2}
- Initial State: 0
- Terminal State: 2
- Actions: ['left', 'right']
- State Transitions:
  - From state 0: 'right' -> 1; 'left' -> 0 (clamped at boundary 0)
  - From state 1: 'right' -> 2; 'left' -> 0
  - From state 2: Terminal (no further transitions permitted)
- Rewards:
  - Reward 1 granted strictly upon transitioning into terminal state 2.
  - All other transitions grant reward 0.
- Episode Cap:
  - Capped at 5 steps max. If terminal state 2 is not reached by step 5, episode terminates with done=True.
- Failure & Recovery:
  - Invalid actions (e.g. 'jump', non-string, None) or stepping after episode termination
    raise InvalidActionError / EpisodeTerminatedError without modifying current state or step count.
  - Subsequent valid actions proceed cleanly (recovery).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


class RLEnvironmentError(Exception):
    """Base exception for RL environment errors."""
    pass


class InvalidActionError(RLEnvironmentError):
    """Raised when an unrecognized action is dispatched."""
    pass


class InvalidStateError(RLEnvironmentError):
    """Raised when an environment state is set to an invalid value."""
    pass


class EpisodeTerminatedError(RLEnvironmentError):
    """Raised when step() is called on an already terminated episode."""
    pass


@dataclass(frozen=True)
class StepResult:
    """
    Standard outcome returned by env.step(action).

    Attributes:
        observation: New state after taking the action {0, 1, 2}.
        reward: Scalar reward received (1 if entered state 2, else 0).
        done: Whether the episode has terminated (state 2 reached or step cap reached).
        info: Diagnostic metadata dictionary.
    """
    observation: int
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class StepTraceEntry:
    """
    Audit log entry recording an environmental transition.
    """
    step_number: int
    from_state: int
    action: str
    to_state: int
    reward: float
    done: bool


class ThreeStateEnv:
    """
    Discrete 3-state RL environment with finite-horizon step cap and state tracing.
    """
    VALID_STATES: Set[int] = {0, 1, 2}
    VALID_ACTIONS: Set[str] = {"left", "right"}
    TERMINAL_STATE: int = 2
    MAX_STEPS: int = 5

    def __init__(self, initial_state: int = 0):
        if initial_state not in self.VALID_STATES:
            raise InvalidStateError(f"Initial state {initial_state} not in valid states {self.VALID_STATES}")
        self._initial_state: int = initial_state
        self._state: int = initial_state
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = (initial_state == self.TERMINAL_STATE)
        self._trace: List[StepTraceEntry] = []

    @property
    def state(self) -> int:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def done(self) -> bool:
        return self._done

    @property
    def trace(self) -> List[StepTraceEntry]:
        return list(self._trace)

    def reset(self, start_state: Optional[int] = None) -> int:
        """
        Resets environment to start state (default 0).
        """
        if start_state is not None:
            if start_state not in self.VALID_STATES:
                raise InvalidStateError(f"Start state {start_state} not in {self.VALID_STATES}")
            self._state = start_state
        else:
            self._state = self._initial_state

        self._step_count = 0
        self._total_reward = 0.0
        self._done = (self._state == self.TERMINAL_STATE)
        self._trace.clear()
        return self._state

    def step(self, action: str) -> StepResult:
        """
        Executes one environmental transition.

        Validations:
        - If episode is already done, raises EpisodeTerminatedError (state unchanged).
        - If action is not in {'left', 'right'}, raises InvalidActionError (state unchanged).
        """
        if self._done:
            raise EpisodeTerminatedError("Cannot step in an episode that has already terminated. Call reset().")

        if not isinstance(action, str) or action not in self.VALID_ACTIONS:
            # Failure point: state and step count remain intact
            raise InvalidActionError(f"Invalid action {action!r}. Must be one of {sorted(self.VALID_ACTIONS)}")

        prev_state = self._state
        self._step_count += 1

        # Compute next state
        if prev_state == 0:
            next_state = 1 if action == "right" else 0
        elif prev_state == 1:
            next_state = 2 if action == "right" else 0
        else:
            # Should not happen as state 2 is terminal, but safe fallback
            next_state = 2

        # Reward is 1 strictly when entering state 2
        reward = 1.0 if next_state == self.TERMINAL_STATE else 0.0
        self._total_reward += reward
        self._state = next_state

        # Check termination conditions: goal reached or step cap reached
        if next_state == self.TERMINAL_STATE or self._step_count >= self.MAX_STEPS:
            self._done = True

        reason = "goal_reached" if next_state == self.TERMINAL_STATE else ("step_cap_reached" if self._step_count >= self.MAX_STEPS else "in_progress")
        info = {
            "step_count": self._step_count,
            "max_steps": self.MAX_STEPS,
            "reason": reason,
        }

        entry = StepTraceEntry(
            step_number=self._step_count,
            from_state=prev_state,
            action=action,
            to_state=next_state,
            reward=reward,
            done=self._done,
        )
        self._trace.append(entry)

        return StepResult(observation=next_state, reward=reward, done=self._done, info=info)


def run_episode(env: ThreeStateEnv, actions: List[str]) -> Tuple[float, List[StepTraceEntry]]:
    """
    Executes a sequence of actions on the environment.
    """
    env.reset()
    for act in actions:
        if env.done:
            break
        env.step(act)
    return env.total_reward, env.trace


def main() -> None:
    """Demonstrate happy paths, step-cap timeout, and failure & recovery."""
    print("=== Discrete 3-State RL Environment: Reward Trace & Recovery ===")

    env = ThreeStateEnv()

    # Scenario 1: Optimal Path [right, right] -> reaches 2, total reward 1
    print("\n--- Scenario 1: Optimal Trajectory ['right', 'right'] ---")
    tot_reward, trace = run_episode(env, ["right", "right"])
    for t in trace:
        print(f"Step {t.step_number}: State {t.from_state} --({t.action})--> State {t.to_state} | Reward: {t.reward} | Done: {t.done}")
    print(f"Total Reward: {tot_reward} (Expected: 1.0), Final State: {env.state}")

    # Scenario 2: Step Cap Reached (Repeated lefts) -> total reward 0
    print("\n--- Scenario 2: Step Cap (Repeated 'left') ---")
    tot_reward2, trace2 = run_episode(env, ["left", "left", "left", "left", "left", "left"])
    for t in trace2:
        print(f"Step {t.step_number}: State {t.from_state} --({t.action})--> State {t.to_state} | Reward: {t.reward} | Done: {t.done}")
    print(f"Total Reward: {tot_reward2} (Expected: 0.0), Steps taken: {env.step_count}")

    # Scenario 3: Failure & Recovery
    print("\n--- Scenario 3: Failure & Recovery Handling ---")
    env.reset()
    env.step("right")
    print(f"After valid 'right': state={env.state}, step={env.step_count}")

    try:
        print("Dispatching invalid action 'jump'...")
        env.step("jump")
    except InvalidActionError as err:
        print(f"Caught expected error: {err}")
        print(f"State preserved after failure: state={env.state}, step={env.step_count}")

    # Recovery: valid 'right' takes agent from state 1 to state 2
    res = env.step("right")
    print(f"Recovered with valid 'right': state={res.observation}, reward={res.reward}, done={res.done}")
    print(f"Final Total Reward: {env.total_reward}")


if __name__ == "__main__":
    main()

