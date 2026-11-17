"""Automated tests for Finite-State Reward Trace (R38-T043)."""

from __future__ import annotations

import unittest

from example import (
    INITIAL_STATE,
    MAX_EPISODE_STEPS,
    SCENARIO_A_ACTIONS,
    SCENARIO_B_ACTIONS,
    TERMINAL_STATE,
    calculate_independent,
    compare_approaches,
    simulate_direct,
    step_direct,
)


class TestDirectSimulationScenarios(unittest.TestCase):
    """Test standard scenarios for the direct RL environment simulation."""

    def test_scenario_a_successful_path(self) -> None:
        """Scenario A: [right, right] leads to terminal state 2 with reward 1."""
        res = simulate_direct(SCENARIO_A_ACTIONS)

        # 1. Correct right-right transition and step count
        self.assertEqual(res.steps_executed, 2)
        # 2. Correct final state 2
        self.assertEqual(res.final_state, 2)
        # 3. Correct cumulative reward 1
        self.assertEqual(res.total_reward, 1)
        # 4. Correct terminal detection & truncation status
        self.assertTrue(res.terminated)
        self.assertFalse(res.truncated)

        # Trace checks
        self.assertEqual(len(res.trace), 3)  # Step 0, Step 1, Step 2
        self.assertEqual(res.trace[0], {"step": 0, "action": None, "state": 0, "reward": 0})
        self.assertEqual(res.trace[1], {"step": 1, "action": "right", "state": 1, "reward": 0})
        self.assertEqual(res.trace[2], {"step": 2, "action": "right", "state": 2, "reward": 1})

    def test_scenario_b_repeated_left(self) -> None:
        """Scenario B: 5 repeated left actions stay in state 0 and hit the 5-step cap."""
        res = simulate_direct(SCENARIO_B_ACTIONS)

        # 5. Repeated-left scenario steps
        self.assertEqual(res.steps_executed, 5)
        # 6. Correct five-step limit
        self.assertFalse(res.terminated)
        self.assertTrue(res.truncated)
        # 7. Zero reward for repeated left
        self.assertEqual(res.total_reward, 0)
        # 8. Correct final state 0
        self.assertEqual(res.final_state, 0)

        # Trace checks: all steps in state 0 with 0 reward
        for step_idx in range(1, 6):
            self.assertEqual(
                res.trace[step_idx],
                {"step": step_idx, "action": "left", "state": 0, "reward": 0},
            )

    def test_excess_actions_capped_at_five_steps(self) -> None:
        """Episode must truncate at 5 steps even if more actions are supplied."""
        actions = ["left"] * 10
        res = simulate_direct(actions)
        self.assertEqual(res.steps_executed, 5)
        self.assertTrue(res.truncated)
        self.assertFalse(res.terminated)
        self.assertEqual(len(res.trace), 6)  # Step 0 + 5 steps


class TestTransitionsAndBoundaryConditions(unittest.TestCase):
    """Test specific boundary transitions, terminal handling, and reward timing."""

    def test_empty_action_sequence(self) -> None:
        """An empty action list executes 0 steps, keeping initial state and 0 reward."""
        res = simulate_direct([])
        self.assertEqual(res.steps_executed, 0)
        self.assertEqual(res.final_state, 0)
        self.assertEqual(res.total_reward, 0)
        self.assertFalse(res.terminated)
        self.assertFalse(res.truncated)
        self.assertEqual(len(res.trace), 1)

    def test_no_actions_executed_after_terminal_state(self) -> None:
        """When state 2 is reached, subsequent actions must be ignored."""
        actions = ["right", "right", "left", "left", "right"]
        res = simulate_direct(actions)
        self.assertEqual(res.final_state, 2)
        self.assertEqual(res.steps_executed, 2)
        self.assertTrue(res.terminated)
        self.assertFalse(res.truncated)

    def test_reward_granted_only_on_entry_into_state_2(self) -> None:
        """Verify reward is 0 on intermediate steps and 1 only upon entering state 2."""
        s1, r1, term1 = step_direct(0, "right")
        self.assertEqual(s1, 1)
        self.assertEqual(r1, 0)
        self.assertFalse(term1)

        s2, r2, term2 = step_direct(1, "right")
        self.assertEqual(s2, 2)
        self.assertEqual(r2, 1)
        self.assertTrue(term2)

    def test_boundary_left_from_state_0_bounces(self) -> None:
        """State 0 + left stays in state 0 with 0 reward."""
        next_s, r, term = step_direct(0, "left")
        self.assertEqual(next_s, 0)
        self.assertEqual(r, 0)
        self.assertFalse(term)

    def test_left_from_state_1_moves_to_state_0(self) -> None:
        """State 1 + left moves back to state 0 with 0 reward."""
        next_s, r, term = step_direct(1, "left")
        self.assertEqual(next_s, 0)
        self.assertEqual(r, 0)
        self.assertFalse(term)

    def test_stepping_from_terminal_state_raises_runtime_error(self) -> None:
        """Directly stepping from state 2 is invalid."""
        with self.assertRaises(RuntimeError):
            step_direct(2, "right")
        with self.assertRaises(RuntimeError):
            step_direct(2, "left")


class TestErrorAndInvalidInputHandling(unittest.TestCase):
    """Test validation of bad inputs, unknown actions, and bad initial states."""

    def test_invalid_action_rejected(self) -> None:
        """Unknown actions raise ValueError."""
        for invalid in ["up", "down", "jump", "RIGHT", ""]:
            with self.subTest(action=invalid):
                with self.assertRaises(ValueError):
                    simulate_direct([invalid])
                with self.assertRaises(ValueError):
                    step_direct(0, invalid)
                with self.assertRaises(ValueError):
                    calculate_independent([invalid])

    def test_invalid_action_types_rejected(self) -> None:
        """Non-string actions raise TypeError."""
        for invalid in [123, None, ["left"]]:
            with self.subTest(action=invalid):
                with self.assertRaises(TypeError):
                    step_direct(0, invalid)  # type: ignore[arg-type]

    def test_invalid_initial_state_rejected(self) -> None:
        """Initial states outside {0, 1, 2} raise ValueError."""
        for bad_state in [-1, 3, 99]:
            with self.subTest(state=bad_state):
                with self.assertRaises(ValueError):
                    simulate_direct(["right"], initial_state=bad_state)
                with self.assertRaises(ValueError):
                    calculate_independent(["right"], initial_state=bad_state)

    def test_invalid_initial_state_type_rejected(self) -> None:
        """Non-integer initial state raises TypeError."""
        with self.assertRaises(TypeError):
            simulate_direct(["right"], initial_state="0")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            calculate_independent(["right"], initial_state="0")  # type: ignore[arg-type]


class TestTraceConsistencyAndInvariants(unittest.TestCase):
    """Verify trace integrity, step numbering, and cumulative reward invariant."""

    def test_trace_cumulative_reward_invariant(self) -> None:
        """Sum of rewards in the trace must strictly equal total_reward."""
        test_sequences = [
            [],
            ["left"],
            ["right"],
            ["right", "right"],
            ["right", "left", "right", "right"],
            ["left", "left", "left", "left", "left", "left"],
        ]
        for seq in test_sequences:
            with self.subTest(seq=seq):
                res = simulate_direct(seq)
                reward_sum = sum(step["reward"] for step in res.trace)
                self.assertEqual(reward_sum, res.total_reward)
                self.assertEqual(res.steps_executed, len(res.trace) - 1)
                self.assertEqual(res.trace[-1]["state"], res.final_state)


class TestCompareTwoApproaches(unittest.TestCase):
    """Verify agreement between Approach 1 (Direct) and Approach 2 (Independent)."""

    def test_approaches_agree_on_scenario_a(self) -> None:
        comp = compare_approaches(SCENARIO_A_ACTIONS)
        self.assertTrue(comp["match"], f"Scenario A mismatch: {comp['diffs']}")
        self.assertEqual(comp["direct"]["final_state"], 2)
        self.assertEqual(comp["direct"]["total_reward"], 1)
        self.assertEqual(comp["direct"]["steps_executed"], 2)
        self.assertTrue(comp["direct"]["terminated"])
        self.assertFalse(comp["direct"]["truncated"])

    def test_approaches_agree_on_scenario_b(self) -> None:
        comp = compare_approaches(SCENARIO_B_ACTIONS)
        self.assertTrue(comp["match"], f"Scenario B mismatch: {comp['diffs']}")
        self.assertEqual(comp["direct"]["final_state"], 0)
        self.assertEqual(comp["direct"]["total_reward"], 0)
        self.assertEqual(comp["direct"]["steps_executed"], 5)
        self.assertFalse(comp["direct"]["terminated"])
        self.assertTrue(comp["direct"]["truncated"])

    def test_approaches_agree_on_various_paths(self) -> None:
        """Comprehensive verification across varied action sequences."""
        test_cases = [
            [],
            ["right"],
            ["right", "left"],
            ["right", "left", "right"],
            ["right", "left", "right", "right"],
            ["left", "right", "right"],
            ["right", "right", "right", "right"],
            ["left"] * 8,
        ]
        for actions in test_cases:
            with self.subTest(actions=actions):
                comp = compare_approaches(actions)
                self.assertTrue(comp["match"], f"Failed on {actions}: {comp['diffs']}")


if __name__ == "__main__":
    unittest.main()

