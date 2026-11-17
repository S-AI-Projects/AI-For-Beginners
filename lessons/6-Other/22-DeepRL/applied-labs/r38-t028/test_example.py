"""
Unit tests for 3-state RL discrete environment with failure and recovery (Issue #28 / R38-T028).

Tests:
1. Optimal trajectory [right, right] ends at state 2 with total reward 1.
2. Repeated left actions cap at 5 steps with total reward 0 and done=True.
3. Intermediate reward checks: strictly 0 until transitioning into state 2.
4. Failure & Recovery: invalid action raises InvalidActionError, leaves internal state
   and step count intact, and allows recovery on the next valid action.
5. Invalid initial/reset state rejection (InvalidStateError).
6. Post-termination step rejection (EpisodeTerminatedError).
7. Trace logging verification.
"""

import unittest
from example import (
    EpisodeTerminatedError,
    InvalidActionError,
    InvalidStateError,
    RLEnvironmentError,
    StepResult,
    ThreeStateEnv,
    run_episode,
)


class TestThreeStateEnvironment(unittest.TestCase):
    def test_optimal_trajectory(self):
        """[right, right] ends at 2 with total reward 1."""
        env = ThreeStateEnv()
        res1 = env.step("right")
        self.assertEqual(res1.observation, 1)
        self.assertEqual(res1.reward, 0.0)
        self.assertFalse(res1.done)

        res2 = env.step("right")
        self.assertEqual(res2.observation, 2)
        self.assertEqual(res2.reward, 1.0)
        self.assertTrue(res2.done)
        self.assertEqual(env.total_reward, 1.0)
        self.assertEqual(env.step_count, 2)

    def test_repeated_left_ends_at_step_cap(self):
        """Repeated left ends at step cap 5 with total reward 0."""
        env = ThreeStateEnv()
        for i in range(4):
            res = env.step("left")
            self.assertEqual(res.observation, 0)
            self.assertEqual(res.reward, 0.0)
            self.assertFalse(res.done)
            self.assertEqual(env.step_count, i + 1)

        # 5th step hits the cap
        res5 = env.step("left")
        self.assertEqual(res5.observation, 0)
        self.assertEqual(res5.reward, 0.0)
        self.assertTrue(res5.done)
        self.assertEqual(env.step_count, 5)
        self.assertEqual(env.total_reward, 0.0)
        self.assertEqual(res5.info["reason"], "step_cap_reached")

    def test_zigzag_trajectory(self):
        """Test sequence [right, left, right, right] -> reaches 2 on step 4."""
        env = ThreeStateEnv()
        actions = ["right", "left", "right", "right"]
        # step 1: 0 -> 1 (reward 0)
        # step 2: 1 -> 0 (reward 0)
        # step 3: 0 -> 1 (reward 0)
        # step 4: 1 -> 2 (reward 1, done)
        for idx, act in enumerate(actions):
            res = env.step(act)
            if idx == 3:
                self.assertEqual(res.observation, 2)
                self.assertEqual(res.reward, 1.0)
                self.assertTrue(res.done)
            else:
                self.assertEqual(res.reward, 0.0)
                self.assertFalse(res.done)
        self.assertEqual(env.total_reward, 1.0)
        self.assertEqual(env.step_count, 4)

    def test_failure_and_recovery_invalid_action(self):
        """
        Exercise an invalid action, verify expected failure,
        verify state preservation, and verify clean recovery on valid action.
        """
        env = ThreeStateEnv()
        env.step("right")  # state 0 -> 1, step_count = 1
        self.assertEqual(env.state, 1)
        self.assertEqual(env.step_count, 1)

        # Failure: dispatch bad actions
        for bad_act in ["up", "down", "jump", 123, None, ""]:
            with self.assertRaises(InvalidActionError):
                env.step(bad_act)  # type: ignore

            # Invariant: internal state, step count, and total reward MUST remain unchanged
            self.assertEqual(env.state, 1, f"State mutated after bad action {bad_act!r}")
            self.assertEqual(env.step_count, 1, f"Step count mutated after bad action {bad_act!r}")
            self.assertEqual(env.total_reward, 0.0)
            self.assertFalse(env.done)

        # Recovery: valid action 'right' successfully transitions from state 1 -> 2
        recovered_res = env.step("right")
        self.assertEqual(recovered_res.observation, 2)
        self.assertEqual(recovered_res.reward, 1.0)
        self.assertTrue(recovered_res.done)
        self.assertEqual(env.step_count, 2)
        self.assertEqual(env.total_reward, 1.0)

    def test_post_termination_stepping_rejected(self):
        """Stepping after episode termination raises EpisodeTerminatedError."""
        env = ThreeStateEnv()
        env.step("right")
        env.step("right")
        self.assertTrue(env.done)

        # Attempt to step after done
        with self.assertRaises(EpisodeTerminatedError):
            env.step("left")
        self.assertEqual(env.state, 2)
        self.assertEqual(env.step_count, 2)

    def test_invalid_initial_and_reset_state(self):
        """Rejects initial states or reset states outside {0, 1, 2}."""
        with self.assertRaises(InvalidStateError):
            ThreeStateEnv(initial_state=3)

        with self.assertRaises(InvalidStateError):
            ThreeStateEnv(initial_state=-1)

        env = ThreeStateEnv()
        with self.assertRaises(InvalidStateError):
            env.reset(start_state=99)

    def test_state_trace_recording(self):
        """Verify audit trace correctly records all transitions."""
        env = ThreeStateEnv()
        tot_reward, trace = run_episode(env, ["right", "right"])

        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0].step_number, 1)
        self.assertEqual(trace[0].from_state, 0)
        self.assertEqual(trace[0].action, "right")
        self.assertEqual(trace[0].to_state, 1)
        self.assertEqual(trace[0].reward, 0.0)
        self.assertFalse(trace[0].done)

        self.assertEqual(trace[1].step_number, 2)
        self.assertEqual(trace[1].from_state, 1)
        self.assertEqual(trace[1].action, "right")
        self.assertEqual(trace[1].to_state, 2)
        self.assertEqual(trace[1].reward, 1.0)
        self.assertTrue(trace[1].done)


if __name__ == "__main__":
    unittest.main()

