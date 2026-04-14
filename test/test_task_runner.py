import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import task_runner


class DummyEnv:
    def __init__(self):
        self.num_lines = 3
        self.num_stations = 4
        self.max_budget = 10
        self.action_space = MagicMock()
        self.action_space.sample.return_value = 2
        self.mediator = MagicMock()
        self.mediator.score = 7
        self.step_calls = 0

    def reset(self, seed=None):
        del seed
        return [0, 0, 0, 0, 1, 2, 3], {"reset": True}

    def step(self, action):
        del action
        self.step_calls += 1
        return [0, 0, 0, 0, 1, 2, 3], 1.5, self.step_calls >= 2, False, {"ok": True}


class TestTaskRunner(unittest.TestCase):
    def test_parse_env_overrides(self):
        parsed = task_runner._parse_env_overrides('{"max_budget": 12}')
        self.assertEqual(parsed["max_budget"], 12)
        self.assertEqual(task_runner._parse_env_overrides(None), {})

    def test_load_policy_builtin(self):
        policy = task_runner._load_policy("random")
        self.assertIs(policy, task_runner.random_policy)

    def test_load_policy_rejects_invalid_custom_string(self):
        with self.assertRaises(ValueError):
            task_runner._load_policy("not_callable_name")

    def test_run_task_uses_selected_spec(self):
        dummy_spec = MagicMock()
        dummy_spec.task_id = "dummy"
        dummy_spec.description = "dummy task"
        dummy_spec.default_steps = 5
        dummy_spec.env_factory.return_value = DummyEnv()

        with patch("task_runner.get_task_spec", return_value=dummy_spec):
            result = task_runner.run_task(
                task_id="dummy",
                policy_name="random",
                steps=4,
                seed=1,
                env_overrides={"max_budget": 8},
            )

        dummy_spec.env_factory.assert_called_once_with({"max_budget": 8})
        self.assertEqual(result["task_id"], "dummy")
        self.assertEqual(result["steps_executed"], 2)
        self.assertTrue(result["done"])
        self.assertEqual(result["final_score"], 7)

    def test_keep_capacity_policy_prefers_add_when_low_lines(self):
        env = DummyEnv()
        env.num_lines = 2
        env.num_stations = 3
        env.max_budget = 10
        obs = [0, 0, 0, 1, 1]
        action = task_runner.keep_capacity_policy(env, obs, {})
        self.assertEqual(action, 1)

    def test_list_tasks_contains_catalog_entries(self):
        tasks = task_runner.list_tasks()
        self.assertGreaterEqual(len(tasks), 1)
        self.assertIn("task_id", tasks[0])


if __name__ == "__main__":
    unittest.main()
