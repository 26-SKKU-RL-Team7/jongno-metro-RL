import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from jongno_line_env import JongnoLineSelectionEnv
from jongno_env import JongnoMetroEnv


class TestJongnoLineSelectionEnv(unittest.TestCase):
    def test_reset_step_one_episode(self) -> None:
        env = JongnoLineSelectionEnv(line_eval_rollout_steps=5)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.num_stations,))
        self.assertIn("candidate_feasible", info)
        n0 = len(env.mediator.paths)
        obs2, reward, terminated, truncated, info2 = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs2.shape, obs.shape)
        if info2.get("action_ok"):
            self.assertEqual(len(env.mediator.paths), n0 + 1)
            self.assertIn("extra_lines", info2)

    def test_jongno_metro_extra_lines_increases_paths(self) -> None:
        env_base = JongnoMetroEnv()
        env_base.reset(seed=1)
        n_base = len(env_base.mediator.paths)

        extra = {"test_extra": ["종각", "광화문"]}
        env2 = JongnoMetroEnv(extra_lines=extra)
        env2.reset(seed=1)
        self.assertEqual(len(env2.mediator.paths), n_base + 1)
        ids = [p.id for p in env2.mediator.paths]
        self.assertIn("test_extra", ids)


if __name__ == "__main__":
    unittest.main()
