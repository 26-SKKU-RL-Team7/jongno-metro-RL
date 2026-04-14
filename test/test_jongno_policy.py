import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from jongno_policy import get_action_mask, get_feature_vector, greedy_action


class DummyEnv:
    def __init__(self):
        self.num_stations = 3
        self.num_lines = 2
        self.max_budget = 4
        self.num_hours = 24


class TestJongnoPolicy(unittest.TestCase):
    def test_feature_vector_shape(self):
        env = DummyEnv()
        obs = np.array([1, 2, 3, 1, 2, 5], dtype=np.float32)
        features = get_feature_vector(env, obs)
        self.assertEqual(features.shape[0], 3 + env.num_stations + env.num_lines)

    def test_action_mask_add_remove_rules(self):
        env = DummyEnv()
        # stations(3) + lines(2) + hour(1)
        obs = np.array([0, 1, 2, 1, 2, 3], dtype=np.float32)
        mask = get_action_mask(env, obs)
        # noop valid
        self.assertTrue(mask[0])
        # add actions valid because total(3) < budget(4)
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        # remove for line0 invalid(count=1), line1 valid(count=2)
        self.assertFalse(mask[3])
        self.assertTrue(mask[4])

    def test_greedy_action_respects_mask(self):
        env = DummyEnv()
        obs = np.array([0, 0, 0, 4, 0, 10], dtype=np.float32)
        # total_active == max_budget, add actions must be masked out.
        feature_dim = 3 + env.num_stations + env.num_lines
        weights = np.zeros((1 + 2 * env.num_lines, feature_dim), dtype=np.float32)
        weights[1, :] = 10.0  # prefer masked add action if mask is ignored
        action = greedy_action(weights, env, obs)
        self.assertNotEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
