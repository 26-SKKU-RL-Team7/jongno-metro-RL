import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from gnn_lite_policy import (
    add_gradients,
    apply_gradients,
    backward,
    forward,
    init_params,
    sample_action,
    zeros_like,
)
from task_catalog import get_task_spec


class TestGNNLitePolicy(unittest.TestCase):
    def setUp(self):
        spec = get_task_spec("jongno_dispatch")
        self.env = spec.env_factory({})
        reset_result = self.env.reset(seed=123)
        self.obs = (
            np.asarray(reset_result[0], dtype=np.float32)
            if isinstance(reset_result, tuple)
            else np.asarray(reset_result, dtype=np.float32)
        )
        self.rng = np.random.default_rng(123)
        self.params, self.static = init_params(self.env, hidden_dim=8, rng=self.rng)

    def test_forward_and_sample_action_shapes(self):
        logits, _cache = forward(self.params, self.static, self.env, self.obs)
        self.assertEqual(logits.shape[0], 1 + 2 * self.env.num_lines)
        action, probs, cache = sample_action(
            self.params, self.static, self.env, self.obs, self.rng
        )
        self.assertGreaterEqual(action, 0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertIn("line_feat", cache)

    def test_backward_and_update(self):
        action, probs, cache = sample_action(
            self.params, self.static, self.env, self.obs, self.rng
        )
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        dlogits = one_hot - probs
        grads = backward(self.params, self.static, cache, dlogits)
        accum = add_gradients(zeros_like(self.params), grads)
        updated = apply_gradients(self.params, accum, lr=0.01)
        self.assertEqual(updated.w_self.shape, self.params.w_self.shape)
        self.assertEqual(updated.w_add.shape, self.params.w_add.shape)


if __name__ == "__main__":
    unittest.main()
