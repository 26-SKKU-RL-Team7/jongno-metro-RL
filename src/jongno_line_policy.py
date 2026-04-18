from __future__ import annotations

from typing import Any

import numpy as np


def get_line_feature_vector(env: Any, obs: np.ndarray) -> np.ndarray:
    """Bias + normalized mean-demand observation (same shape as env.observation_space)."""
    return np.concatenate([np.array([1.0], dtype=np.float32), obs.astype(np.float32)])


def get_line_action_mask(env: Any, _obs: np.ndarray) -> np.ndarray:
    feasible = getattr(env, "_candidate_feasible", None)
    if feasible is None:
        return np.ones((env.action_space.n,), dtype=bool)
    return np.array(feasible, dtype=bool)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask, logits, -1e9)
    max_logit = float(np.max(masked_logits))
    shifted = masked_logits - max_logit
    exp_vals = np.exp(shifted) * mask.astype(np.float32)
    total = float(np.sum(exp_vals))
    if total <= 0:
        probs = np.zeros_like(masked_logits, dtype=np.float32)
        probs[0] = 1.0
        return probs
    return (exp_vals / total).astype(np.float32)


def sample_line_action(
    weights: np.ndarray, env: Any, obs: np.ndarray, rng: np.random.Generator
) -> tuple[int, np.ndarray, np.ndarray]:
    features = get_line_feature_vector(env, obs)
    logits = weights @ features
    mask = get_line_action_mask(env, obs)
    probs = _masked_softmax(logits, mask)
    action = int(rng.choice(len(probs), p=probs))
    return action, probs, features


def greedy_line_action(weights: np.ndarray, env: Any, obs: np.ndarray) -> int:
    features = get_line_feature_vector(env, obs)
    logits = weights @ features
    mask = get_line_action_mask(env, obs)
    masked_logits = np.where(mask, logits, -1e9)
    return int(np.argmax(masked_logits))
