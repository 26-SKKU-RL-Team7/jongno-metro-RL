from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import station_capacity


@dataclass(frozen=True)
class PolicyParams:
    weights: np.ndarray


def get_feature_vector(env: Any, obs: np.ndarray) -> np.ndarray:
    num_stations = int(env.num_stations)
    num_lines = int(env.num_lines)
    max_budget = max(1.0, float(env.max_budget))
    max_station_capacity = max(1.0, float(station_capacity))
    num_hours = max(1.0, float(getattr(env, "num_hours", 1)))

    station_counts = obs[:num_stations] / max_station_capacity
    line_counts = obs[num_stations : num_stations + num_lines] / max_budget
    hour_idx = float(obs[-1]) / num_hours
    total_active_metros = float(np.sum(obs[num_stations : num_stations + num_lines])) / max_budget

    # Bias + aggregated + raw slots keep this compact but expressive.
    return np.concatenate(
        [
            np.array([1.0, hour_idx, total_active_metros], dtype=np.float32),
            station_counts.astype(np.float32),
            line_counts.astype(np.float32),
        ]
    )


def get_action_mask(env: Any, obs: np.ndarray) -> np.ndarray:
    num_stations = int(env.num_stations)
    num_lines = int(env.num_lines)
    max_budget = int(env.max_budget)

    mask = np.zeros((1 + 2 * num_lines,), dtype=bool)
    mask[0] = True  # noop always valid

    line_counts = np.asarray(obs[num_stations : num_stations + num_lines], dtype=np.float32)
    total_active = int(np.sum(line_counts))

    if total_active < max_budget:
        mask[1 : 1 + num_lines] = True  # add train actions

    # remove only when a line has more than one active train.
    for idx in range(num_lines):
        if line_counts[idx] > 1.0:
            mask[1 + num_lines + idx] = True

    return mask


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


def sample_action(
    weights: np.ndarray, env: Any, obs: np.ndarray, rng: np.random.Generator
) -> tuple[int, np.ndarray, np.ndarray]:
    features = get_feature_vector(env, obs)
    logits = weights @ features
    mask = get_action_mask(env, obs)
    probs = _masked_softmax(logits, mask)
    action = int(rng.choice(len(probs), p=probs))
    return action, probs, features


def greedy_action(weights: np.ndarray, env: Any, obs: np.ndarray) -> int:
    features = get_feature_vector(env, obs)
    logits = weights @ features
    mask = get_action_mask(env, obs)
    masked_logits = np.where(mask, logits, -1e9)
    return int(np.argmax(masked_logits))
