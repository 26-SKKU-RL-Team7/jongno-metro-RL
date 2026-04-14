from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from config import station_capacity


@dataclass
class GNNLiteParams:
    w_self: np.ndarray
    w_neigh: np.ndarray
    w_add: np.ndarray
    b_add: np.ndarray
    w_remove: np.ndarray
    b_remove: np.ndarray
    w_noop: np.ndarray
    b_noop: float


def _build_topology_matrices(env: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_names = list(env.station_name_list)
    station_idx = {name: idx for idx, name in enumerate(station_names)}
    num_stations = len(station_names)
    num_lines = int(env.num_lines)

    adjacency = np.zeros((num_stations, num_stations), dtype=np.float32)
    incidence = np.zeros((num_stations, num_lines), dtype=np.float32)

    for line_idx, (_line_name, station_seq) in enumerate(env.env_config["lines"].items()):
        for st_name in station_seq:
            if st_name in station_idx:
                incidence[station_idx[st_name], line_idx] = 1.0
        for a, b in zip(station_seq[:-1], station_seq[1:]):
            if a in station_idx and b in station_idx:
                ai = station_idx[a]
                bi = station_idx[b]
                adjacency[ai, bi] = 1.0
                adjacency[bi, ai] = 1.0

    adjacency += np.eye(num_stations, dtype=np.float32)
    degree = np.sum(adjacency, axis=1, keepdims=True)
    adjacency_norm = adjacency / np.maximum(degree, 1.0)
    line_station_counts = np.sum(incidence, axis=0, keepdims=True).T
    line_station_counts = np.maximum(line_station_counts, 1.0)
    return adjacency_norm, incidence, line_station_counts


def init_params(
    env: Any, hidden_dim: int, rng: np.random.Generator
) -> tuple[GNNLiteParams, Dict[str, np.ndarray]]:
    input_dim = 3
    line_feat_dim = hidden_dim + 2
    params = GNNLiteParams(
        w_self=rng.normal(0.0, 0.08, size=(input_dim, hidden_dim)).astype(np.float32),
        w_neigh=rng.normal(0.0, 0.08, size=(input_dim, hidden_dim)).astype(np.float32),
        w_add=rng.normal(0.0, 0.08, size=(line_feat_dim, 1)).astype(np.float32),
        b_add=np.zeros((1,), dtype=np.float32),
        w_remove=rng.normal(0.0, 0.08, size=(line_feat_dim, 1)).astype(np.float32),
        b_remove=np.zeros((1,), dtype=np.float32),
        w_noop=rng.normal(0.0, 0.08, size=(2,)).astype(np.float32),
        b_noop=0.0,
    )
    adjacency_norm, incidence, line_station_counts = _build_topology_matrices(env)
    static = {
        "adjacency_norm": adjacency_norm,
        "incidence": incidence,
        "line_station_counts": line_station_counts,
    }
    return params, static


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask, logits, -1e9)
    max_logit = float(np.max(masked_logits))
    shifted = masked_logits - max_logit
    exps = np.exp(shifted) * mask.astype(np.float32)
    total = float(np.sum(exps))
    if total <= 0:
        probs = np.zeros_like(masked_logits, dtype=np.float32)
        probs[0] = 1.0
        return probs
    return (exps / total).astype(np.float32)


def get_action_mask(env: Any, obs: np.ndarray) -> np.ndarray:
    num_stations = int(env.num_stations)
    num_lines = int(env.num_lines)
    max_budget = int(env.max_budget)
    line_counts = np.asarray(obs[num_stations : num_stations + num_lines], dtype=np.float32)
    total_active = int(np.sum(line_counts))

    mask = np.zeros((1 + 2 * num_lines,), dtype=bool)
    mask[0] = True
    if total_active < max_budget:
        mask[1 : 1 + num_lines] = True
    for idx in range(num_lines):
        if line_counts[idx] > 1.0:
            mask[1 + num_lines + idx] = True
    return mask


def forward(
    params: GNNLiteParams, static: Dict[str, np.ndarray], env: Any, obs: np.ndarray
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    num_stations = int(env.num_stations)
    num_lines = int(env.num_lines)
    max_station_capacity = max(1.0, float(station_capacity))
    max_budget = max(1.0, float(env.max_budget))
    num_hours = max(1.0, float(getattr(env, "num_hours", 1)))

    station_counts = obs[:num_stations].astype(np.float32) / max_station_capacity
    line_counts = obs[num_stations : num_stations + num_lines].astype(np.float32) / max_budget
    hour = float(obs[-1]) / num_hours

    adjacency_norm = static["adjacency_norm"]
    incidence = static["incidence"]
    line_station_counts = static["line_station_counts"]

    degree_feature = np.mean(adjacency_norm, axis=1).astype(np.float32)
    x_station = np.stack(
        [
            station_counts,
            np.full((num_stations,), hour, dtype=np.float32),
            degree_feature,
        ],
        axis=1,
    )  # (S,3)

    neigh_input = adjacency_norm @ x_station
    pre_hidden = x_station @ params.w_self + neigh_input @ params.w_neigh
    h_station = np.tanh(pre_hidden)  # (S,H)

    line_emb = (incidence.T @ h_station) / line_station_counts
    line_feat = np.concatenate(
        [
            line_emb,
            line_counts.reshape(-1, 1),
            np.full((num_lines, 1), hour, dtype=np.float32),
        ],
        axis=1,
    )  # (L,H+2)

    add_logits = (line_feat @ params.w_add).reshape(-1) + params.b_add[0]
    remove_logits = (line_feat @ params.w_remove).reshape(-1) + params.b_remove[0]
    noop_features = np.array([hour, float(np.sum(line_counts))], dtype=np.float32)
    noop_logit = float(noop_features @ params.w_noop + params.b_noop)

    logits = np.concatenate(
        [np.array([noop_logit], dtype=np.float32), add_logits, remove_logits]
    )
    cache = {
        "x_station": x_station,
        "neigh_input": neigh_input,
        "pre_hidden": pre_hidden,
        "h_station": h_station,
        "line_emb": line_emb,
        "line_feat": line_feat,
        "line_counts": line_counts,
        "hour": np.array([hour], dtype=np.float32),
        "noop_features": noop_features,
    }
    return logits, cache


def sample_action(
    params: GNNLiteParams,
    static: Dict[str, np.ndarray],
    env: Any,
    obs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    logits, cache = forward(params, static, env, obs)
    mask = get_action_mask(env, obs)
    probs = _masked_softmax(logits, mask)
    action = int(rng.choice(len(probs), p=probs))
    return action, probs, cache


def greedy_action(
    params: GNNLiteParams, static: Dict[str, np.ndarray], env: Any, obs: np.ndarray
) -> int:
    logits, _cache = forward(params, static, env, obs)
    mask = get_action_mask(env, obs)
    masked = np.where(mask, logits, -1e9)
    return int(np.argmax(masked))


def backward(
    params: GNNLiteParams,
    static: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    dlogits: np.ndarray,
) -> GNNLiteParams:
    num_lines = cache["line_feat"].shape[0]
    add_slice = slice(1, 1 + num_lines)
    remove_slice = slice(1 + num_lines, 1 + 2 * num_lines)

    d_noop = float(dlogits[0])
    d_add = dlogits[add_slice].reshape(-1, 1)  # (L,1)
    d_remove = dlogits[remove_slice].reshape(-1, 1)  # (L,1)

    line_feat = cache["line_feat"]
    h_station = cache["h_station"]
    pre_hidden = cache["pre_hidden"]
    x_station = cache["x_station"]
    neigh_input = cache["neigh_input"]
    noop_features = cache["noop_features"]

    incidence = static["incidence"]
    line_station_counts = static["line_station_counts"]
    adjacency_norm = static["adjacency_norm"]

    grad_w_add = line_feat.T @ d_add
    grad_b_add = np.array([float(np.sum(d_add))], dtype=np.float32)
    grad_w_remove = line_feat.T @ d_remove
    grad_b_remove = np.array([float(np.sum(d_remove))], dtype=np.float32)
    grad_w_noop = (d_noop * noop_features).astype(np.float32)
    grad_b_noop = d_noop

    d_line_feat = d_add @ params.w_add.T + d_remove @ params.w_remove.T
    d_line_emb = d_line_feat[:, : h_station.shape[1]]
    d_h_station = incidence @ (d_line_emb / line_station_counts)

    d_pre_hidden = d_h_station * (1.0 - np.tanh(pre_hidden) ** 2)

    grad_w_self = x_station.T @ d_pre_hidden
    grad_w_neigh = neigh_input.T @ d_pre_hidden

    # Note: we intentionally stop gradients to input features for a lightweight setup.
    _ = adjacency_norm

    return GNNLiteParams(
        w_self=grad_w_self.astype(np.float32),
        w_neigh=grad_w_neigh.astype(np.float32),
        w_add=grad_w_add.astype(np.float32),
        b_add=grad_b_add.astype(np.float32),
        w_remove=grad_w_remove.astype(np.float32),
        b_remove=grad_b_remove.astype(np.float32),
        w_noop=grad_w_noop.astype(np.float32),
        b_noop=float(grad_b_noop),
    )


def apply_gradients(params: GNNLiteParams, grads: GNNLiteParams, lr: float) -> GNNLiteParams:
    return GNNLiteParams(
        w_self=params.w_self + lr * grads.w_self,
        w_neigh=params.w_neigh + lr * grads.w_neigh,
        w_add=params.w_add + lr * grads.w_add,
        b_add=params.b_add + lr * grads.b_add,
        w_remove=params.w_remove + lr * grads.w_remove,
        b_remove=params.b_remove + lr * grads.b_remove,
        w_noop=params.w_noop + lr * grads.w_noop,
        b_noop=float(params.b_noop + lr * grads.b_noop),
    )


def zeros_like(params: GNNLiteParams) -> GNNLiteParams:
    return GNNLiteParams(
        w_self=np.zeros_like(params.w_self),
        w_neigh=np.zeros_like(params.w_neigh),
        w_add=np.zeros_like(params.w_add),
        b_add=np.zeros_like(params.b_add),
        w_remove=np.zeros_like(params.w_remove),
        b_remove=np.zeros_like(params.b_remove),
        w_noop=np.zeros_like(params.w_noop),
        b_noop=0.0,
    )


def add_gradients(a: GNNLiteParams, b: GNNLiteParams) -> GNNLiteParams:
    return GNNLiteParams(
        w_self=a.w_self + b.w_self,
        w_neigh=a.w_neigh + b.w_neigh,
        w_add=a.w_add + b.w_add,
        b_add=a.b_add + b.b_add,
        w_remove=a.w_remove + b.w_remove,
        b_remove=a.b_remove + b.b_remove,
        w_noop=a.w_noop + b.w_noop,
        b_noop=float(a.b_noop + b.b_noop),
    )
