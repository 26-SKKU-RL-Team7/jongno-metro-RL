from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from gnn_lite_policy import GNNLiteParams, greedy_action as gnn_greedy_action, init_params
from jongno_policy import greedy_action as linear_greedy_action
from task_catalog import get_task_spec
from train_jongno_gnn_lite import train as train_gnn
from train_jongno_policy import train as train_linear


def _evaluate_linear(
    task_id: str,
    model_path: Path,
    eval_episodes: int,
    max_steps: int,
    seed: int,
    env_overrides: Dict[str, Any],
) -> Dict[str, float]:
    payload = np.load(model_path, allow_pickle=True)
    weights = np.asarray(payload["weights"], dtype=np.float32)
    task_spec = get_task_spec(task_id)
    env = task_spec.env_factory(env_overrides)

    rewards: list[float] = []
    scores: list[int] = []
    for ep in range(eval_episodes):
        reset_result = env.reset(seed=seed + ep)
        if isinstance(reset_result, tuple):
            obs = np.asarray(reset_result[0], dtype=np.float32)
        else:
            obs = np.asarray(reset_result, dtype=np.float32)
        total_reward = 0.0
        for _ in range(max_steps):
            action = linear_greedy_action(weights, env, obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _info = step_result
            total_reward += float(reward)
            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                break
        rewards.append(total_reward)
        scores.append(int(getattr(env.mediator, "score", 0)))

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
    }


def _evaluate_gnn(
    task_id: str,
    model_path: Path,
    eval_episodes: int,
    max_steps: int,
    seed: int,
    env_overrides: Dict[str, Any],
) -> Dict[str, float]:
    payload = np.load(model_path, allow_pickle=True)
    task_spec = get_task_spec(task_id)
    env = task_spec.env_factory(env_overrides)
    hidden_dim = int(np.asarray(payload["w_self"]).shape[1])
    _, static = init_params(env, hidden_dim=hidden_dim, rng=np.random.default_rng(seed))
    params = GNNLiteParams(
        w_self=np.asarray(payload["w_self"], dtype=np.float32),
        w_neigh=np.asarray(payload["w_neigh"], dtype=np.float32),
        w_add=np.asarray(payload["w_add"], dtype=np.float32),
        b_add=np.asarray(payload["b_add"], dtype=np.float32),
        w_remove=np.asarray(payload["w_remove"], dtype=np.float32),
        b_remove=np.asarray(payload["b_remove"], dtype=np.float32),
        w_noop=np.asarray(payload["w_noop"], dtype=np.float32),
        b_noop=float(np.asarray(payload["b_noop"], dtype=np.float32).reshape(-1)[0]),
    )

    rewards: list[float] = []
    scores: list[int] = []
    for ep in range(eval_episodes):
        reset_result = env.reset(seed=seed + ep)
        if isinstance(reset_result, tuple):
            obs = np.asarray(reset_result[0], dtype=np.float32)
        else:
            obs = np.asarray(reset_result, dtype=np.float32)
        total_reward = 0.0
        for _ in range(max_steps):
            action = gnn_greedy_action(params, static, env, obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _info = step_result
            total_reward += float(reward)
            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                break
        rewards.append(total_reward)
        scores.append(int(getattr(env.mediator, "score", 0)))

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
    }


def run_comparison(
    *,
    task_id: str,
    train_episodes: int,
    train_max_steps: int,
    eval_episodes: int,
    eval_max_steps: int,
    seed: int,
    output_dir: Path,
    env_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_model_path = output_dir / "jongno_linear_compare.npz"
    gnn_model_path = output_dir / "jongno_gnn_compare.npz"

    linear_summary = train_linear(
        task_id=task_id,
        episodes=train_episodes,
        max_steps=train_max_steps,
        gamma=0.99,
        learning_rate=0.01,
        seed=seed,
        output_path=linear_model_path,
        env_overrides=env_overrides,
    )
    gnn_summary = train_gnn(
        task_id=task_id,
        episodes=train_episodes,
        max_steps=train_max_steps,
        gamma=0.99,
        learning_rate=0.005,
        hidden_dim=16,
        seed=seed,
        output_path=gnn_model_path,
        env_overrides=env_overrides,
    )

    linear_eval = _evaluate_linear(
        task_id, linear_model_path, eval_episodes, eval_max_steps, seed + 10_000, env_overrides
    )
    gnn_eval = _evaluate_gnn(
        task_id, gnn_model_path, eval_episodes, eval_max_steps, seed + 20_000, env_overrides
    )

    result = {
        "task_id": task_id,
        "train": {
            "episodes": train_episodes,
            "max_steps": train_max_steps,
        },
        "eval": {
            "episodes": eval_episodes,
            "max_steps": eval_max_steps,
        },
        "linear": {
            "train_summary": linear_summary,
            "eval_summary": linear_eval,
            "model_path": str(linear_model_path),
        },
        "gnn_lite": {
            "train_summary": gnn_summary,
            "eval_summary": gnn_eval,
            "model_path": str(gnn_model_path),
        },
        "delta": {
            "mean_reward_gnn_minus_linear": gnn_eval["mean_reward"]
            - linear_eval["mean_reward"],
            "mean_score_gnn_minus_linear": gnn_eval["mean_score"] - linear_eval["mean_score"],
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare linear vs GNN-lite policies"
    )
    parser.add_argument("--task", default="jongno_dispatch")
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--train-max-steps", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/compare")
    parser.add_argument(
        "--env-overrides",
        default=None,
        help='JSON dict to override env config, e.g. \'{"max_budget": 12}\'',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides: Dict[str, Any] = {}
    if args.env_overrides:
        overrides = dict(json.loads(args.env_overrides))
    result = run_comparison(
        task_id=str(args.task),
        train_episodes=int(args.train_episodes),
        train_max_steps=int(args.train_max_steps),
        eval_episodes=int(args.eval_episodes),
        eval_max_steps=int(args.eval_max_steps),
        seed=int(args.seed),
        output_dir=Path(args.output_dir),
        env_overrides=overrides,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
