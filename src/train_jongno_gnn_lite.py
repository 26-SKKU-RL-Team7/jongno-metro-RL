from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnn_lite_policy import (
    GNNLiteParams,
    add_gradients,
    apply_gradients,
    backward,
    init_params,
    sample_action,
    zeros_like,
)
from task_catalog import get_task_spec


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    if len(returns) > 1:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns


def _save_params(path: Path, params: GNNLiteParams, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        model_type="gnn_lite",
        w_self=params.w_self,
        w_neigh=params.w_neigh,
        w_add=params.w_add,
        b_add=params.b_add,
        w_remove=params.w_remove,
        b_remove=params.b_remove,
        w_noop=params.w_noop,
        b_noop=np.array([params.b_noop], dtype=np.float32),
        task_id=meta["task_id"],
        seed=meta["seed"],
        episodes=meta["episodes"],
        max_steps=meta["max_steps"],
        gamma=meta["gamma"],
        learning_rate=meta["learning_rate"],
        hidden_dim=meta["hidden_dim"],
    )


def train(
    *,
    task_id: str,
    episodes: int,
    max_steps: int,
    gamma: float,
    learning_rate: float,
    hidden_dim: int,
    seed: int,
    output_path: Path,
    env_overrides: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    task_spec = get_task_spec(task_id)
    env = task_spec.env_factory(env_overrides)
    params, static = init_params(env, hidden_dim=hidden_dim, rng=rng)

    rewards_history: list[float] = []
    scores_history: list[int] = []
    best_score = -10**9
    best_params = params

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        reset_result = env.reset(seed=episode_seed)
        if isinstance(reset_result, tuple):
            obs = np.asarray(reset_result[0], dtype=np.float32)
        else:
            obs = np.asarray(reset_result, dtype=np.float32)

        episode_rewards: list[float] = []
        episode_actions: list[int] = []
        episode_probs: list[np.ndarray] = []
        episode_cache: list[dict[str, np.ndarray]] = []

        for _ in range(max_steps):
            action, probs, cache = sample_action(params, static, env, obs, rng)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _info = step_result
            episode_actions.append(action)
            episode_probs.append(probs)
            episode_cache.append(cache)
            episode_rewards.append(float(reward))
            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                break

        returns = discounted_returns(episode_rewards, gamma)
        grads = zeros_like(params)
        for step_idx, action in enumerate(episode_actions):
            probs = episode_probs[step_idx]
            one_hot = np.zeros_like(probs, dtype=np.float32)
            one_hot[action] = 1.0
            dlogits = (one_hot - probs) * returns[step_idx]
            step_grads = backward(params, static, episode_cache[step_idx], dlogits)
            grads = add_gradients(grads, step_grads)

        params = apply_gradients(params, grads, learning_rate)

        total_reward = float(np.sum(episode_rewards))
        final_score = int(getattr(env.mediator, "score", 0))
        rewards_history.append(total_reward)
        scores_history.append(final_score)
        if final_score > best_score:
            best_score = final_score
            best_params = GNNLiteParams(
                w_self=params.w_self.copy(),
                w_neigh=params.w_neigh.copy(),
                w_add=params.w_add.copy(),
                b_add=params.b_add.copy(),
                w_remove=params.w_remove.copy(),
                b_remove=params.b_remove.copy(),
                w_noop=params.w_noop.copy(),
                b_noop=float(params.b_noop),
            )

        if (episode_idx + 1) % 20 == 0 or episode_idx == 0:
            print(
                f"[gnn-lite {episode_idx + 1}/{episodes}] "
                f"reward={total_reward:.3f} score={final_score}"
            )

    meta = {
        "task_id": task_id,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
    }
    _save_params(output_path, best_params, meta)

    return {
        "task_id": task_id,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "best_score": int(best_score),
        "last_score": int(scores_history[-1]) if scores_history else 0,
        "mean_reward": float(np.mean(rewards_history)) if rewards_history else 0.0,
        "model_path": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GNN-lite policy on JongnoMetroEnv")
    parser.add_argument("--task", default="jongno_dispatch", help="Task ID")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="artifacts/jongno_gnn_lite.npz",
        help="Output .npz path for trained GNN-lite params",
    )
    parser.add_argument(
        "--env-overrides",
        default=None,
        help='JSON dict to override env config, e.g. \'{"max_budget": 12}\'',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides = {}
    if args.env_overrides:
        overrides = dict(json.loads(args.env_overrides))
    summary = train(
        task_id=args.task,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        gamma=float(args.gamma),
        learning_rate=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        output_path=Path(args.output),
        env_overrides=overrides,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
