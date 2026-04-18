from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jongno_policy import sample_action
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


def train(
    *,
    task_id: str,
    episodes: int,
    max_steps: int,
    gamma: float,
    learning_rate: float,
    seed: int,
    output_path: Path,
    env_overrides: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    task_spec = get_task_spec(task_id)
    env = task_spec.env_factory(env_overrides)

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs = np.asarray(reset_result[0], dtype=np.float32)
    else:
        obs = np.asarray(reset_result, dtype=np.float32)

    num_actions = int(env.action_space.n)
    feature_dim = int(
        getattr(env, "policy_feature_dim", 3 + env.num_stations + env.num_lines)
    )
    weights = rng.normal(loc=0.0, scale=0.01, size=(num_actions, feature_dim)).astype(
        np.float32
    )

    rewards_history: list[float] = []
    scores_history: list[int] = []
    best_score = -10**9
    best_weights = weights.copy()

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        reset_result = env.reset(seed=episode_seed)
        if isinstance(reset_result, tuple):
            obs = np.asarray(reset_result[0], dtype=np.float32)
        else:
            obs = np.asarray(reset_result, dtype=np.float32)

        probs_list: list[np.ndarray] = []
        features_list: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps):
            action, probs, features = sample_action(weights, env, obs, rng)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _info = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _info = step_result

            probs_list.append(probs)
            features_list.append(features)
            actions.append(int(action))
            rewards.append(float(reward))
            obs = np.asarray(next_obs, dtype=np.float32)
            if done:
                break

        returns = discounted_returns(rewards, gamma)
        grad = np.zeros_like(weights, dtype=np.float32)
        for step_idx, action in enumerate(actions):
            probs = probs_list[step_idx]
            features = features_list[step_idx]
            one_hot = np.zeros_like(probs)
            one_hot[action] = 1.0
            # REINFORCE gradient for linear softmax policy.
            grad += np.outer((one_hot - probs) * returns[step_idx], features)

        weights += learning_rate * grad

        total_reward = float(np.sum(rewards))
        final_score = int(getattr(env.mediator, "score", 0))
        rewards_history.append(total_reward)
        scores_history.append(final_score)
        if final_score > best_score:
            best_score = final_score
            best_weights = weights.copy()

        if (episode_idx + 1) % 20 == 0 or episode_idx == 0:
            print(
                f"[episode {episode_idx + 1}/{episodes}] "
                f"reward={total_reward:.3f} score={final_score}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        weights=best_weights,
        task_id=task_id,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    summary = {
        "task_id": task_id,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "best_score": int(best_score),
        "last_score": int(scores_history[-1]) if scores_history else 0,
        "mean_reward": float(np.mean(rewards_history)) if rewards_history else 0.0,
        "model_path": str(output_path),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight masked policy on JongnoMetroEnv"
    )
    parser.add_argument("--task", default="jongno_dispatch", help="Task ID")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="artifacts/jongno_policy.npz",
        help="Output .npz path for trained weights",
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
        seed=int(args.seed),
        output_path=Path(args.output),
        env_overrides=overrides,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
