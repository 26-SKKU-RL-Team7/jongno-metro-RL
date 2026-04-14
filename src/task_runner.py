from __future__ import annotations

import argparse
import importlib
import json
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from task_catalog import TASK_SPECS, get_task_spec


PolicyFn = Callable[[Any, Any, Dict[str, Any]], int]


def random_policy(env: Any, obs: Any, info: Dict[str, Any]) -> int:
    del obs, info
    return int(env.action_space.sample())


def keep_capacity_policy(env: Any, obs: Any, info: Dict[str, Any]) -> int:
    del info
    num_lines = env.num_lines
    line_counts = np.asarray(obs[env.num_stations : env.num_stations + num_lines])
    min_idx = int(np.argmin(line_counts))
    max_idx = int(np.argmax(line_counts))
    min_count = float(line_counts[min_idx]) if line_counts.size > 0 else 0.0
    max_count = float(line_counts[max_idx]) if line_counts.size > 0 else 0.0

    # action: 1..num_lines -> add, num_lines+1..2*num_lines -> remove, 0 -> noop
    if max_count - min_count >= 2 and max_count > 1:
        return num_lines + 1 + max_idx
    if min_count <= 1 and np.sum(line_counts) < env.max_budget:
        return 1 + min_idx
    return 0


BUILT_IN_POLICIES: Dict[str, PolicyFn] = {
    "random": random_policy,
    "keep_capacity": keep_capacity_policy,
}


def _parse_env_overrides(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    return dict(json.loads(raw))


def _load_policy(policy_name: str) -> PolicyFn:
    if policy_name in BUILT_IN_POLICIES:
        return BUILT_IN_POLICIES[policy_name]
    if ":" not in policy_name:
        raise ValueError(
            "Custom policy must be in module:function format, e.g. my_policy:act"
        )
    module_name, func_name = policy_name.split(":", 1)
    module = importlib.import_module(module_name)
    policy = getattr(module, func_name)
    if not callable(policy):
        raise ValueError(f"Policy {policy_name} is not callable")
    return policy


def _step_env(env: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, float(reward), done, dict(info)
    obs, reward, done, info = step_result
    return obs, float(reward), bool(done), dict(info)


def run_task(
    task_id: str,
    policy_name: str,
    steps: int | None = None,
    seed: int | None = 42,
    env_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    env_overrides = env_overrides or {}
    task_spec = get_task_spec(task_id)
    policy = _load_policy(policy_name)
    effective_steps = steps if steps is not None else task_spec.default_steps
    env = task_spec.env_factory(env_overrides)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs, info = reset_result, {}

    total_reward = 0.0
    steps_executed = 0
    done = False
    last_info: Dict[str, Any] = dict(info)
    for _ in range(effective_steps):
        action = int(policy(env, obs, last_info))
        obs, reward, done, last_info = _step_env(env, action)
        total_reward += reward
        steps_executed += 1
        if done:
            break

    final_score = int(getattr(env.mediator, "score", 0))
    return {
        "task_id": task_id,
        "task_description": task_spec.description,
        "policy": policy_name,
        "seed": seed,
        "steps_requested": effective_steps,
        "steps_executed": steps_executed,
        "done": done,
        "total_reward": total_reward,
        "final_score": final_score,
        "last_info": last_info,
    }


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id": spec.task_id,
            "description": spec.description,
            "default_steps": spec.default_steps,
        }
        for spec in TASK_SPECS.values()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selectable RL tasks")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument(
        "--task",
        default="jongno_dispatch",
        choices=sorted(TASK_SPECS.keys()),
        help="Task ID to run",
    )
    parser.add_argument(
        "--policy",
        default="random",
        help="Policy name: built-in or module:function",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--env-overrides",
        default=None,
        help='JSON dict to override env config, e.g. \'{"max_budget": 10}\'',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list_tasks(), ensure_ascii=False, indent=2))
        return
    result = run_task(
        task_id=args.task,
        policy_name=args.policy,
        steps=args.steps,
        seed=args.seed,
        env_overrides=_parse_env_overrides(args.env_overrides),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
