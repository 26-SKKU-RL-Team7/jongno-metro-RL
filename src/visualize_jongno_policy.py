from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pygame

from config import framerate, screen_color, screen_height, screen_width
from gnn_lite_policy import GNNLiteParams, greedy_action as gnn_greedy_action, init_params
from jongno_policy import greedy_action
from task_catalog import get_task_spec
from ui.viewport import get_viewport_transform


def _get_window_size(window_surface: pygame.surface.Surface) -> tuple[int, int]:
    size = window_surface.get_size()
    if (
        isinstance(size, tuple)
        and len(size) == 2
        and isinstance(size[0], (int, float))
        and isinstance(size[1], (int, float))
    ):
        return (int(size[0]), int(size[1]))
    return (screen_width, screen_height)


def visualize(
    *,
    task_id: str,
    model_path: Path,
    model_type: str,
    max_steps: int,
    seed: int,
    env_overrides: dict[str, object],
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    payload = np.load(model_path, allow_pickle=True)

    task_spec = get_task_spec(task_id)
    env = task_spec.env_factory(env_overrides)
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        obs = np.asarray(reset_result[0], dtype=np.float32)
    else:
        obs = np.asarray(reset_result, dtype=np.float32)

    model_type = model_type.lower()
    weights = None
    gnn_params = None
    gnn_static = None
    if model_type == "linear":
        if "weights" not in payload:
            raise ValueError("Linear model requires `weights` key in .npz")
        weights = np.asarray(payload["weights"], dtype=np.float32)
    elif model_type == "gnn":
        hidden_dim = int(np.asarray(payload["w_self"]).shape[1])
        _, gnn_static = init_params(
            env, hidden_dim=hidden_dim, rng=np.random.default_rng(int(seed))
        )
        gnn_params = GNNLiteParams(
            w_self=np.asarray(payload["w_self"], dtype=np.float32),
            w_neigh=np.asarray(payload["w_neigh"], dtype=np.float32),
            w_add=np.asarray(payload["w_add"], dtype=np.float32),
            b_add=np.asarray(payload["b_add"], dtype=np.float32),
            w_remove=np.asarray(payload["w_remove"], dtype=np.float32),
            b_remove=np.asarray(payload["b_remove"], dtype=np.float32),
            w_noop=np.asarray(payload["w_noop"], dtype=np.float32),
            b_noop=float(np.asarray(payload["b_noop"], dtype=np.float32).reshape(-1)[0]),
        )
    else:
        raise ValueError("model_type must be one of: linear, gnn")

    pygame.init()
    flags = pygame.RESIZABLE
    window_surface = pygame.display.set_mode((screen_width, screen_height), flags, vsync=1)
    game_surface = pygame.Surface((screen_width, screen_height))
    clock = pygame.time.Clock()

    step_idx = 0
    done = False
    while not done and step_idx < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if model_type == "linear":
            assert weights is not None
            action = greedy_action(weights, env, obs)
        else:
            assert gnn_params is not None and gnn_static is not None
            action = gnn_greedy_action(gnn_params, gnn_static, env, obs)
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_result
        del reward, info
        obs = np.asarray(next_obs, dtype=np.float32)

        game_surface.fill(screen_color)
        env.mediator.render(game_surface)

        window_width, window_height = _get_window_size(window_surface)
        viewport = get_viewport_transform(
            window_width, window_height, screen_width, screen_height
        )
        if (viewport.width, viewport.height) != game_surface.get_size():
            scaled_surface = pygame.transform.smoothscale(
                game_surface, (viewport.width, viewport.height)
            )
        else:
            scaled_surface = game_surface
        window_surface.fill(screen_color)
        window_surface.blit(scaled_surface, (viewport.offset_x, viewport.offset_y))
        pygame.display.flip()
        clock.tick(framerate)
        step_idx += 1

    pygame.quit()
    print(
        json.dumps(
            {
                "task_id": task_id,
                "steps_executed": step_idx,
                "final_score": int(getattr(env.mediator, "score", 0)),
                "done": bool(done),
                "model_type": model_type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize passenger flow with a trained Jongno policy"
    )
    parser.add_argument("--task", default="jongno_dispatch", help="Task ID")
    parser.add_argument(
        "--model",
        default="artifacts/jongno_policy.npz",
        help="Path to trained .npz model",
    )
    parser.add_argument(
        "--model-type",
        default="linear",
        choices=["linear", "gnn"],
        help="Model type stored in .npz",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-overrides",
        default=None,
        help='JSON dict to override env config, e.g. \'{"dt_ms": 500}\'',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides = {}
    if args.env_overrides:
        overrides = dict(json.loads(args.env_overrides))
    visualize(
        task_id=args.task,
        model_path=Path(args.model),
        model_type=str(args.model_type),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        env_overrides=overrides,
    )


if __name__ == "__main__":
    main()
