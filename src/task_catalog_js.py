from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from jongno_env_js import JongnoMetroEnv


EnvFactory = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    env_factory: EnvFactory
    default_steps: int = 600


def _build_jongno_dispatch(overrides: Dict[str, Any]) -> JongnoMetroEnv:
    config = {
        "max_budget": 15,
        "dt_ms": 1000,
        "hour_advance_per_step": 1.0,
        "demand_spawn_scale": 0.005,
        "reward_waiting_weight": 0.1,
        "reward_congestion_weight": 2.0,
        "rush_hour_weight": 2.0,
        "score_reward_weight": 1.0,
        "invalid_action_penalty": 5.0,
        "budget_exceeded_penalty": 2.0,
    }
    config.update(overrides)
    return JongnoMetroEnv(**config)


def _build_jongno_peak_stress(overrides: Dict[str, Any]) -> JongnoMetroEnv:
    config = {
        "max_budget": 15,
        "dt_ms": 1000,
        "hour_advance_per_step": 1.5,
        "demand_spawn_scale": 0.008,
        "reward_waiting_weight": 0.25,
        "reward_congestion_weight": 2.0,
        "rush_hour_weight": 2.0,
        "score_reward_weight": 1.0,
        "invalid_action_penalty": 5.0,
        "budget_exceeded_penalty": 3.0,
    }
    config.update(overrides)
    return JongnoMetroEnv(**config)


TASK_SPECS: Dict[str, TaskSpec] = {
    "jongno_dispatch": TaskSpec(
        task_id="jongno_dispatch",
        description="Baseline fixed-network train dispatch optimization",
        env_factory=_build_jongno_dispatch,
        default_steps=600,
    ),
    "jongno_peak_stress": TaskSpec(
        task_id="jongno_peak_stress",
        description="Peak-hour stress dispatch with stronger waiting penalties",
        env_factory=_build_jongno_peak_stress,
        default_steps=800,
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    return TASK_SPECS[task_id]
