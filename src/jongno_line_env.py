"""
Stage-1 Gym environment: choose exactly one new line from `candidate_new_lines`
in jongno_config.json, then score it with a short no-learning simulation rollout.

Stage-2 dispatch uses `JongnoMetroEnv(..., extra_lines={...})` with the chosen line.
"""

from __future__ import annotations

import json
import os
import random
import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mediator import Mediator
from entity.passenger import Passenger
from entity.station import Station
from entity.path import Path
from entity.metro import Metro
from geometry.point import Point
from geometry.type import ShapeType
from utils import get_shape_from_type
from config import (
    station_color,
    station_size,
    passenger_color,
    passenger_size,
    passenger_max_wait_time_ms as default_passenger_max_wait_time_ms,
    max_waiting_passengers as default_max_waiting_passengers,
)


class JongnoLineSelectionEnv(gym.Env):
    """
    One decision per episode: discrete action picks candidate index.
    Reward comes from simulation steps after the new path is created.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | None = None,
        max_budget: int = 15,
        dt_ms: int = 1000,
        hour_advance_per_step: float = 1.0,
        demand_spawn_scale: float = 0.005,
        reward_waiting_weight: float = 0.1,
        score_reward_weight: float = 1.0,
        invalid_action_penalty: float = 25.0,
        line_eval_rollout_steps: int = 120,
        passenger_max_wait_time_ms: int = default_passenger_max_wait_time_ms,
        max_waiting_passengers: int = default_max_waiting_passengers,
        candidate_line_ids: List[str] | None = None,
    ):
        super().__init__()
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "jongno_config.json"
            )
        self.env_config_path = os.path.abspath(config_path)
        with open(self.env_config_path, "r", encoding="utf-8") as f:
            self.env_config: Dict[str, Any] = json.load(f)

        raw_candidates = self.env_config.get("candidate_new_lines")
        if not raw_candidates or not isinstance(raw_candidates, list):
            raise ValueError(
                "jongno_config.json must contain a non-empty list "
                '"candidate_new_lines" (each entry: list of station names).'
            )

        self.candidates: List[List[str]] = []
        self._candidate_feasible: List[bool] = []
        for idx, c in enumerate(raw_candidates):
            if isinstance(c, dict) and "stations" in c:
                names = list(c["stations"])
            else:
                names = list(c)
            ok = (
                len(names) >= 2
                and all(isinstance(n, str) for n in names)
            )
            self._candidate_feasible.append(ok)
            self.candidates.append(names)

        if candidate_line_ids is not None:
            if len(candidate_line_ids) != len(self.candidates):
                raise ValueError(
                    "candidate_line_ids length must match candidate_new_lines"
                )
            self.candidate_ids = list(candidate_line_ids)
        else:
            self.candidate_ids = [f"new_line_{i}" for i in range(len(self.candidates))]

        self.num_candidates = len(self.candidates)
        if self.num_candidates == 0:
            raise ValueError("No candidates loaded.")

        self.station_name_list: List[str] = list(self.env_config["stations"].keys())
        self.station_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.station_name_list)
        }
        self.num_stations = len(self.station_name_list)

        self.demand_hour_keys: List[str] = sorted(self.env_config["demand"].keys())
        self.num_hours = len(self.demand_hour_keys)

        self.max_budget = max_budget
        self.dt_ms = dt_ms
        self.hour_advance_per_step = hour_advance_per_step
        self.demand_spawn_scale = demand_spawn_scale
        self.reward_waiting_weight = reward_waiting_weight
        self.score_reward_weight = score_reward_weight
        self.invalid_action_penalty = invalid_action_penalty
        self.line_eval_rollout_steps = int(line_eval_rollout_steps)
        self.passenger_max_wait_time_ms = passenger_max_wait_time_ms
        self.max_waiting_passengers = max_waiting_passengers

        self.action_space = spaces.Discrete(self.num_candidates)
        # Mean spawn rate per station (across hours) as a compact demand summary.
        self._mean_spawn = self._compute_mean_spawn_rates()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stations,),
            dtype=np.float32,
        )
        self.policy_feature_dim = self.num_stations + 1

        self._dest_probs_no_origin_by_hour: List[np.ndarray] = []
        self._hour_spawn_rates_by_station: List[np.ndarray] = []

        self.current_hour_float = 0.0
        self.current_hour_idx = 0
        self.last_score = 0
        self.mediator: Mediator | None = None
        self.station_name_to_obj: Dict[str, Station] = {}
        self._episode_active = False

    def _compute_mean_spawn_rates(self) -> np.ndarray:
        acc = np.zeros((self.num_stations,), dtype=np.float64)
        n = 0
        for hour_key in self.demand_hour_keys:
            demand = self.env_config["demand"].get(hour_key, {})
            rates = demand.get("spawn_rates", {})
            for i, name in enumerate(self.station_name_list):
                acc[i] += float(rates.get(name, 0.0))
            n += 1
        if n == 0:
            return np.zeros_like(acc, dtype=np.float32)
        mean = acc / float(n)
        m = float(np.max(mean)) if mean.size else 1.0
        if m <= 0:
            m = 1.0
        return (mean / m).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._mean_spawn.copy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.mediator = Mediator(is_rl_mode=True)
        self.mediator.num_metros = self.max_budget
        self.mediator.passenger_max_wait_time_ms = self.passenger_max_wait_time_ms
        self.mediator.max_waiting_passengers = self.max_waiting_passengers

        self._build_base_map()
        self._precompute_demand_sampling()

        self.current_hour_float = 0.0
        self.current_hour_idx = 0
        self.last_score = self.mediator.score
        self._episode_active = True

        for i, names in enumerate(self.candidates):
            ok = self._candidate_feasible[i]
            if ok:
                for n in names:
                    if n not in self.station_name_to_obj:
                        ok = False
                        break
            self._candidate_feasible[i] = ok

        info = {"candidate_feasible": list(self._candidate_feasible)}
        return self._get_obs(), info

    def _build_base_map(self) -> None:
        assert self.mediator is not None
        self.station_name_to_obj = {}
        shapes = [
            ShapeType.CIRCLE,
            ShapeType.TRIANGLE,
            ShapeType.RECT,
            ShapeType.CROSS,
            ShapeType.DIAMOND,
        ]
        for idx, (st_name, coords) in enumerate(self.env_config["stations"].items()):
            shape_obj = get_shape_from_type(
                shapes[idx % len(shapes)], station_color, station_size
            )
            station = Station(shape=shape_obj, position=Point(coords["x"], coords["y"]))
            station.name = st_name
            self.mediator.stations.append(station)
            self.station_name_to_obj[st_name] = station

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
        lines: Dict[str, List[str]] = self.env_config["lines"]
        for idx, (line_name, route_station_names) in enumerate(lines.items()):
            path = Path(color=colors[idx % len(colors)])
            path.id = line_name
            for st_name in route_station_names:
                if st_name in self.station_name_to_obj:
                    path.add_station(self.station_name_to_obj[st_name])
            self.mediator.paths.append(path)

        num_base = len(self.mediator.paths)
        self.mediator.rl_path_creation_cap = num_base + 1

        for path in self.mediator.paths:
            if len(self.mediator.metros) >= self.max_budget:
                break
            metro = Metro()
            path.add_metro(metro)
            self.mediator.metros.append(metro)

        self.mediator.find_travel_plan_for_passengers()

    def _precompute_demand_sampling(self) -> None:
        assert self.mediator is not None
        self._hour_spawn_rates_by_station = []
        self._dest_probs_no_origin_by_hour = []

        for hour_key in self.demand_hour_keys:
            demand_for_hour = self.env_config["demand"].get(hour_key, {})
            spawn_rates: Dict[str, float] = demand_for_hour.get("spawn_rates", {})
            dest_probs: Dict[str, float] = demand_for_hour.get("dest_probs", {})

            spawn_arr = np.array(
                [float(spawn_rates.get(name, 0.0)) for name in self.station_name_list],
                dtype=np.float32,
            )
            self._hour_spawn_rates_by_station.append(spawn_arr)

            base_probs = np.array(
                [float(dest_probs.get(name, 0.0)) for name in self.station_name_list],
                dtype=np.float64,
            )
            hour_origin_dest_probs = np.zeros(
                (len(self.station_name_list), len(self.station_name_list)),
                dtype=np.float64,
            )
            for origin_idx in range(len(self.station_name_list)):
                probs = base_probs.copy()
                probs[origin_idx] = 0.0
                s = probs.sum()
                if s <= 0:
                    other_idxs = [
                        i for i in range(len(self.station_name_list)) if i != origin_idx
                    ]
                    probs = np.zeros_like(base_probs, dtype=np.float64)
                    if other_idxs:
                        probs[other_idxs] = 1.0 / len(other_idxs)
                    else:
                        probs[origin_idx] = 1.0
                else:
                    probs /= s
                hour_origin_dest_probs[origin_idx] = probs

            self._dest_probs_no_origin_by_hour.append(hour_origin_dest_probs)

        def _spawn_passengers_from_demand() -> None:
            assert self.mediator is not None
            if self.num_hours == 0:
                return
            hour_idx = self.current_hour_idx
            spawn_arr = self._hour_spawn_rates_by_station[hour_idx]
            dt_minutes = self.dt_ms / 60_000.0
            spawn_scale = self.demand_spawn_scale

            for origin_idx, station_name in enumerate(self.station_name_list):
                station = self.station_name_to_obj[station_name]
                lam = float(spawn_arr[origin_idx]) * dt_minutes * spawn_scale
                if lam <= 0:
                    continue
                spawn_count = int(np.random.poisson(lam))
                if spawn_count <= 0:
                    continue
                available = station.capacity - len(station.passengers)
                if available <= 0:
                    continue
                spawn_count = min(spawn_count, available)
                dest_probs = self._dest_probs_no_origin_by_hour[hour_idx][origin_idx]
                for _ in range(spawn_count):
                    dest_idx = int(
                        np.random.choice(len(self.station_name_list), p=dest_probs)
                    )
                    dest_name = self.station_name_list[dest_idx]
                    dest_station = self.station_name_to_obj[dest_name]
                    dest_shape = get_shape_from_type(
                        dest_station.shape.type, passenger_color, passenger_size
                    )
                    passenger = Passenger(dest_shape)
                    passenger.destination_station_name = dest_name
                    station.add_passenger(passenger)
                    self.mediator.passengers.append(passenger)

        self.mediator.is_passenger_spawn_time = lambda: True
        self.mediator.spawn_passengers = _spawn_passengers_from_demand  # type: ignore[method-assign]

    def _waiting_ratio_sum(self) -> float:
        assert self.mediator is not None
        if self.mediator.passenger_max_wait_time_ms <= 0:
            return 0.0
        return float(
            sum(
                (p.wait_ms / self.mediator.passenger_max_wait_time_ms)
                for p in self.mediator.passengers
            )
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.mediator is not None
        if not self._episode_active:
            raise RuntimeError("Call reset() before step().")

        self._episode_active = False
        terminated = True
        truncated = False

        if (
            not isinstance(action, (int, np.integer))
            or int(action) < 0
            or int(action) >= self.num_candidates
        ):
            reward = -self.invalid_action_penalty
            info = {
                "action_ok": False,
                "reason": "invalid_action_index",
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        cand_idx = int(action)
        if not self._candidate_feasible[cand_idx]:
            reward = -self.invalid_action_penalty
            info = {
                "action_ok": False,
                "reason": "infeasible_candidate",
                "candidate_idx": cand_idx,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        names = self.candidates[cand_idx]
        indices = [self.station_name_to_idx[n] for n in names]
        path = self.mediator.create_path_from_station_indices(indices, loop=False)
        if path is None:
            reward = -self.invalid_action_penalty
            info = {
                "action_ok": False,
                "reason": "create_path_failed",
                "candidate_idx": cand_idx,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        path.id = self.candidate_ids[cand_idx]
        self.mediator.find_travel_plan_for_passengers()

        score_before = self.mediator.score
        wait_before = self._waiting_ratio_sum()

        for _ in range(self.line_eval_rollout_steps):
            self.current_hour_float += self.hour_advance_per_step
            self.current_hour_idx = int(self.current_hour_float) % max(1, self.num_hours)
            self.mediator.step_time(self.dt_ms)

        score_delta = self.mediator.score - score_before
        wait_after = self._waiting_ratio_sum()
        wait_penalty_term = self.reward_waiting_weight * (wait_after - wait_before)
        reward = (
            self.score_reward_weight * float(score_delta)
            - wait_penalty_term
        )

        info = {
            "action_ok": True,
            "candidate_idx": cand_idx,
            "chosen_line_id": path.id,
            "chosen_station_names": list(names),
            "score_delta": float(score_delta),
            "extra_lines": {path.id: list(names)},
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def merge_extra_lines_into_config(
    base_config_path: str, extra_lines: Dict[str, List[str]], out_path: str
) -> None:
    """Write a new JSON config = base + extra_lines merged into ``lines``."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lines = dict(cfg.get("lines", {}))
    lines.update(extra_lines)
    cfg = copy.deepcopy(cfg)
    cfg["lines"] = lines
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
