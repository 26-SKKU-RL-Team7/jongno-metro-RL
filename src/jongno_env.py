import json
import os
import random
from typing import Dict, List, Tuple

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
    station_capacity,
    station_color,
    station_size,
    passenger_color,
    passenger_size,
    passenger_max_wait_time_ms as default_passenger_max_wait_time_ms,
    max_waiting_passengers as default_max_waiting_passengers,
)

class JongnoMetroEnv(gym.Env):
    def __init__(
        self,
        config_path: str | None = None,
        max_budget: int = 15,
        dt_ms: int = 1000,
        hour_advance_per_step: float = 1.0,
        demand_spawn_scale: float = 0.005,
        reward_waiting_weight: float = 0.1,
        score_reward_weight: float = 1.0,
        invalid_action_penalty: float = 5.0,
        budget_exceeded_penalty: float = 2.0,
        passenger_max_wait_time_ms: int = default_passenger_max_wait_time_ms,
        max_waiting_passengers: int = default_max_waiting_passengers,
    ):
        super().__init__()
        # 1. JSON 설정 로드하여 실제 가용 노선 수 확인
        if config_path is None:
            # src/ -> project root
            config_path = os.path.join(os.path.dirname(__file__), "..", "jongno_config.json")
        self.env_config_path = os.path.abspath(config_path)
        with open(self.env_config_path, "r", encoding="utf-8") as f:
            self.env_config: Dict = json.load(f)
        
        self.num_stations = len(self.env_config["stations"])
        self.num_lines = len(self.env_config["lines"]) # 실제 생성될 노선 수
        self.max_budget = max_budget
        self.dt_ms = dt_ms
        self.hour_advance_per_step = hour_advance_per_step
        self.demand_spawn_scale = demand_spawn_scale
        self.reward_waiting_weight = reward_waiting_weight
        self.score_reward_weight = score_reward_weight
        self.invalid_action_penalty = invalid_action_penalty
        self.budget_exceeded_penalty = budget_exceeded_penalty
        self.passenger_max_wait_time_ms = passenger_max_wait_time_ms
        self.max_waiting_passengers = max_waiting_passengers

        # Station order for consistent observation/demand indexing.
        self.station_name_list: List[str] = list(self.env_config["stations"].keys())
        self.station_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.station_name_list)
        }

        self.demand_hour_keys: List[str] = sorted(self.env_config["demand"].keys())
        self.num_hours = len(self.demand_hour_keys)
        
        # Action Space: 0(대기) + 추가(num_lines) + 회수(num_lines)
        self.action_space = spaces.Discrete(1 + self.num_lines * 2)
        
        # State: [역별 승객 수] + [노선별 가동 열차 수] + [현재 시간 인덱스]
        obs_high = float(max(station_capacity, self.max_budget))
        self.observation_space = spaces.Box(
            low=0,
            high=obs_high,
            shape=(self.num_stations + self.num_lines + 1,), 
            dtype=np.float32
        )

        # demand-based sampling caches
        self._dest_probs_no_origin_by_hour: List[np.ndarray] = []
        self._hour_spawn_rates_by_station: List[np.ndarray] = []

        # runtime
        self.current_hour_float = 0.0
        self.current_hour_idx = 0
        self.last_score = 0
        self.mediator: Mediator | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.mediator = Mediator(is_rl_mode=True)
        # RL budget: allow adding up to max_budget metros total.
        self.mediator.num_metros = self.max_budget
        # Match passenger waiting rules to demand-based scaling.
        self.mediator.passenger_max_wait_time_ms = self.passenger_max_wait_time_ms
        self.mediator.max_waiting_passengers = self.max_waiting_passengers

        self._build_fixed_map()
        self._precompute_demand_sampling()

        self.current_hour_float = 0.0  # start from the first hour key
        self.current_hour_idx = 0
        self.last_score = self.mediator.score
        return self._get_obs(), {}

    def _precompute_demand_sampling(self) -> None:
        """
        Precompute destination sampling distributions aligned with `station_name_list`.
        For each hour and each origin station we store P(dest | hour, origin) with origin excluded.
        """
        assert self.mediator is not None

        # Spawn rates per station per hour
        self._hour_spawn_rates_by_station = []
        # Destination distribution per (hour, origin_idx)
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

            # For each origin, zero-out origin probability and renormalize.
            hour_origin_dest_probs = np.zeros((len(self.station_name_list), len(self.station_name_list)), dtype=np.float64)
            for origin_idx in range(len(self.station_name_list)):
                probs = base_probs.copy()
                probs[origin_idx] = 0.0
                s = probs.sum()
                if s <= 0:
                    # If demand data is missing/zero, fall back to uniform over other stations.
                    other_idxs = [i for i in range(len(self.station_name_list)) if i != origin_idx]
                    probs = np.zeros_like(base_probs, dtype=np.float64)
                    if other_idxs:
                        probs[other_idxs] = 1.0 / len(other_idxs)
                    else:
                        probs[origin_idx] = 1.0
                else:
                    probs /= s
                hour_origin_dest_probs[origin_idx] = probs

            self._dest_probs_no_origin_by_hour.append(hour_origin_dest_probs)

        # Enable demand-driven passenger spawning (override Mediator's spawn logic)
        def _spawn_passengers_from_demand() -> None:
            assert self.mediator is not None
            if self.num_hours == 0:
                return
            hour_key = self.demand_hour_keys[self.current_hour_idx]
            demand_for_hour = self.env_config["demand"].get(hour_key, {})
            spawn_rates = demand_for_hour.get("spawn_rates", {})

            # Convert per-minute demand into "this step's expected arrivals".
            dt_minutes = self.dt_ms / 60_000.0
            spawn_scale = self.demand_spawn_scale

            hour_idx = self.current_hour_idx
            spawn_arr = self._hour_spawn_rates_by_station[hour_idx]

            for origin_idx, station_name in enumerate(self.station_name_list):
                station = self.station_name_to_obj[station_name]
                # expected arrivals from demand (Poisson)
                lam = float(spawn_arr[origin_idx]) * dt_minutes * spawn_scale
                if lam <= 0:
                    continue
                # Poisson sample for this station during this step
                spawn_count = int(np.random.poisson(lam))
                if spawn_count <= 0:
                    continue

                available = station.capacity - len(station.passengers)
                if available <= 0:
                    continue
                spawn_count = min(spawn_count, available)

                dest_probs = self._dest_probs_no_origin_by_hour[hour_idx][origin_idx]
                for _ in range(spawn_count):
                    dest_idx = int(np.random.choice(len(self.station_name_list), p=dest_probs))
                    dest_name = self.station_name_list[dest_idx]
                    dest_station = self.station_name_to_obj[dest_name]

                    dest_shape = get_shape_from_type(
                        dest_station.shape.type, passenger_color, passenger_size
                    )
                    passenger = Passenger(dest_shape)
                    passenger.destination_station_name = dest_name  # used by Mediator routing

                    station.add_passenger(passenger)
                    self.mediator.passengers.append(passenger)

        self.mediator.is_passenger_spawn_time = lambda: True
        self.mediator.spawn_passengers = _spawn_passengers_from_demand  # type: ignore[method-assign]

    def _build_fixed_map(self):
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
            shape_obj = get_shape_from_type(shapes[idx % len(shapes)], station_color, station_size)
            station = Station(shape=shape_obj, position=Point(coords["x"], coords["y"]))
            station.name = st_name
            self.mediator.stations.append(station)
            self.station_name_to_obj[st_name] = station

        colors = [(255,0,0), (0,255,0), (0,0,255), (255,165,0), (128,0,128)]
        for idx, (line_name, route_station_names) in enumerate(self.env_config["lines"].items()):
            path = Path(color=colors[idx % len(colors)])
            path.id = line_name
            for st_name in route_station_names:
                if st_name in self.station_name_to_obj:
                    path.add_station(self.station_name_to_obj[st_name])
            self.mediator.paths.append(path)

        # Fixed network: create initial 1 metro per path (up to budget).
        for path in self.mediator.paths:
            if len(self.mediator.metros) >= self.max_budget:
                break
            metro = Metro()
            path.add_metro(metro) # 노선에 열차 할당
            self.mediator.metros.append(metro) # 시스템 전체 열차 목록에 추가

        # No passengers yet, but keep internal travel plan structures valid.
        self.mediator.find_travel_plan_for_passengers()

    def _get_obs(self):
        assert self.mediator is not None
        obs = np.zeros(self.num_stations + self.num_lines + 1, dtype=np.float32)
        for i, station in enumerate(self.mediator.stations):
            obs[i] = len(station.passengers)
        for i, path in enumerate(self.mediator.paths):
            obs[self.num_stations + i] = sum(1 for m in path.metros if not getattr(m, 'is_retiring', False))
        obs[-1] = float(self.current_hour_idx)
        return obs

    def step(self, action):
        assert self.mediator is not None

        penalty = 0.0
        action_ok = True

        # 1) Apply action at current state
        if 1 <= action <= self.num_lines:
            path_idx = action - 1
            path_id = self.mediator.paths[path_idx].id
            if len(self.mediator.metros) < self.max_budget:
                action_ok = self.mediator.apply_action({"type": "add_train", "path_id": path_id})
                if not action_ok:
                    penalty -= self.invalid_action_penalty
            else:
                penalty -= self.budget_exceeded_penalty
                action_ok = False

        elif self.num_lines < action <= self.num_lines * 2:
            path_idx = action - 1 - self.num_lines
            path = self.mediator.paths[path_idx]
            active_metros = [
                m for m in path.metros if not getattr(m, "is_retiring", False)
            ]
            if len(active_metros) > 1:
                action_ok = self.mediator.apply_action({"type": "remove_train", "path_id": path.id})
                if not action_ok:
                    penalty -= self.invalid_action_penalty
            else:
                penalty -= self.invalid_action_penalty
                action_ok = False

        elif action == 0:
            pass  # noop
        else:
            penalty -= self.invalid_action_penalty
            action_ok = False

        # 2) Advance demand hour slice (demand sampling happens inside step_time)
        self.current_hour_float += self.hour_advance_per_step
        self.current_hour_idx = int(self.current_hour_float) % self.num_hours

        # 3) 물리 엔진 진행 (+ demand-driven passenger spawning)
        self.mediator.step_time(self.dt_ms)

        obs = self._get_obs()

        # Reward: use delta-score (not cumulative score) + waiting penalty.
        score_delta = self.mediator.score - self.last_score
        self.last_score = self.mediator.score

        if self.mediator.passenger_max_wait_time_ms > 0:
            waiting_ratio_sum = float(
                sum(
                    (p.wait_ms / self.mediator.passenger_max_wait_time_ms)
                    for p in self.mediator.passengers
                )
            )
        else:
            waiting_ratio_sum = 0.0

        reward = (
            self.score_reward_weight * float(score_delta)
            - self.reward_waiting_weight * waiting_ratio_sum
            + penalty
        )

        info = {
            "action_ok": action_ok,
            "budget_left": self.max_budget - len(self.mediator.metros),
            "score_delta": score_delta,
            "waiting_ratio_sum": waiting_ratio_sum,
        }
        return obs, reward, self.mediator.is_game_over, False, info