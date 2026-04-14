[![Demo](https://i.imgur.com/xpUow2f.png)](https://youtu.be/W5fCgqlECeI)

# python_mini_metro
This repo uses `pygame-ce` to implement Mini Metro, a fun 2D strategic game where you try to optimize the max number of passengers your metro system can handle. Both human and program inputs are supported. One of the purposes of this implementation is to enable reinforcement learning agents to be trained on it.

# Installation
`pip install -r requirements.txt`

# How to run
## To play manually
* If you are running for the first time, install the requirements using `pip install -r requirements.txt`
* Activate the virtual environment by running `conda activate py313`
* Run `python src/main.py`
* Hold down the mouse left button on a station and drag onto other stations to create a path for the metro.
* Press SPACE to pause / unpause the game.
* Press `1`, `2`, or `3` to set game speed to 1x, 2x, or 4x.
* View the score on the top left corner of the screen.
* The number of grey circles on bottom of the screen is the number of available metro lines left.
* Click on the colored circle at the bottom to cancel an established line.
* Click on the empty circles at the bottom to buy new lines with scores.

## To play programmatically
Use the Gym-like environment in `src/env.py`:

```python
from env import MiniMetroEnv

env = MiniMetroEnv(dt_ms=16)
obs = env.reset(seed=42)
obs, reward, done, info = env.step(
    {"type": "create_path", "stations": [0, 1, 2], "loop": False}
)
obs, reward, done, info = env.step({"type": "remove_path", "path_index": 0})
```

### API
- `MiniMetroEnv(dt_ms: int | None = None)`
  - `dt_ms` is the default simulated milliseconds advanced after each `step(...)`.
  - If `dt_ms=None`, time only advances when you pass `dt_ms` to `step(...)`.
- `reset(seed: int | None = None) -> observation`
  - Resets the game and returns the initial observation.
  - If `seed` is provided, Python and NumPy RNG are seeded for deterministic runs.
- `step(action: dict | None = None, dt_ms: int | None = None) -> (observation, reward, done, info)`
  - Applies one action, optionally advances time, then returns:
    - `observation`: latest state
    - `reward` (`int`): score delta since previous step
    - `done` (`bool`): `True` when game is over
    - `info` (`dict`): currently contains `{"action_ok": bool}`

### Valid `action` inputs
- `None`
  - Treated as `{"type": "noop"}`.
- `{"type": "noop"}`
  - No direct game command; only time progression happens (if `dt_ms` resolves to an integer).
- `{"type": "create_path", "stations": [i0, i1, ...], "loop": bool}`
  - Required:
    - `stations`: list of station indices (`int`) with length `>= 2`
    - all indices must satisfy `0 <= idx < len(observation["structured"]["stations"])`
  - Optional:
    - `loop` (default `False`): when `True`, creates a loop that ends at the first station.
  - Fails (`action_ok=False`) if inputs are invalid or if no unlocked line is available.
- `{"type": "remove_path", "path_index": k}`
  - Removes an existing path by index.
  - Valid only when `0 <= k < len(observation["structured"]["paths"])`.
- `{"type": "remove_path", "path_id": "..."}` 
  - Removes an existing path by path id string from `observation["structured"]["paths"][*]["id"]`.
- `{"type": "buy_line"}`
  - Buys the next locked line if affordable.
  - Price follows configured incremental unlock costs (derived from `path_unlock_milestones`).
- `{"type": "buy_line", "path_index": k}`
  - Attempts to buy a specific locked line button index.
  - Must be the next purchasable locked index (sequential purchase rule); otherwise fails.
  - `path_index` must be an integer in `[0, num_paths - 1]`.
- `{"type": "pause"}`
  - Pauses simulation updates.
- `{"type": "resume"}`
  - Resumes simulation updates.

Any unknown `type`, or malformed action payload, returns `info["action_ok"] == False`.

### `step(..., dt_ms=...)` behavior
- `dt_ms` argument to `step(...)` overrides constructor `dt_ms` for that call.
- If effective `dt_ms` is an integer, simulation advances by that many milliseconds.
- If effective `dt_ms` is `None`, action is applied but time is not advanced.

### Observation shape
`observation` is:
- `observation["structured"]`: Python dict/list representation
  - includes `stations`, `paths`, `metros`, `passengers`, `score`, `time_ms`, `steps`, `is_paused`, `is_game_over`, and ID-to-index maps in `index`.
- `observation["arrays"]`: NumPy-friendly arrays/lists
  - includes station positions/types/counts, path station-index sequences, metro positions/path indices, passenger destination types and locations.

# Testing
`python -m unittest -v`

# Task-based experiment runner
You can run different problem definitions (tasks) from one entrypoint:

- List tasks:
  - `python src/task_runner.py --list`
- Run a specific task:
  - `python src/task_runner.py --task jongno_dispatch --policy random --steps 300`
- Use a different built-in policy:
  - `python src/task_runner.py --task jongno_peak_stress --policy keep_capacity`
- Override environment settings for a run:
  - `python src/task_runner.py --task jongno_dispatch --env-overrides "{\"max_budget\": 10, \"reward_waiting_weight\": 0.2}"`
- Plug in your own policy code:
  - `python src/task_runner.py --task jongno_dispatch --policy my_policy_module:act`
  - `act(env, obs, info) -> int` should return a valid discrete action.

For step-by-step contributor instructions, see `INSTRUCTION_TASK_RUNNER.md`.
For Korean project overview and MDP change guide, see `PROJECT_GUIDE_KR.md`.

# Train and visualize (lightweight RL)
You can train a lightweight masked policy (NumPy REINFORCE) and then visualize
passenger movement with the trained policy:

- Train:
  - `python src/train_jongno_policy.py --task jongno_dispatch --episodes 200 --max-steps 400 --output artifacts/jongno_policy.npz`
- Visualize:
  - `python src/visualize_jongno_policy.py --task jongno_dispatch --model artifacts/jongno_policy.npz --model-type linear --max-steps 600`

Tips:
- Use `--env-overrides` to change run settings, e.g.
  - `python src/train_jongno_policy.py --env-overrides "{\"dt_ms\": 500, \"max_budget\": 12}"`
- Visualization keeps the original simulator rendering (`Mediator.render`) so you
  can inspect passenger boarding, transfer, and waiting behavior directly.

## GNN-lite RL training and comparison
You can also train a graph-based lightweight policy and compare it against the
linear baseline:

- Train GNN-lite:
  - `python src/train_jongno_gnn_lite.py --task jongno_dispatch --episodes 200 --max-steps 400 --output artifacts/jongno_gnn_lite.npz`
- Visualize GNN-lite:
  - `python src/visualize_jongno_policy.py --task jongno_dispatch --model artifacts/jongno_gnn_lite.npz --model-type gnn --max-steps 600`
- Train + evaluate linear vs GNN-lite:
  - `python src/compare_jongno_policies.py --task jongno_dispatch --train-episodes 120 --eval-episodes 8`

`compare_jongno_policies.py` prints JSON including mean reward/score for both
policies and the delta (`gnn_lite - linear`).

# Acknowledgements
This project is forked from [python_mini_metro](https://github.com/yanfengliu/python_mini_metro) by [Yanfeng Liu](https://github.com/yanfengliu).
