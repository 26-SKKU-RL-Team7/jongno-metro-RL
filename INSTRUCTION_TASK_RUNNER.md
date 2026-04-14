# Task Runner Instruction

This document explains how to add and run different RL problems (tasks) and policies in this project.

## 1) Quick start

- List available tasks:
  - `python src/task_runner.py --list`
- Run one task with a built-in policy:
  - `python src/task_runner.py --task jongno_dispatch --policy random --steps 300`
- Run with another built-in policy:
  - `python src/task_runner.py --task jongno_peak_stress --policy keep_capacity`
- Override task environment arguments for one run:
  - `python src/task_runner.py --task jongno_dispatch --env-overrides "{\"max_budget\": 10, \"reward_waiting_weight\": 0.2}"`

## 2) Current architecture

- Task definitions live in `src/task_catalog.py`
- Unified runner lives in `src/task_runner.py`
- Runner output is JSON (easy to log/parse)

Task execution flow:
1. Pick a task id (`--task`)
2. Build env from task spec (`env_factory`)
3. Load policy (`--policy`)
4. Run rollout for `--steps`
5. Print result JSON

## 3) Add a new problem (task)

Edit `src/task_catalog.py`.

### Step A: add env builder

Create a new builder function that returns your environment:

```python
def _build_my_new_task(overrides: Dict[str, Any]) -> JongnoMetroEnv:
    config = {
        "max_budget": 12,
        "dt_ms": 1000,
        "hour_advance_per_step": 1.2,
        "demand_spawn_scale": 0.007,
        "reward_waiting_weight": 0.3,
    }
    config.update(overrides)
    return JongnoMetroEnv(**config)
```

### Step B: register task in `TASK_SPECS`

```python
"my_new_task": TaskSpec(
    task_id="my_new_task",
    description="Short sentence for this problem definition",
    env_factory=_build_my_new_task,
    default_steps=700,
),
```

### Step C: run it

`python src/task_runner.py --task my_new_task --policy random`

## 4) Add a new policy

You have 2 options.

### Option A: built-in policy

Add a function in `src/task_runner.py`:

```python
def my_policy(env, obs, info) -> int:
    return 0
```

Then register it:

```python
BUILT_IN_POLICIES = {
    "random": random_policy,
    "keep_capacity": keep_capacity_policy,
    "my_policy": my_policy,
}
```

Run:
`python src/task_runner.py --task jongno_dispatch --policy my_policy`

### Option B: external policy plug-in

Create a separate module, for example `src/my_policy_module.py`:

```python
def act(env, obs, info) -> int:
    # return valid discrete action integer
    return 0
```

Run:
`python src/task_runner.py --task jongno_dispatch --policy my_policy_module:act`

## 5) Policy function contract

- Signature: `policy(env, obs, info) -> int`
- Must return one discrete action integer
- Action must be in the environment action space
- `info` is the previous step info (empty at first step)

## 6) Good editing checklist (for contributors)

- When adding a task:
  - Keep task name short and explicit
  - Add one-line description with objective and difficulty
  - Set a realistic `default_steps`
- When adding a policy:
  - Handle edge cases (invalid obs shape, empty lines)
  - Keep fallback action safe (usually `0`, noop)
- Always run tests:
  - `python -m unittest -v`

## 7) Common troubleshooting

- `ValueError: Custom policy must be in module:function format`
  - Fix `--policy` input to `module_name:function_name`
- `ModuleNotFoundError` for custom policy
  - Make sure module path is importable from project root
- Frequent invalid action penalties
  - Clip policy output to valid action range or use env action sampling as fallback
