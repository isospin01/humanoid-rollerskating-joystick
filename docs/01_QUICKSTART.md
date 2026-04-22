# 01 · Quickstart

Get from zero to a running training job in under 10 minutes.

## Prerequisites

- **OS:** Linux (recommended). macOS/Windows for CPU smoke tests only.
- **GPU:** CUDA-compatible GPU recommended for full training; CPU is fine for smokes.
- **Python:** 3.12 or 3.13
- **Package manager:** [uv](https://docs.astral.sh/uv/) (strongly recommended).

## Install

```bash
git clone <YOUR_FORK_OR_UPSTREAM_URL>
cd humanoid-rollerskating-skater-joystick
uv sync
```

`uv sync` pulls `mjlab` from Git (pinned in `pyproject.toml`) and installs the vendored
`rsl_rl` from `./rsl_rl/`.

## Verify the install

Three checks, increasing in scope:

### 1. Task registry

```bash
uv run python -c "from mjlab_roller.tasks.bootstrap import bootstrap_task_registry; \
  bootstrap_task_registry(); \
  from mjlab_roller.tasks.registry import list_tasks; print(list_tasks())"
```
Expected: a list containing `Mjlab-Roller-Joystick-Flat-Unitree-G1` and `Mjlab-Roller-Flat-Unitree-G1`.

### 2. Unit tests

```bash
make test
```

### 3. Environment smoke (headless, CPU)

```bash
make smoke
```

Runs one reset + one step on the joystick env with `num_envs=1`. Takes a few seconds.

## Your first training job

### Full training (long)

```bash
make train
```
equivalent to:
```bash
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1
```

Logs to `logs/rsl_rl/g1_roller_skater_ppo/<timestamp>/`.
Use TensorBoard to monitor:
```bash
tensorboard --logdir logs/rsl_rl/g1_roller_skater_ppo
```

### Short PPO smoke (CPU, ~1 min)

```bash
make train-smoke
```

## Play a trained policy

### With a USB gamepad

```bash
make play-joystick
```

### With sampled commands (no gamepad)

```bash
make play-sampled
```

See [03_EXPERIMENTS.md](03_EXPERIMENTS.md) for loading checkpoints, resuming, and video capture.

## Next steps

- To understand the system → [02_ARCHITECTURE.md](02_ARCHITECTURE.md)
- To run experiments → [03_EXPERIMENTS.md](03_EXPERIMENTS.md)
- To change configs → [04_CONFIGURATION.md](04_CONFIGURATION.md)
