# 03 · Experiments

Recipes for running, resuming, visualizing, and comparing training runs.

## Running an experiment

### Default PPO run (the one you'll use 90 % of the time)

```bash
make train
```

Registered task:        `Mjlab-Roller-Joystick-Flat-Unitree-G1`
Experiment name:        `g1_roller_skater_ppo`
Iterations:             `50_000`
Log directory:          `logs/rsl_rl/g1_roller_skater_ppo/<timestamp>/`
Checkpoint interval:    every `500` iterations
Steps per env per iter: `24`

### AMP-augmented run (motion-priors assisted)

```bash
make train-amp
```

Registered task:        `Mjlab-Roller-Joystick-AMP-Flat-Unitree-G1`
Experiment name:        `g1_roller_skater_amp_ppo`
Iterations:             `100_000`
Motion files:           `dataset/rollerskating_amp/`

### Legacy cycle-based baseline

```bash
make train-legacy
```

### CPU smoke (sanity check — 1 minute)

```bash
make train-smoke
```

Uses `num_envs=8`, `max_iterations=2`. Verifies the stack compiles & runs end-to-end.

## Overriding config from the CLI

`train.py` uses [tyro](https://brentyi.github.io/tyro/) — every dataclass field is a flag.
Examples:

```bash
# change env count + iteration budget
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 \
    --env.scene.num_envs 4096 \
    --agent.max_iterations 20000

# disable observation noise (smoother debugging)
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 \
    --env.observations.policy.enable_corruption False

# record periodic videos during training
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 \
    --video --video-interval 2000 --video-length 200
```

## Resuming from a checkpoint

### Local checkpoint

```bash
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 \
    --agent.resume True \
    --agent.load_run <timestamp-dir>   # e.g. 2026-04-15_12-34-56
```

### W&B run

```bash
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 \
    --wandb-run-path <entity>/<project>/<run-id>
```

### Reset critic only (keep actor weights)

Useful when restarting with a different reward shaping:

```bash
python -m mjlab_roller.cli.train <task> --reset-critic True --agent.resume True --agent.load_run ...
```

## Playing / visualizing

```bash
# trained policy + gamepad
make play-joystick

# trained policy + scripted command sweep
make play-sampled

# randomly initialized policy (sanity check env visuals)
python -m mjlab_roller.cli.play <task> --agent zero --command-source sampled --num-envs 1
```

Viewer options (flags on `play.py`):
- `--viewer native`  — MuJoCo native window
- `--viewer viser`   — browser-based Viser viewer

## Recording videos

During training: `--video --video-interval 2000 --video-length 200`.
Standalone recording after training: `python scripts/record_policy_video.py …` (see [scripts/README.md](../scripts/README.md)).

## Tracking & comparison

### TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/
```

Key scalars to watch:
- `Episode/Reward/linear_velocity_track` — should rise early
- `Episode/Reward/angular_velocity_track` — should rise with curriculum stage 2+
- `Train/value_loss`, `Train/surrogate_loss`
- `Train/mean_noise_std` — falls as policy converges
- `Env/curriculum_stage` — confirms staged transitions
- `Env/action_beta` — confirms beta ramp

### W&B

Set `WANDB_MODE=online` and pass `--agent.logger wandb` in the CLI. Offline-mode artifacts
can be synced later with `wandb sync`.

### Helper scripts

See [scripts/README.md](../scripts/README.md):
- `plot_tb_scalar_window.py` — plot a rolling window of a TB scalar
- `count_tb_steps.py` — count iterations in a TB log directory
- `dump_tb_tags.py` — list all available TB tags in a run
- `monitor_curriculum_loop.sh` — watchdog for curriculum transitions

## Experiment hygiene (recommended)

1. **Branch per experiment** — `git switch -c exp/reward-sweep-01`
2. **Name your runs** — set `--agent.experiment_name` to something specific; avoid overwriting the default.
3. **Capture the config** — every run automatically dumps `params/env.yaml` + `params/agent.yaml`.
4. **Record what you're testing** — keep a one-paragraph note in your PR/branch describing the hypothesis.
5. **Compare fairly** — fix `--agent.seed` across runs when comparing reward or hyperparam deltas.

## Multi-GPU training

```bash
python -m mjlab_roller.cli.train <task> --gpu-ids "[0,1,2,3]"
```
or `--gpu-ids all`. Internally uses `torchrun` to shard environments across processes.

## Common issues

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: mjlab` | `uv sync` didn't run cleanly; check the Git pin in `pyproject.toml`. |
| CUDA OOM | Lower `--env.scene.num_envs`; the default targets a 24 GB GPU. |
| Policy always outputs zeros | Check that `bootstrap_task_registry()` was called (it is, inside `train.py`). |
| Joystick not detected | Launch `python scripts/record_demo.py` to verify pygame sees the device. |
