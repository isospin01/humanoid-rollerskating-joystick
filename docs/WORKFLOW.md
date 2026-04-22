# Workflow

This repository now has two task lines:

- default: `Mjlab-Roller-Joystick-Flat-Unitree-G1`
- legacy baseline: `Mjlab-Roller-Flat-Unitree-G1`

The new joystick task is the main path for current work.

## Start Here

Read these files first if you want to understand the joystick training flow:

1. `SKATER_Paper_Summary.md`
2. `src/mjlab_roller/cli/train.py`
3. `src/mjlab_roller/tasks/skater/config/g1/__init__.py`
4. `src/mjlab_roller/tasks/skater/config/g1/env_cfgs.py`
5. `src/mjlab_roller/tasks/skater/skater_env_cfg.py`
6. `src/mjlab_roller/envs/g1_skater_joystick_rl_env.py`
7. `src/mjlab_roller/tasks/skater/mdp/rewards.py`
8. `src/mjlab_roller/tasks/skater/mdp/observations.py`
9. `src/mjlab_roller/tasks/skater/mdp/command.py`

Read the legacy cycle stack only if you want a baseline comparison:

1. `src/mjlab_roller/tasks/roller/config/g1/__init__.py`
2. `src/mjlab_roller/tasks/roller/roller_env_cfg.py`
3. `src/mjlab_roller/envs/g1_roller_rl_env.py`

## Typical Workflow

1. Install dependencies with `uv sync`.
2. Run the registry check and unit tests.
3. Do a dummy play smoke with the joystick task.
4. Run a short PPO smoke job.
5. Scale training up under `logs/rsl_rl/g1_roller_skater_ppo`.
6. Use `play` with `--command-source joystick` for live teleop validation.

## Command Reference

Train:

```bash
python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1
```

Play with sampled commands:

```bash
python -m mjlab_roller.cli.play Mjlab-Roller-Joystick-Flat-Unitree-G1 --agent zero --command-source sampled --num-envs 1
```

Play with a gamepad:

```bash
python -m mjlab_roller.cli.play Mjlab-Roller-Joystick-Flat-Unitree-G1 --agent trained --command-source joystick --num-envs 1
```

Legacy baseline:

```bash
python -m mjlab_roller.cli.train Mjlab-Roller-Flat-Unitree-G1
```

## Package Map

- `src/mjlab_roller/cli`
  Training and play entrypoints.
- `src/mjlab_roller/tasks/skater`
  Default joystick-conditioned task namespace.
- `src/mjlab_roller/tasks/roller`
  Legacy cycle-based baseline namespace.
- `src/mjlab_roller/envs`
  Runtime environment implementations.
- `src/mjlab_roller/teleop`
  Pygame joystick adapter.
- `src/mjlab_roller/assets`
  MuJoCo XML and robot configuration ownership.
- `rsl_rl`
  PPO backend.
