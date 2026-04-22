# Smoke Test

This is the minimum runtime checklist for the joystick-conditioned PPO setup.

Goal:

- prove the local Python environment can import the project and `mjlab`
- prove the joystick skater environment can construct and step
- prove a short PPO job starts and finishes

This is not a learning-quality test. It only checks that the runtime path is wired correctly.

## 1. Preconditions

Use the repo root and install dependencies:

```bash
uv sync
```

The project expects Python `>=3.12,<3.14`.

## 2. Basic Import Check

Run:

```bash
uv run python -c "import mjlab; import mjlab_roller; print('imports ok')"
```

Expected result:

- the command prints `imports ok`

## 3. Task Bootstrap Check

Run:

```bash
uv run python -c "from mjlab_roller.tasks.bootstrap import bootstrap_task_registry; bootstrap_task_registry(); from mjlab_roller.tasks.registry import list_tasks; print(list_tasks())"
```

Expected result:

- the output includes `Mjlab-Roller-Joystick-Flat-Unitree-G1`
- the legacy `Mjlab-Roller-Flat-Unitree-G1` task is still present

## 4. Headless Env Smoke

Run:

```bash
MUJOCO_GL=egl uv run python - <<'PY'
import torch
from mjlab_roller.tasks.bootstrap import bootstrap_task_registry
from mjlab_roller.tasks.registry import load_env_cfg, load_env_cls
bootstrap_task_registry()
task_id = "Mjlab-Roller-Joystick-Flat-Unitree-G1"
env_cfg = load_env_cfg(task_id)
env_cfg.scene.num_envs = 1
env = load_env_cls(task_id)(cfg=env_cfg, device="cpu", render_mode=None)
env.reset()
env.step(torch.zeros(env.action_space.shape, device=env.device))
env.close()
print("env smoke ok")
PY
```

What this checks:

- task registration
- MuJoCo model load
- contact sensor setup
- command manager wiring
- observation, reward, and termination managers

## 5. Viewer Smoke With Sampled Commands

Run:

```bash
python -m mjlab_roller.cli.play Mjlab-Roller-Joystick-Flat-Unitree-G1 --agent zero --command-source sampled --num-envs 1 --viewer native
```

Expected result:

- the viewer opens
- the robot appears standing on skates
- the sim runs without immediate crash

## 6. Viewer Smoke With Real Joystick

Run:

```bash
python -m mjlab_roller.cli.play Mjlab-Roller-Joystick-Flat-Unitree-G1 --agent zero --command-source joystick --num-envs 1 --viewer native
```

Expected result:

- the runtime detects the gamepad
- left stick `Y/X` changes forward and lateral command
- right stick `X` changes yaw command

## 7. PPO Smoke Test

If your local GPU stack is compatible with the installed `torch`, run the default command:

```bash
MUJOCO_GL=egl uv run python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 --env.scene.num_envs 8 --agent.max_iterations 2
```

If your GPU driver and `torch` build are mismatched, force CPU mode:

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl uv run python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1 --env.scene.num_envs 8 --agent.max_iterations 2
```

Expected result:

- training starts
- 2 iterations complete
- logs appear under `logs/rsl_rl/g1_roller_skater_ppo`

## 8. Pass Criteria

I would call the repo runtime-ready for joystick PPO only if all of these pass:

1. import check
2. registry check
3. headless env step
4. sampled-command play smoke
5. 2-iteration PPO smoke

That does not guarantee good skating yet. It only means the implementation path is healthy enough to begin tuning.
