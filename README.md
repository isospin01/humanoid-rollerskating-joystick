# `mjlab_roller` — Humanoid Roller-Skating RL

PPO training stack for a **Unitree G1** humanoid on **inline skates**, driven by body-frame
joystick commands `[vx, vy, wz]`. Implements the [SKATER](SKATER_Paper_Summary.md) implicit-gait
reward on top of [mjlab](https://github.com/mujocolab/mjlab) + a vendored
[RSL-RL](https://github.com/leggedrobotics/rsl_rl) PPO backend.

---

## At a Glance

| | |
|---|---|
| **Default task** | `Mjlab-Roller-Joystick-Flat-Unitree-G1` |
| **Legacy baseline** | `Mjlab-Roller-Flat-Unitree-G1` (cycle-based) |
| **Robot** | Unitree G1, 23 controlled joints + passive inline wheels |
| **Policy** | MLP actor–critic `(512, 256, 128)` with ELU |
| **Algorithm** | PPO (GAE, adaptive KL-based LR schedule) |
| **Control rate** | 50 Hz (`dt=0.005s`, decimation=4) |
| **Episode** | 20 s |
| **Experiment name** | `g1_roller_skater_ppo` |

---

## Documentation Map

Start here — every doc is short and topical. Read them in order the first time.

1. [docs/01_QUICKSTART.md](docs/01_QUICKSTART.md) — install, smoke tests, train/play commands
2. [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) — data flow, module responsibilities
3. [docs/03_EXPERIMENTS.md](docs/03_EXPERIMENTS.md) — how to run / track / compare experiments
4. [docs/04_CONFIGURATION.md](docs/04_CONFIGURATION.md) — **the config index — where to change what**
5. [docs/05_DEPLOYMENT.md](docs/05_DEPLOYMENT.md) — ONNX export & on-robot pipeline
6. [SKATER_Paper_Summary.md](SKATER_Paper_Summary.md) — the reward/curriculum design rationale

## Repository Layout

```
mjlab_roller/
├── src/mjlab_roller/
│   ├── cli/         ← train.py · play.py · build_amp_dataset.py · validate_amp_dataset.py
│   ├── envs/        ← MuJoCo env classes  (skater_joystick + legacy roller)
│   ├── tasks/
│   │   ├── skater/  ← DEFAULT joystick task — rewards, obs, commands, curriculum
│   │   └── roller/  ← LEGACY cycle-based baseline (kept for comparison)
│   ├── rl/          ← PPO config, ONNX exporter, vecenv wrapper
│   ├── teleop/      ← pygame gamepad driver
│   ├── assets/      ← MuJoCo XML + G1 robot spec
│   ├── core/        ← project paths, control spec
│   └── data/        ← AMP dataset loader
├── rsl_rl/          ← Vendored RSL-RL (PPO / AMP-PPO backend)
├── dataset/         ← Reference poses + AMP motion clips
├── scripts/         ← Utility scripts (see scripts/README.md)
├── tests/           ← Unit/smoke tests
└── docs/            ← This directory (see doc map above)
```

## Quick Commands

```bash
# install
uv sync

# full training
make train              # or: python -m mjlab_roller.cli.train Mjlab-Roller-Joystick-Flat-Unitree-G1

# play with gamepad
make play-joystick

# play with sampled commands (no gamepad)
make play-sampled

# smoke-test the whole stack
make smoke

# run tests
make test
```

See [docs/03_EXPERIMENTS.md](docs/03_EXPERIMENTS.md) for the full experiment recipe book.

## What's Not in Git

`.gitignore` excludes: `logs/`, `artifacts/`, `export/`, `media/`, `wandb/`, checkpoints,
and the bulky `AMPdatasetgen/` motion-capture pipeline. Regenerate or keep them outside the repo.

## License

Apache-2.0 — see [LICENSE](LICENSE). Contribution guide in [CONTRIBUTING.md](CONTRIBUTING.md).
