# Documentation Index

Read in order the first time. After that, each doc stands alone.

| # | Doc | What it covers | When to open it |
|---|-----|----------------|-----------------|
| 1 | [01_QUICKSTART.md](01_QUICKSTART.md) | Install, smoke tests, the 4 commands you'll use daily | First time setting up, or onboarding someone |
| 2 | [02_ARCHITECTURE.md](02_ARCHITECTURE.md) | Data flow (obs → policy → action → reward), module responsibilities | You want to understand *how* the system works |
| 3 | [03_EXPERIMENTS.md](03_EXPERIMENTS.md) | How to run, resume, video-record, and compare experiments | Before starting or reviewing any training run |
| 4 | [04_CONFIGURATION.md](04_CONFIGURATION.md) | **Config index** — one table maps every knob to a file + line | You want to change a reward weight, curriculum stage, or hyperparam |
| 5 | [05_DEPLOYMENT.md](05_DEPLOYMENT.md) | ONNX export & on-robot deployment pipeline | You're shipping a trained policy to hardware |

Supporting documents (historical / reference):

- [../SKATER_Paper_Summary.md](../SKATER_Paper_Summary.md) — full paper-level design rationale for the reward & curriculum
- [PROJECT_SETTINGS.md](PROJECT_SETTINGS.md) — plain-language snapshot of the default task settings
- [WORKFLOW.md](WORKFLOW.md) — original workflow notes
- [SMOKE_TEST.md](SMOKE_TEST.md) — detailed smoke-test procedures
- [DEPLOYMENT_PIPELINE.md](DEPLOYMENT_PIPELINE.md) · [DEPLOYMENT_PIPELINE_CN.md](DEPLOYMENT_PIPELINE_CN.md) — deprecated, superseded by `05_DEPLOYMENT.md`

## Navigating the code

- For the **default joystick task**, everything lives under `src/mjlab_roller/tasks/skater/`.
- For **environment dynamics**, see `src/mjlab_roller/envs/g1_skater_joystick_rl_env.py`.
- For **CLI entrypoints**, see `src/mjlab_roller/cli/`.
- For the **PPO backend**, see the vendored `rsl_rl/` directory.
