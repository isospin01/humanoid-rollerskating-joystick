# 05 · Deployment

How to ship a trained policy to the physical Unitree G1.

> **Note.** This file is the entrypoint — it summarizes the pipeline and points to the
> full deep-dive in [`DEPLOYMENT_PIPELINE.md`](DEPLOYMENT_PIPELINE.md) (English) or
> [`DEPLOYMENT_PIPELINE_CN.md`](DEPLOYMENT_PIPELINE_CN.md) (中文). This repo contains
> the **export** half of the pipeline; the on-robot Python stack lives in
> [`scripts/deploy/`](../scripts/deploy/).

## Pipeline at a glance

```
    Training (sim)                Export (ONNX)                 On-robot
  ───────────────── ──▶  ───────────────────────── ──▶  ─────────────────
   rsl_rl PPO                 exporter_utils.py            scripts/deploy/
   → checkpoints              → policy.onnx                → verify_pipeline
                              → obs_norm stats             → inference loop
                              → joint-name metadata        → Unitree SDK2
```

## 1 · Export a checkpoint to ONNX

The exporter embeds:
- observation normalization stats (running mean/var)
- joint ordering
- action scaling
- obs-term names

Code: [`src/mjlab_roller/rl/exporter_utils.py`](../src/mjlab_roller/rl/exporter_utils.py)
and [`src/mjlab_roller/tasks/roller/rl/exporter.py`](../src/mjlab_roller/tasks/roller/rl/exporter.py).

Typical invocation (inside `play.py` / training): an ONNX file is written beside the `.pt` checkpoint under `logs/rsl_rl/<exp>/<run>/exported/policy.onnx`.

## 2 · Key numbers (for the inference loop)

| | |
|--|--|
| Control rate | **50 Hz** (`dt=0.02 s`) |
| Actions | joint-position targets (23 dim), offset by `default_joint_pos`, scaled by `G1_23Dof_ACTION_SCALE × beta` |
| Observations | 9 policy terms, concatenated with history-length `10` |
| Normalization | running mean/var, baked into ONNX |

## 3 · On-robot code

In [`scripts/deploy/`](../scripts/deploy/):

| File | Role |
|------|------|
| `config.py` | Joint order, action scales, safety limits. |
| `observation_builder.py` | Reconstructs the policy observation from SDK2 state reads. |
| `safety.py` | E-stop, joint-limit clamps, torque clamps. |
| `verify_pipeline.py` | Offline sanity check: runs ONNX against a logged observation trace. |

## 4 · Validation stages (before you turn on motors)

1. **ONNX correctness:** `verify_pipeline.py` against a logged sim obs.
2. **Dry run on stand:** robot suspended, inference loop active, motors disabled.
3. **Low-torque gains:** motors active, scaled PD gains, operator-held tether.
4. **Full-gain stand:** verify standing stability before commanding velocities.
5. **Low-speed teleop:** small `vx` commands only, skate surface with barriers.

## 5 · Safety defaults

Hard-wired in `scripts/deploy/safety.py`:
- Per-joint position clamps (± margin around training range).
- Torque saturation.
- IMU-based tilt E-stop.
- Heartbeat watchdog (policy thread → SDK thread).

## 6 · Risks & mitigations

See [DEPLOYMENT_PIPELINE.md § 13](DEPLOYMENT_PIPELINE.md) for the full list. Top three:
1. **Sim-to-real wheel friction gap** — address with domain randomization (already in § 7 of [04_CONFIGURATION.md](04_CONFIGURATION.md)).
2. **Latency budget** — SDK2 round-trip must stay under one policy step (20 ms).
3. **IMU orientation convention** — the `projected_gravity` obs is the single most common source of sign bugs.
