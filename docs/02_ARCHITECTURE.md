# 02 · Architecture

How the pieces fit together. If you're going to touch the code, read this first.

## Big Picture

```
                  ┌──────────────────────────────────────────────┐
                  │               CLI (entry)                    │
                  │   cli/train.py        cli/play.py            │
                  └───────────────┬──────────────────────────────┘
                                  │ task_id
                                  ▼
                  ┌──────────────────────────────────────────────┐
                  │          Task Registry                       │
                  │  tasks/registry.py    tasks/bootstrap.py     │
                  └───────────────┬──────────────────────────────┘
                                  │ loads env_cfg, rl_cfg
                                  ▼
          ┌──────────────────────────────────────────────────────┐
          │                 RL Environment                       │
          │   envs/g1_skater_joystick_rl_env.py                  │
          │                                                      │
          │   ┌─ obs manager ─┐ ┌─ reward manager ─┐             │
          │   │ observations.py│ │  rewards.py      │             │
          │   └────────────────┘ └──────────────────┘             │
          │   ┌─ command mgr ─┐ ┌─ termination mgr ┐             │
          │   │ command.py     │ │ terminations.py  │             │
          │   └────────────────┘ └──────────────────┘             │
          │   ┌─ event manager ┐                                  │
          │   │ events.py       │   MuJoCo sim (mjlab)            │
          │   └─────────────────┘                                 │
          └──────────────────────────┬───────────────────────────┘
                                     │ VecEnv interface
                                     ▼
          ┌──────────────────────────────────────────────────────┐
          │              PPO Training Backend                    │
          │              (vendored rsl_rl/)                      │
          │   OnPolicyRunner → PPO → ActorCritic MLP             │
          └──────────────────────────┬───────────────────────────┘
                                     │ checkpoints
                                     ▼
                             logs/rsl_rl/<exp>/
```

## Control-Loop Data Flow (single env step)

```
 joystick / curriculum
        │ [vx, vy, wz]
        ▼
   ┌────────────┐
   │ Observation│  command, base_lin_vel, base_ang_vel, projected_gravity,
   │  Manager   │  joint_pos_rel_ref, joint_vel, last_action,
   │            │  wheel_contact_summary, skate_separation
   └─────┬──────┘
         ▼
   ┌────────────┐
   │ Actor MLP  │  (512, 256, 128) ELU  →  joint position targets (23)
   └─────┬──────┘
         │ × beta  (curriculum action-limit scaling)
         ▼
   ┌────────────┐
   │ MuJoCo sim │  physics @ dt=0.005s × 4 (decimation) = 50 Hz policy rate
   └─────┬──────┘
         ▼
   ┌────────────┐
   │  Reward    │  velocity-tracking + alive +
   │  Manager   │  symmetry/energy/constraint penalties (20+ terms)
   └─────┬──────┘
         ▼
   ┌────────────┐
   │Termination │  fallen, joint-limit, illegal-contact, timeout
   │  Manager   │
   └────────────┘
```

## Module Responsibilities

### Entry layer — `src/mjlab_roller/cli/`

| File | Role |
|------|------|
| `train.py` | PPO training entrypoint. Assembles env + runner, handles resume, video, multi-GPU. |
| `play.py` | Policy rollout / teleop. `--agent {trained,zero}` × `--command-source {joystick,sampled}`. |
| `build_amp_dataset.py` | Build AMP motion-clip datasets from reference data. |
| `validate_amp_dataset.py` | Sanity-check AMP dataset shape / timing. |

### Task definition — `src/mjlab_roller/tasks/skater/` (default)

| File | Role |
|------|------|
| `skater_env_cfg.py` | **Master env config factory.** Assembles all managers (obs / reward / command / termination / event). Start here. |
| `config/g1/env_cfgs.py` | G1-specific env variants (PPO, AMP). Defines **curriculum stages**. |
| `config/g1/rl_cfg.py` | **PPO hyperparameters** — actor/critic dims, LR, clip, epochs, iterations. |
| `config/g1/__init__.py` | Registers task IDs with the registry. |
| `mdp/rewards.py` | All 20+ reward functions (velocity tracking, symmetry, energy, constraints). |
| `mdp/observations.py` | Observation term implementations. |
| `mdp/command.py` | `SkaterJoystickCommandCfg` — joystick backend + sampled commands. |
| `mdp/terminations.py` | Episode-end conditions. |
| `mdp/events.py` | Reset / domain-randomization events. |

### Environment — `src/mjlab_roller/envs/`

| File | Role |
|------|------|
| `g1_skater_joystick_rl_env.py` | Main env class. Manages sim, curriculum stage transitions, contact sensors, beta scheduling. |
| `g1_roller_rl_env.py` | Legacy cycle-based env (kept for comparison). |

### RL backend — `src/mjlab_roller/rl/` and `rsl_rl/`

| File | Role |
|------|------|
| `rl/config.py` | Dataclasses mirroring the RSL-RL runner/algo/policy configs. |
| `rl/exporter_utils.py` | ONNX export — attaches obs-normalization & metadata. |
| `rl/vecenv_wrapper.py` | Adapts mjlab env to RSL-RL VecEnv interface. |
| `rsl_rl/algorithms/ppo.py` | PPO update step (GAE, clipping, value loss). |
| `rsl_rl/modules/actor_critic.py` | MLP actor-critic network. |
| `rsl_rl/runners/on_policy_runner.py` | Rollout + update loop, checkpointing. |

### Assets & robot spec — `src/mjlab_roller/assets/robots/roller/`

| File | Role |
|------|------|
| `g1.py` | **Joint names, actuator gains, action scale, initial keyframes.** Change here if you add/remove joints. |
| `*.xml` | MuJoCo model (G1 + skates). |

### Teleop — `src/mjlab_roller/teleop/pygame_joystick.py`

Pygame gamepad driver. Axis mapping:
`left-Y → vx`, `left-X → vy`, `right-X → wz`. Deadzone `0.10`, smoothing `0.20`.

## Task Registry

Tasks are registered by ID in `tasks/registry.py`. Bootstrap discovers them by importing
every `tasks/*/config/*/__init__.py`. To **add a new task variant**:

1. Create `tasks/<task>/config/<robot>/env_cfgs.py` with an env-cfg factory.
2. Create `rl_cfg.py` with a runner-cfg factory.
3. Register both in `config/<robot>/__init__.py` via `register_task(...)`.
4. The bootstrap will pick it up automatically.

See [04_CONFIGURATION.md](04_CONFIGURATION.md) for exact knobs.
