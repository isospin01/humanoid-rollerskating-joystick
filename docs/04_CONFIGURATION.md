# 04 · Configuration Index

**If you want to change something, this is the page.** One table per concept, each
row pointing to the exact file (and often the exact function) that owns the value.

All paths are relative to repo root.

---

## 1. PPO hyperparameters

File: [`src/mjlab_roller/tasks/skater/config/g1/rl_cfg.py`](../src/mjlab_roller/tasks/skater/config/g1/rl_cfg.py)

| Knob | Default | Function |
|------|---------|----------|
| Actor hidden dims | `(512, 256, 128)` | `unitree_g1_skater_ppo_runner_cfg` |
| Critic hidden dims | `(512, 256, 128)` | ″ |
| Activation | `"elu"` | ″ |
| Initial noise std | `1.0` | ″ |
| Clip ratio | `0.2` | ″ |
| Entropy coef | `0.005` | ″ |
| Value-loss coef | `1.0` | ″ |
| Learning rate | `1e-3` (adaptive) | ″ |
| Desired KL | `0.01` | ″ |
| Discount γ | `0.99` | ″ |
| GAE λ | `0.95` | ″ |
| Max grad norm | `1.0` | ″ |
| Learning epochs | `5` | ″ |
| Mini-batches | `4` | ″ |
| Steps per env per iter | `24` | ″ |
| Max iterations | `50_000` | ″ |
| Checkpoint interval | `500` | ″ |
| **AMP-variant** knobs | see `unitree_g1_skater_amp_ppo_runner_cfg` |  |

---

## 2. Reward weights

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) → `rewards = {...}` block.

| Reward term | Weight | Purpose |
|-------------|--------|---------|
| `linear_velocity_track` | `+3.2` | Exponential tracking of commanded `[vx, vy]` |
| `angular_velocity_track` | `+1.2` | Exponential tracking of commanded `wz` |
| `alive_reward` | `+0.15` | Per-step survival bonus |
| `leg_symmetry` | `+0.5` | Encourage L/R symmetric leg motion |
| `arm_symmetry` | `+0.5` | Encourage L/R symmetric arm swing |
| `base_ang_vel_xy_l2` | `-0.05` | Penalize base roll/pitch rate |
| `controlled_joint_vel_l2` | `-1e-3` | Regularize joint velocities |
| `controlled_joint_acc_l2` | `-2.5e-7` | Regularize joint accelerations |
| `action_rate_l2` | `-0.05` | Smooth the action trajectory |
| `controlled_joint_pos_limits` | `-5.0` | Soft joint-range barrier |
| `energy_consumption` | `-2e-5` | `|τ · q̇|` penalty |
| `arms_deviation` | `-0.4` | Keep arms close to standing pose |
| `waist_deviation` | `-2.0` | Keep waist close to standing pose |
| `ankle_roll_deviation` | `-0.2` | Discourage ankle-roll from neutral |
| `flat_orientation` | `-7.0` | Upright torso |
| `base_height_error` | `-2.0` | Maintain `0.80 m` base height |
| `feet_too_near` | `-1.0` | Min skate separation `0.2 m` |
| `feet_too_far` | `-5.0` | Max skate separation `0.5 m` |
| `wheel_axial_slip` | `-0.1` | Discourage wheel axial (side) slip |
| `wheel_air_time_penalty` | `-1.0` | Penalize <4 wheels in contact |
| `undesired_contacts` | `-1.0` | Any illegal-contact-sensor hit |
| `bad_skate_double_air_penalty` | `-2.0` | Both skates airborne (soft) |
| `excessive_lateral_slip_penalty` | `-2.0` | Lateral slip > threshold (soft) |

Reward implementations: [`src/mjlab_roller/tasks/skater/mdp/rewards.py`](../src/mjlab_roller/tasks/skater/mdp/rewards.py)

---

## 3. Curriculum stages

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) → `curriculum_stages = (...)` tuple.

Default PPO schedule (5 stages, action-limit β ramping 0.35 → 1.0):

| Start iter | β | `vx` range (m/s) | `vy` range (m/s) | `wz` range (rad/s) | Resample (s) |
|-----------:|---|------------------|------------------|--------------------|--------------|
| 0 | 0.35 | `(0.0, 0.3)` | `(-0.05, 0.05)` | `(-0.25, 0.25)` | `4.0–6.0` |
| 5 000 | 0.55 | `(0.0, 0.7)` | `(-0.10, 0.10)` | `(-0.50, 0.50)` | `3.0–5.0` |
| 15 000 | 0.80 | `(0.0, 1.1)` | `(-0.15, 0.15)` | `(-0.75, 0.75)` | `2.0–4.0` |
| 30 000 | 1.00 | `(0.0, 1.5)` | `(-0.20, 0.20)` | `(-1.00, 1.00)` | `1.5–3.0` |

AMP variant stages (β always 1.0): see `unitree_g1_skater_amp_env_cfg` in
[`config/g1/env_cfgs.py`](../src/mjlab_roller/tasks/skater/config/g1/env_cfgs.py).

---

## 4. Observations

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) → `policy_terms`/`critic_terms`.

Policy (actor) observations — history length `10`, flattened:

| Term | Noise | Scale |
|------|-------|-------|
| `command` | — | — |
| `base_lin_vel` | `±0.1` | — |
| `base_ang_vel` | `±0.2` | `0.25` |
| `projected_gravity` | `±0.05` | — |
| `joint_pos` (rel. reference pose) | `±0.01` | — |
| `joint_vel` | `±1.5` | `0.05` |
| `actions` (last) | — | — |
| `wheel_contact_summary` | — | — |
| `skate_separation` | — | — |

Critic (value) adds: per-skate linear/angular velocity, per-skate contact forces,
per-boot scrape forces. Critic sees unnoised observations.

Implementations: [`mdp/observations.py`](../src/mjlab_roller/tasks/skater/mdp/observations.py)

---

## 5. Commands (joystick / sampled)

File: [`src/mjlab_roller/tasks/skater/mdp/command.py`](../src/mjlab_roller/tasks/skater/mdp/command.py) → `SkaterJoystickCommandCfg`.

| Property | Value |
|----------|-------|
| Frame | body |
| Axes | `[vx, vy, wz]` |
| Gamepad mapping | L-stick Y → `vx`, L-stick X → `vy`, R-stick X → `wz` |
| Deadzone | `0.10` |
| Smoothing (EMA) | `0.20` |
| Negative `vx` clamp | `0.0` |

---

## 6. Terminations

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) → `terminations = {...}`.

| Name | Condition |
|------|-----------|
| `time_out` | Episode length exceeded (`20 s`). |
| `fell_over` | Base tilt > 70°. |
| `excessive_boot_scrape` | Boot-frame contact > 0.5 s. |

Implementations: [`mdp/terminations.py`](../src/mjlab_roller/tasks/skater/mdp/terminations.py)

---

## 7. Events / Domain randomization

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) → `events = {...}`.

| Event | On reset | Range |
|-------|----------|-------|
| Joint-pos jitter | always | `±0.01` rad |
| Link mass scale | DR | `×[0.9, 1.1]` per body |
| Torso / skate CoM | DR | `±0.01 m` per axis |
| Wheel static friction | DR | `[0.1, 0.8]` |
| Wheel dynamic friction | DR | `[0.1, 0.4]` |
| Actuator stiffness / damping | always | `×[0.9, 1.1]` |
| Wheel joint damping | always | `[0.002, 0.005]` |

Implementations: [`mdp/events.py`](../src/mjlab_roller/tasks/skater/mdp/events.py)

---

## 8. Simulation & scene

File: [`src/mjlab_roller/tasks/skater/skater_env_cfg.py`](../src/mjlab_roller/tasks/skater/skater_env_cfg.py) (bottom of `make_g1_skater_env_cfg`).

| Knob | Value |
|------|-------|
| Physics timestep | `0.005 s` |
| Decimation | `4` |
| Policy rate | `50 Hz` |
| Episode length | `20 s` |
| Terrain | flat plane |
| `nconmax` / `njmax` | `128` / `1500` |
| MuJoCo iterations | `10` |
| MuJoCo LS iterations | `20` |
| Viewer focus body | `torso_link` |

---

## 9. Robot spec (G1 + skates)

File: [`src/mjlab_roller/assets/robots/roller/g1.py`](../src/mjlab_roller/assets/robots/roller/g1.py)

| Knob | What to change here |
|------|---------------------|
| `CONTROLLED_JOINT_NAMES` (in `core/control_spec.py`) | Set of joints the policy controls |
| `G1_23Dof_ACTION_SCALE` | Per-joint action scaling |
| `STANDING_SKATE_INIT_KEYFRAME` | Reset pose of the robot |
| `STANDING_SKATE_CONTROLLED_JOINT_POS` | Reference pose for `joint_pos_rel_reference` obs and deviation rewards |
| Actuator stiffness/damping | Motor PD gains |

---

## 10. Task registry (how to add a new task)

Registry:       [`src/mjlab_roller/tasks/registry.py`](../src/mjlab_roller/tasks/registry.py)
Bootstrap:      [`src/mjlab_roller/tasks/bootstrap.py`](../src/mjlab_roller/tasks/bootstrap.py)
Register here:  [`src/mjlab_roller/tasks/skater/config/g1/__init__.py`](../src/mjlab_roller/tasks/skater/config/g1/__init__.py)

Recipe:
1. Write a new env-cfg factory in `config/<robot>/env_cfgs.py`.
2. Write an RL-cfg factory in `config/<robot>/rl_cfg.py`.
3. Call `register_task(id=..., env_cls=..., env_cfg=..., rl_cfg=...)` in `config/<robot>/__init__.py`.
4. Bootstrap runs automatically in `cli/train.py` and `cli/play.py`.

---

## 11. Quick "I want to …" lookup

| I want to… | Go to… |
|------------|--------|
| Change a reward weight | § 2 |
| Extend velocity range | § 3 (curriculum last stage) |
| Add a new observation | `mdp/observations.py` + register in § 4 |
| Swap the network size | § 1 (`actor_hidden_dims`) |
| Train longer | § 1 (`max_iterations`) |
| Tighten / loosen termination | § 6 |
| Add a new randomization | § 7 |
| Change robot joints | § 9 |
| Register a new task variant | § 10 |
