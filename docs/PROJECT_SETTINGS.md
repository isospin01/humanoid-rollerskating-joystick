# Project Settings Snapshot

This file is a plain-language snapshot of the default joystick-skater task currently committed in the repo.

Source of truth:

- `SKATER_Paper_Summary.md`
- `src/mjlab_roller/tasks/skater/skater_env_cfg.py`
- `src/mjlab_roller/tasks/skater/config/g1/env_cfgs.py`
- `src/mjlab_roller/envs/g1_skater_joystick_rl_env.py`
- `src/mjlab_roller/tasks/skater/mdp/rewards.py`
- `src/mjlab_roller/tasks/skater/mdp/observations.py`
- `src/mjlab_roller/tasks/skater/mdp/command.py`
- `src/mjlab_roller/tasks/skater/mdp/terminations.py`
- `src/mjlab_roller/tasks/skater/config/g1/rl_cfg.py`

## 1. Current Default Training Stack

- Task ID: `Mjlab-Roller-Joystick-Flat-Unitree-G1`
- Experiment name: `g1_roller_skater_ppo`
- Robot: Unitree G1 roller-skating setup with 23 controlled joints
- Policy: feedforward actor-critic MLP
- RL algorithm: PPO
- Environment family: custom manager-based MuJoCo RL environment
- Legacy baseline: `Mjlab-Roller-Flat-Unitree-G1`

## 2. Commands

The main command term is `skate`.

Command contract:

- frame: body
- axes: `[vx, vy, wz]`
- joystick mapping:
  - left stick `Y -> vx`
  - left stick `X -> vy`
  - right stick `X -> wz`
- negative `vx` is clamped to `0.0`
- joystick backend: `pygame`
- deadzone: `0.10`
- smoothing: `0.20`

Training starts from the first curriculum stage:

- `vx ∈ [0.0, 0.3]`
- `vy ∈ [-0.05, 0.05]`
- `wz ∈ [-0.25, 0.25]`
- command hold time `4.0-6.0 s`

## 3. Standing Initial State

- root position: `(-0.03, 0.0, 0.80)`
- robot starts in a symmetric standing-on-skates pose
- the standing pose is also the reference posture for joint-deviation rewards

## 4. Simulation

- terrain: flat plane
- physics timestep: `0.005 s`
- decimation: `4`
- effective policy step: `0.02 s`
- policy rate: `50 Hz`
- episode length: `20.0 s`
- viewer target body: `torso_link`
- `nconmax = 128`
- `njmax = 300`
- `ccd_iterations = 50`
- `contact_sensor_maxmatch = 64`

## 5. Sensors

The default joystick task uses:

- `left_skate_contact`
- `right_skate_contact`
- `left_boot_scrape`
- `right_boot_scrape`
- `illegal_contact`

## 6. Observations

### Actor

Single-step actor observation contents:

- command `[vx, vy, wz]`
- base linear velocity
- base angular velocity
- projected gravity
- joint position relative to standing pose
- joint velocity
- previous action
- wheel contact summary
- skate separation

Actor settings:

- history length: `10`
- flattened history width: `920`
- corruption enabled during training
- corruption disabled in play mode

### Critic

Single-step critic observation contents:

- actor observation terms
- left and right skate local linear velocity
- left and right skate local angular velocity
- left and right skate contact forces
- left and right boot scrape forces

Critic width:

- `134`

## 7. Reward Terms

The default task uses 21 active reward terms.

Task rewards:

- linear velocity track: `3.2`
- angular velocity track: `1.2`
- alive reward: `0.15`

Regularization:

- base angular velocity XY: `-0.05`
- controlled joint velocity L2: `-0.001`
- controlled joint acceleration L2: `-2.5e-7`
- action rate L2: `-0.05`
- controlled joint position limits: `-5.0`
- energy consumption: `-2e-5`

Joint and posture terms:

- arms deviation: `-0.4`
- waist deviation: `-2.0`
- ankle roll deviation: `-0.2`
- flat orientation: `-7.0`
- base height error: `-2.0`

Inter-foot and wheel constraints:

- feet too near (`< 0.2 m`): `-1.0`
- feet too far (`> 0.5 m`): `-5.0`
- wheel axial slip: `-0.1`
- wheel air time penalty (`< 4` wheels in contact): `-1.0`

Symmetry and contact:

- leg symmetry: `+0.5`
- arm symmetry: `+0.5`
- undesired contacts: `-1.0`

## 8. Curriculum

The joystick task uses four curriculum stages and scales policy actions with `beta` before applying them to the action manager.

Stage 0, iteration `0-4999`:

- `beta = 0.35`
- `vx ∈ [0.0, 0.3]`
- `vy ∈ [-0.05, 0.05]`
- `wz ∈ [-0.25, 0.25]`
- hold time `4.0-6.0 s`

Stage 1, iteration `5000-14999`:

- `beta = 0.55`
- `vx ∈ [0.0, 0.7]`
- `vy ∈ [-0.10, 0.10]`
- `wz ∈ [-0.50, 0.50]`
- hold time `3.0-5.0 s`

Stage 2, iteration `15000-29999`:

- `beta = 0.80`
- `vx ∈ [0.0, 1.1]`
- `vy ∈ [-0.15, 0.15]`
- `wz ∈ [-0.75, 0.75]`
- hold time `2.0-4.0 s`

Stage 3, iteration `30000+`:

- `beta = 1.00`
- `vx ∈ [0.0, 1.5]`
- `vy ∈ [-0.20, 0.20]`
- `wz ∈ [-1.00, 1.00]`
- hold time `1.5-3.0 s`

## 9. Domain Randomization

Current randomization hooks implement:

- link mass scale `U(0.9, 1.1)`
- torso CoM offset `[-0.01, 0.01] m`
- left and right skate CoM offset `[-0.01, 0.01] m`
- wheel friction axis 0 `U(0.1, 0.8)`
- wheel friction axis 2 `U(0.1, 0.4)`
- actuator stiffness scale `U(0.9, 1.1)`
- actuator damping scale `U(0.9, 1.1)`
- wheel joint damping `U(0.002, 0.005)`

The actuator and wheel damping randomization is implemented as a startup helper that mutates the compiled MuJoCo model globally for the current process.

## 10. PPO Defaults

- actor hidden dims: `(512, 256, 128)`
- critic hidden dims: `(512, 256, 128)`
- `num_steps_per_env = 24`
- `max_iterations = 50000`
- `save_interval = 500`

## 11. Export Metadata

ONNX export now attaches:

- `command_axes=vx,vy,wz`
- `command_frame=body`
- `joystick_mapping=left_y:vx,left_x:vy,right_x:wz`
- `action_beta_max=1.0`
