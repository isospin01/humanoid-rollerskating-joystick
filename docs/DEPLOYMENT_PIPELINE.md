# Sim-to-Real Deployment Pipeline for G1 Roller-Skating Policy

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Policy Summary](#2-policy-summary)
3. [Observation Space: Sim vs Real Mapping](#3-observation-space-sim-vs-real-mapping)
4. [Action Space and Actuation](#4-action-space-and-actuation)
5. [Observation Normalization](#5-observation-normalization)
6. [Policy Export](#6-policy-export)
7. [Hardware Prerequisites](#7-hardware-prerequisites)
8. [Deployment Software Stack](#8-deployment-software-stack)
9. [Inference Loop Design](#9-inference-loop-design)
10. [Safety System](#10-safety-system)
11. [System Identification Checklist](#11-system-identification-checklist)
12. [Validation Stages](#12-validation-stages)
13. [Known Risks and Mitigations](#13-known-risks-and-mitigations)
14. [File Layout](#14-file-layout)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   HOST COMPUTER (GPU)                        │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Joystick /  │───▶│  Observation  │───▶│  ONNX/PyTorch  │  │
│  │  Gamepad     │    │  Builder     │    │  Policy (50Hz) │  │
│  └─────────────┘    └──────────────┘    └───────┬────────┘  │
│                            ▲                     │           │
│                            │                     ▼           │
│                     ┌──────┴───────┐    ┌────────────────┐  │
│                     │  State       │    │  Action Post-  │  │
│                     │  Receiver    │    │  Processor     │  │
│                     └──────┬───────┘    └───────┬────────┘  │
│                            │                     │           │
└────────────────────────────┼─────────────────────┼───────────┘
                             │  Ethernet           │
                             │  192.168.123.x      │
                       ┌─────┴─────────────────────┴─────┐
                       │        UNITREE G1 (EDU)         │
                       │                                  │
                       │  ┌────────────────────────────┐ │
                       │  │  Unitree SDK2 Low-Level    │ │
                       │  │  • Motor state readout     │ │
                       │  │  • PD position commands    │ │
                       │  │  • IMU data readout        │ │
                       │  └────────────────────────────┘ │
                       │  ┌────────────────────────────┐ │
                       │  │  Physical Robot             │ │
                       │  │  23 actuated DoF + skates   │ │
                       │  └────────────────────────────┘ │
                       └──────────────────────────────────┘
```

The policy runs at **50 Hz** (matching the sim control rate: `timestep=0.005s × decimation=4 = 0.02s`).
Communication uses Unitree SDK2 Python over CycloneDDS on ethernet.

---

## 2. Policy Summary

| Property | Value |
|---|---|
| Algorithm | PPO (or AMP-PPO variant) |
| Network | MLP `512 → 256 → 128` (ELU activations) |
| Obs normalization | EmpiricalNormalization (running mean/var) |
| Actor input dim | **920** (92 per frame × 10-frame history) |
| Actor output dim | **23** (joint position offsets) |
| Control type | Joint position targets via PD controller |
| Control frequency | **50 Hz** |

---

## 3. Observation Space: Sim vs Real Mapping

The policy observation is a **92-dimensional** vector per frame, with **10 frames** of history
(oldest first), flattened into a **920-D** input.

### Per-Frame Observation Breakdown (92 dims)

| # | Term | Dim | Sim Source | Real Source | Scale | Noise (training) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `command` | **3** | `[vx_cmd, vy_cmd, wz_cmd]` | Joystick / gamepad | 1.0 | None | Body-frame velocity commands |
| 2 | `base_lin_vel` | **3** | `root_link_lin_vel_b` | **State estimator** (see §3.1) | 1.0 | U(-0.1, 0.1) | Body-frame linear velocity |
| 3 | `base_ang_vel` | **3** | `root_link_ang_vel_b` | IMU gyroscope | **0.25** | U(-0.2, 0.2) | Body-frame angular velocity |
| 4 | `projected_gravity` | **3** | `projected_gravity` | IMU orientation → rotate [0,0,-1] | 1.0 | U(-0.05, 0.05) | Gravity in body frame |
| 5 | `joint_pos` | **23** | `joint_pos - standing_ref` | Motor encoders − standing_ref | 1.0 | U(-0.01, 0.01) | Relative to standing pose |
| 6 | `joint_vel` | **23** | `joint_vel` | Motor encoders (diff or hw velocity) | **0.05** | U(-1.5, 1.5) | Scaled ×0.05 |
| 7 | `actions` | **23** | `last_action` | Previous action buffer | 1.0 | None | Previous policy output |
| 8 | `wheel_contact` | **8** | Contact sensor filter | **Estimated** (see §3.2) | 1.0 | None | Per-wheel binary contact |
| 9 | `skate_separation` | **3** | Site position diff | **FK computation** (see §3.3) | 1.0 | None | World-frame vector L→R skate |
| | **Total** | **92** | | | | | |

### 3.1 Base Linear Velocity (Critical)

**Sim:** Ground-truth body-frame linear velocity is directly available.

**Real:** The G1's onboard IMU provides angular velocity and linear acceleration, **not** linear
velocity. A state estimator is required. Options ranked by reliability:

1. **IMU + Leg Kinematics Estimator (Recommended)**
   - Use the approach from the Unitree SDK's built-in state estimator if available on EDU firmware.
   - Alternatively, implement a complementary filter or EKF fusing:
     - IMU accelerometer (integrated, drift-corrected)
     - Forward kinematics of the legs (wheel contact points provide constraint)
   - This is the standard approach used by ETH RSL (ANYmal), Agility Robotics, and the SKATER paper's deployment.

2. **External Motion Capture (development/validation only)**
   - OptiTrack / Vicon → body-frame velocity via finite differences of position.
   - Useful for initial validation, not final deployment.

3. **Train with noisy/delayed velocity (robustness fallback)**
   - The training already applies U(-0.1, 0.1) noise to this channel.
   - Consider increasing this noise range during training if the real estimator is noisy.

### 3.2 Wheel Contact Flags

**Sim:** Contact sensor detects force > 1N per wheel, with a 2-step filter (OR of current + previous step).

**Real options:**
1. **Motor current thresholding** — Wheel joints are passive (no motor), so this doesn't apply directly. Instead:
2. **Force/pressure sensors under skate frame** — If available, threshold on vertical force per wheel.
3. **Leg kinematics + gravity** — Estimate ground reaction force from joint torques via inverse dynamics. Threshold to determine contact.
4. **Conservative default** — Set all 8 contacts to `1.0` (all wheels on ground). The policy was trained with a penalty for wheels leaving the ground, so it expects most wheels to be in contact most of the time. This is a reasonable starting approximation.

**Recommendation:** Start with option 4 (all contacts = 1.0) for initial deployment. The policy is robust to this since it was trained to keep wheels on the ground. Refine later with force estimation.

### 3.3 Skate Separation Vector

**Sim:** Computed from MuJoCo site positions of front/rear markers on each skate.

**Real:** Compute via forward kinematics from encoder readings. The kinematic chain is:
```
pelvis → hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll → skate
```
Use the MJCF model's body transforms and the measured joint positions to compute FK for
`left_skate_front_marker`, `left_skate_rear_marker`, `right_skate_front_marker`,
`right_skate_rear_marker`. The separation vector is:
```
skate_separation = mean(right_markers) - mean(left_markers)
```
This is expressed in world frame. Use the IMU orientation to rotate from body to world frame.

---

## 4. Action Space and Actuation

### 4.1 Action Semantics

The policy outputs **23 continuous values** in approximately [-1, 1]. These are converted to
joint position targets as follows:

```
target_position[i] = default_pos[i] + action[i] × action_scale[i] × beta
```

Where:
- `default_pos`: Standing reference pose (from `STANDING_SKATE_CONTROLLED_JOINT_POS`)
- `action_scale`: Per-joint scaling derived from `0.25 × effort_limit / stiffness`
- `beta`: Curriculum parameter, **1.0** at final training stage (full deployment)

### 4.2 Joint Order (must match exactly between sim and real)

```
Index  Joint Name                    Actuator Class   Effort (Nm)
─────  ────────────────────────────  ──────────────   ───────────
 0     left_hip_pitch_joint          7520_14          88.0
 1     left_hip_roll_joint           7520_22          139.0
 2     left_hip_yaw_joint            7520_14          88.0
 3     left_knee_joint               7520_22          139.0
 4     left_ankle_pitch_joint        ANKLE(2×5020)    50.0
 5     left_ankle_roll_joint         ANKLE(2×5020)    50.0
 6     right_hip_pitch_joint         7520_14          88.0
 7     right_hip_roll_joint          7520_22          139.0
 8     right_hip_yaw_joint           7520_14          88.0
 9     right_knee_joint              7520_22          139.0
10     right_ankle_pitch_joint       ANKLE(2×5020)    50.0
11     right_ankle_roll_joint        ANKLE(2×5020)    50.0
12     waist_yaw_joint               7520_14          88.0
13     waist_roll_joint              WAIST(2×5020)    50.0
14     waist_pitch_joint             WAIST(2×5020)    50.0
15     left_shoulder_pitch_joint     5020             25.0
16     left_shoulder_roll_joint      5020             25.0
17     left_shoulder_yaw_joint       5020             25.0
18     left_elbow_joint              5020             25.0
19     right_shoulder_pitch_joint    5020             25.0
20     right_shoulder_roll_joint     5020             25.0
21     right_shoulder_yaw_joint      5020             25.0
22     right_elbow_joint             5020             25.0
```

### 4.3 Standing Reference Pose (radians)

```python
STANDING_SKATE_CONTROLLED_JOINT_POS = (
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,       # left leg
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,       # right leg
   0.0, 0.0, 0.05,                           # waist
  -0.10, 0.45, -0.20, 1.10,                 # left arm
  -0.10, -0.45, 0.20, 1.10,                 # right arm
)
```

### 4.4 PD Gains for Real Robot

The sim uses position-mode actuators with stiffness/damping derived from motor physics.
The **exact same PD gains** must be used on the real robot:

```python
# Per actuator class:
# natural_freq = 10 × 2π ≈ 62.83 rad/s
# damping_ratio = 2.0

# Class 5020 (arms):      Kp ≈ ARMATURE_5020 × ω²    Kd ≈ 2 × 2.0 × ARMATURE_5020 × ω
# Class 7520_14 (hips):   Kp ≈ ARMATURE_7520_14 × ω²  ...
# Class 7520_22 (knees):  Kp ≈ ARMATURE_7520_22 × ω²  ...
# Class ANKLE:            Kp = 2 × STIFFNESS_5020      Kd = 2 × DAMPING_5020
# Class WAIST:            Kp = 2 × STIFFNESS_5020      Kd = 2 × DAMPING_5020
```

These values are exported as ONNX metadata (`joint_stiffness`, `joint_damping`) and must
be loaded by the deployment runtime. Send position targets to the Unitree SDK's low-level
motor interface with these Kp/Kd values.

### 4.5 Critical: Joint Mapping to Unitree Motor IDs

The Unitree G1 SDK uses motor IDs (0-28 for the full robot). You **must** create a verified
mapping from the 23 policy joint names to the correct Unitree motor IDs. This mapping depends
on your specific G1 EDU firmware version and configuration.

**Verification procedure:**
1. Command each joint one at a time with a small offset (+0.05 rad) from the standing pose.
2. Visually confirm the correct physical joint moves in the expected direction.
3. Verify sign convention matches (positive direction in sim = positive direction on real).

---

## 5. Observation Normalization

The policy uses `EmpiricalNormalization` which normalizes observations by running statistics:

```
obs_normalized = (obs_raw - mean) / (std + eps)
```

Where `eps = 0.01` and `mean`/`std` are accumulated during training.

### Export Requirements

The normalizer state is saved inside the PyTorch checkpoint under:
```
model_state_dict.actor_obs_normalizer._mean   # shape: [1, 920]
model_state_dict.actor_obs_normalizer._var    # shape: [1, 920]
model_state_dict.actor_obs_normalizer._std    # shape: [1, 920]
model_state_dict.actor_obs_normalizer.count   # scalar
```

**For ONNX export:** The existing `_OnnxPolicyExporter` bakes normalization into the ONNX
graph, so the exported ONNX model accepts raw observations and handles normalization internally.

**For PyTorch deployment:** Load the full checkpoint and use `policy.act_inference(obs_dict)`
which calls `self.actor_obs_normalizer(obs)` internally.

**Critical:** The normalizer must be in `eval` mode (`.eval()`) during deployment to prevent
updating statistics with real-world data, which would cause distribution shift.

---

## 6. Policy Export

### 6.1 PyTorch Checkpoint (Recommended for initial deployment)

```python
import torch

checkpoint = torch.load("model_50000.pt", map_location="cpu")
# Keys: model_state_dict, optimizer_state_dict, iter, infos

# Load into ActorCritic
from rsl_rl.modules import ActorCritic
policy = ActorCritic(...)
policy.load_state_dict(checkpoint["model_state_dict"])
policy.eval()

# Inference
with torch.no_grad():
    action = policy.act_inference(obs_dict)
```

### 6.2 ONNX Export (Recommended for production)

The codebase already has ONNX export infrastructure. Extend the skater task's runner
to call `export_roller_policy_as_onnx` (adapting from the roller task's `RollerOnPolicyRunner`).

```python
# Export script (to be created)
from mjlab_roller.tasks.roller.rl.exporter import export_roller_policy_as_onnx, attach_onnx_metadata

normalizer = policy.actor_obs_normalizer if policy.actor_obs_normalization else None
export_roller_policy_as_onnx(policy, normalizer=normalizer, path="./export/", filename="skater_policy.onnx")
attach_onnx_metadata(env, run_path="checkpoint_name", path="./export/", filename="skater_policy.onnx")
```

ONNX metadata includes: `joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`,
`action_scale`, `command_axes`, `joystick_mapping`, `action_beta_max`.

### 6.3 TorchScript (Alternative)

```python
traced = torch.jit.trace(policy.actor, torch.randn(1, 920))
traced.save("skater_policy.pt")
# Must also save normalizer mean/std separately
```

---

## 7. Hardware Prerequisites

### 7.1 Unitree G1 EDU

- **Firmware:** v1.3.0 or later (for SDK2 low-level motor control)
- **Mode:** Debug/development mode (L2+B, then L2+R2 on remote)
- **Network:** Ethernet connection, IP 192.168.123.x subnet
- **SDK:** `unitree_sdk2_python` with CycloneDDS supporting `unitree_hg` IDL

### 7.2 Custom Inline Skates

The sim model uses custom inline skate attachments with:
- 4 passive wheels per foot, mounted under each ankle
- Wheel diameter and spacing per `g1.xml` geometry definitions
- The physical skates must match the sim model's geometry as closely as possible

**Critical dimensions to match:**
- Wheel spacing (wheelbase length)
- Wheel diameter
- Skate mounting point relative to ankle roll joint
- Skate frame height (affects standing height)

### 7.3 Host Computer

- GPU with CUDA support (for PyTorch inference, though CPU suffices at 50Hz for a 3-layer MLP)
- Ubuntu 20.04+ with Python 3.10+
- Ethernet interface configured to 192.168.123.x

### 7.4 Joystick / Gamepad

- Any SDL2-compatible gamepad (Xbox, PS4/5, etc.)
- Mapped via pygame to `[vx, vy, wz]` commands
- The training config uses: `left_y → vx`, `left_x → vy`, `right_x → wz`

### 7.5 Optional: Safety Harness / Gantry

Strongly recommended for initial testing. The robot should be suspended with slack in the
harness so it can skate freely but is caught if it falls.

---

## 8. Deployment Software Stack

### 8.1 Communication Layer

```python
# CycloneDDS configuration for G1
import os
os.environ["CYCLONEDDS_URI"] = (
    '<CycloneDDS><Domain><General>'
    '<NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress>'
    '</General></Domain></CycloneDDS>'
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg import LowCmd_, LowState_
```

### 8.2 State Reader

Reads from G1 at the SDK's native rate (typically 500Hz or 1000Hz), maintains latest state:

```python
class G1StateReader:
    """Subscribes to G1 LowState and provides latest readings."""

    def __init__(self):
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.latest_state = None
        self.lock = threading.Lock()

    def callback(self, msg):
        with self.lock:
            self.latest_state = msg

    def get_joint_pos(self, joint_ids: list[int]) -> np.ndarray:
        """Get current joint positions for specified motor IDs."""
        with self.lock:
            return np.array([self.latest_state.motor_state[i].q for i in joint_ids])

    def get_joint_vel(self, joint_ids: list[int]) -> np.ndarray:
        with self.lock:
            return np.array([self.latest_state.motor_state[i].dq for i in joint_ids])

    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (quaternion, angular_velocity) from IMU."""
        with self.lock:
            quat = np.array(self.latest_state.imu_state.quaternion)  # [w,x,y,z]
            gyro = np.array(self.latest_state.imu_state.gyroscope)   # [wx,wy,wz]
            return quat, gyro
```

### 8.3 Motor Commander

Sends position targets at 50Hz:

```python
class G1MotorCommander:
    """Publishes joint position targets to G1 motors."""

    def __init__(self, motor_ids: list[int], kp: np.ndarray, kd: np.ndarray):
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.motor_ids = motor_ids
        self.kp = kp
        self.kd = kd

    def send_targets(self, target_positions: np.ndarray):
        cmd = LowCmd_()
        for i, mid in enumerate(self.motor_ids):
            cmd.motor_cmd[mid].mode = 0x01  # servo mode
            cmd.motor_cmd[mid].q = float(target_positions[i])
            cmd.motor_cmd[mid].dq = 0.0
            cmd.motor_cmd[mid].tau = 0.0
            cmd.motor_cmd[mid].kp = float(self.kp[i])
            cmd.motor_cmd[mid].kd = float(self.kd[i])
        self.publisher.write(cmd)
```

---

## 9. Inference Loop Design

### 9.1 Main Loop (50 Hz)

```
┌─────────────────────────────────────────────┐
│              50 Hz Control Loop              │
│                                              │
│  1. Read state (IMU + encoders)              │
│  2. Estimate base velocity                   │
│  3. Compute FK for skate separation          │
│  4. Build observation vector (92-D)          │
│  5. Push into history buffer (shift left)    │
│  6. Flatten history → 920-D                  │
│  7. Run policy inference (< 1ms on GPU)      │
│  8. Compute joint targets from action        │
│  9. Send targets to motor commander          │
│ 10. Store action in buffer for next obs      │
│                                              │
│  Total budget per step: 20ms                 │
│  Typical inference: 0.1-0.5ms (MLP on GPU)  │
│  Remaining: sensor I/O + FK + safety checks  │
└─────────────────────────────────────────────┘
```

### 9.2 Observation Construction (Pseudocode)

```python
def build_observation(state_reader, velocity_estimator, fk_solver,
                      command, last_action, wheel_contacts,
                      standing_ref, ang_vel_scale=0.25, joint_vel_scale=0.05):
    # 1. Command [3]
    cmd = np.array([command.vx, command.vy, command.wz])

    # 2. Base linear velocity in body frame [3]
    base_lin_vel_b = velocity_estimator.get_velocity()

    # 3. Base angular velocity in body frame, scaled [3]
    _, gyro = state_reader.get_imu()
    base_ang_vel_b = gyro * ang_vel_scale

    # 4. Projected gravity [3]
    quat, _ = state_reader.get_imu()
    gravity_world = np.array([0.0, 0.0, -1.0])
    proj_gravity = quat_apply_inverse(quat, gravity_world)

    # 5. Joint positions relative to standing reference [23]
    joint_pos = state_reader.get_joint_pos(POLICY_MOTOR_IDS)
    joint_pos_rel = joint_pos - standing_ref

    # 6. Joint velocities, scaled [23]
    joint_vel = state_reader.get_joint_vel(POLICY_MOTOR_IDS)
    joint_vel_scaled = joint_vel * joint_vel_scale

    # 7. Previous actions [23]
    prev_actions = last_action

    # 8. Wheel contact flags [8]
    contacts = wheel_contacts  # np.ones(8) initially

    # 9. Skate separation [3]
    skate_sep = fk_solver.compute_skate_separation(joint_pos, quat)

    obs_frame = np.concatenate([
        cmd,              # 3
        base_lin_vel_b,   # 3
        base_ang_vel_b,   # 3
        proj_gravity,     # 3
        joint_pos_rel,    # 23
        joint_vel_scaled, # 23
        prev_actions,     # 23
        contacts,         # 8
        skate_sep,        # 3
    ])  # Total: 92

    return obs_frame
```

### 9.3 History Buffer Management

```python
class ObservationHistory:
    def __init__(self, frame_dim=92, history_length=10):
        self.buffer = np.zeros((history_length, frame_dim))
        self.history_length = history_length

    def push(self, obs_frame: np.ndarray):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs_frame

    def get_flat(self) -> np.ndarray:
        return self.buffer.flatten()  # shape: (920,)

    def reset(self, initial_frame: np.ndarray):
        self.buffer[:] = initial_frame[np.newaxis, :]
```

### 9.4 Action Post-Processing

```python
def action_to_joint_targets(action: np.ndarray,
                            default_pos: np.ndarray,
                            action_scale: np.ndarray,
                            beta: float = 1.0) -> np.ndarray:
    """Convert policy output to joint position targets."""
    return default_pos + action * action_scale * beta
```

### 9.5 Full Inference Loop

```python
def run_deployment(policy, state_reader, commander, joystick,
                   velocity_estimator, fk_solver, config):
    history = ObservationHistory(frame_dim=92, history_length=10)
    last_action = np.zeros(23)
    wheel_contacts = np.ones(8)
    rate = Rate(config.control_freq)  # 50 Hz

    # Initialize history with standing observation
    initial_obs = build_observation(
        state_reader, velocity_estimator, fk_solver,
        Command(0, 0, 0), last_action, wheel_contacts,
        config.standing_ref
    )
    history.reset(initial_obs)

    while running:
        # Read command
        command = joystick.read()

        # Build current observation frame
        obs_frame = build_observation(
            state_reader, velocity_estimator, fk_solver,
            command, last_action, wheel_contacts,
            config.standing_ref
        )
        history.push(obs_frame)

        # Flatten and run policy
        obs_flat = torch.from_numpy(history.get_flat()).float().unsqueeze(0)
        with torch.no_grad():
            action = policy.act_inference({"policy": obs_flat})
        action_np = action.squeeze(0).cpu().numpy()

        # Convert to joint targets
        targets = action_to_joint_targets(
            action_np, config.default_pos,
            config.action_scale, beta=1.0
        )

        # Safety checks
        targets = apply_joint_limits(targets, config.joint_limits)
        targets = apply_rate_limit(targets, last_targets, config.max_delta)

        # Send to robot
        commander.send_targets(targets)

        # Store for next observation
        last_action = action_np

        rate.sleep()
```

---

## 10. Safety System

### 10.1 Joint Limit Clamping

Clamp all target positions to the MJCF joint limits (with 10% soft margin from training):

```python
def apply_joint_limits(targets, limits, soft_factor=0.9):
    lower = limits[:, 0] * soft_factor
    upper = limits[:, 1] * soft_factor
    return np.clip(targets, lower, upper)
```

### 10.2 Rate Limiting

Prevent sudden large jumps in joint targets:

```python
def apply_rate_limit(targets, prev_targets, max_delta_per_step):
    delta = targets - prev_targets
    delta = np.clip(delta, -max_delta_per_step, max_delta_per_step)
    return prev_targets + delta
```

### 10.3 Orientation Kill Switch

If the robot tilts beyond the training termination angle (70°), switch to a safe recovery mode
(go limp or hold last known safe position):

```python
def check_orientation_safe(projected_gravity, limit_rad=1.22):  # 70°
    cos_angle = -projected_gravity[2]  # dot with [0,0,-1]
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return angle < limit_rad
```

### 10.4 Emergency Stop

- Hardware E-stop button (Unitree remote L1+L2+A)
- Software kill: Ctrl+C sets `running = False` and commands the standing pose
- Standing recovery: Hold the standing reference pose with full PD gains for 5s before shutdown

### 10.5 Startup Sequence

1. Power on G1 in damping/debug mode
2. **Safety harness engaged** (for initial tests)
3. Slowly move from current position to standing pose over 3 seconds (linear interpolation)
4. Wait 2 seconds for stabilization
5. Begin policy loop with zero velocity command for 5 seconds
6. Enable joystick commands

---

## 11. System Identification Checklist

Before deployment, verify these parameters match between sim and real:

| Parameter | Sim Value | How to Verify/Calibrate |
|---|---|---|
| Joint zero positions | MJCF qpos0 | Command zero to all joints, measure with protractor/encoder |
| Joint direction signs | MJCF joint axis | Command +0.1 rad, verify physical direction matches |
| PD gains (Kp, Kd) | See §4.4 | Start at sim values, tune on real if oscillation/sluggishness |
| Joint limits | MJCF range | Read from G1 firmware, compare |
| Link masses | MJCF body mass | Weigh robot limbs if accessible; DR covers ±10% |
| IMU orientation | MJCF imu_in_pelvis site | Verify quaternion convention (wxyz vs xyzw) |
| Control frequency | 50 Hz | Measure actual loop timing, ensure no jitter > 2ms |
| Skate dimensions | MJCF geometry | Measure physical skates, update XML if needed |
| Wheel friction | DR range [0.1, 0.8] | Test on target surface; should fall within training range |

---

## 12. Validation Stages

### Stage 1: Desk Check (No Hardware)
- [ ] Export ONNX model, verify input/output dimensions
- [ ] Run inference in simulation headless, compare actions with training play
- [ ] Verify joint mapping table against G1 SDK motor IDs
- [ ] Verify normalization statistics are correctly loaded

### Stage 2: Sim-in-the-Loop (Software Validation)
- [ ] Run the deployment code against the MuJoCo sim (replace SDK with sim API)
- [ ] Verify observation construction matches training observations exactly
- [ ] Verify action post-processing reproduces identical joint targets
- [ ] Run for 10,000 steps, compare policy behavior with training play

### Stage 3: Static Hardware Test
- [ ] Connect to G1 in debug mode
- [ ] Command standing pose, verify all joints reach correct positions
- [ ] Read IMU data, verify quaternion convention
- [ ] Test each joint individually: +0.05 rad offset, verify correct joint and direction
- [ ] Read joint velocities while manually moving joints, verify sign convention

### Stage 4: Suspended Test (Safety Harness)
- [ ] Hang robot in harness, feet barely touching ground
- [ ] Run policy with zero velocity command
- [ ] Verify stable stance (no oscillation, no drift)
- [ ] Apply small vx command (0.1 m/s), observe leg motion
- [ ] Test emergency stop procedure

### Stage 5: Ground Test (Harness + Flat Surface)
- [ ] Place on smooth, flat surface with harness slack
- [ ] Zero velocity command: robot should stand stably on skates
- [ ] Gradually increase vx: 0.1 → 0.3 → 0.5 → 1.0 m/s
- [ ] Test turning: small wz commands
- [ ] Test combined vx + wz
- [ ] Monitor and log all observations for post-analysis

### Stage 6: Free Skating (Final Validation)
- [ ] Remove harness (spotter present)
- [ ] Conservative velocity commands only
- [ ] Full velocity range testing
- [ ] Long-duration stability test (> 60s continuous skating)

---

## 13. Known Risks and Mitigations

### 13.1 Base Velocity Estimation Error

**Risk:** The policy was trained with ground-truth velocity. Real estimation will be noisy/biased.

**Mitigation:**
- The training applies U(-0.1, 0.1) noise to `base_lin_vel`. Consider increasing to U(-0.2, 0.2)
  and retraining if real estimator is noisier.
- Use a Kalman filter or complementary filter for velocity estimation.
- Validate by comparing estimated vs mocap-measured velocity during Stage 4.

### 13.2 Sim-Real Actuator Dynamics Gap

**Risk:** Real motor dynamics may differ from sim position-mode actuators.

**Mitigation:**
- Training randomizes actuator Kp (±10%) and Kd (±10%).
- Tune PD gains on real hardware. If the real motors have significant backlash or friction
  not modeled in sim, consider training with additional actuator modeling.

### 13.3 Wheel-Ground Friction Mismatch

**Risk:** Real skating surface may have different friction than training range.

**Mitigation:**
- Training randomizes static friction [0.1, 0.8] and dynamic friction [0.1, 0.4].
- Test on the intended deployment surface. Most indoor smooth floors fall within this range.
- If outdoor deployment is needed, retrain with extended friction ranges.

### 13.4 Communication Latency

**Risk:** DDS communication delay may cause the policy to operate on stale state.

**Mitigation:**
- The G1 SDK over ethernet has < 1ms latency, well within the 20ms control budget.
- Monitor actual loop timing. If jitter exceeds 5ms, investigate OS scheduling (use
  `SCHED_FIFO` or real-time kernel).
- Consider training with 1-step observation delay for extra robustness.

### 13.5 Skate Geometry Mismatch

**Risk:** Physical skate dimensions don't match the MJCF model.

**Mitigation:**
- Measure physical skates precisely and update `g1.xml` before final training.
- Key: wheel spacing, wheel radius, skate height, mounting angle.
- The FK-based skate separation computation must use the real geometry.

---

## 14. File Layout

Proposed files to add for deployment:

```
scripts/deploy/
├── export_policy.py           # Export checkpoint → ONNX with metadata
├── deploy_g1.py               # Main deployment loop
├── g1_state_reader.py         # Unitree SDK state subscriber
├── g1_motor_commander.py      # Unitree SDK motor command publisher
├── observation_builder.py     # Build 92-D obs from sensors
├── observation_history.py     # 10-frame history buffer
├── velocity_estimator.py      # IMU + FK based base velocity estimation
├── fk_solver.py               # Forward kinematics for skate separation
├── safety.py                  # Joint limits, rate limits, orientation check
├── joystick_command.py        # Gamepad → [vx, vy, wz] mapping
├── config.py                  # Joint mapping, PD gains, limits, reference pose
└── README.md                  # Deployment instructions
```

---

## Appendix A: Observation Dimension Verification

To verify the observation dimensions match between sim and deployment:

```python
# In simulation
env = ...  # create env
obs = env.reset()
policy_obs = obs["policy"]
print(f"Policy obs shape: {policy_obs.shape}")  # Expected: (num_envs, 920)
print(f"Per-frame dim: {policy_obs.shape[-1] // 10}")  # Expected: 92
```

## Appendix B: Quick Reference — Quaternion Convention

The MJCF model and Unitree SDK both use **[w, x, y, z]** quaternion ordering.
Verify this matches your `quat_apply_inverse` implementation.

```python
def quat_apply_inverse(quat_wxyz, vec):
    """Rotate vec by the inverse of quaternion (w,x,y,z format)."""
    w, x, y, z = quat_wxyz
    # Inverse rotation: conjugate for unit quaternion
    # q_conj = [w, -x, -y, -z]
    # v_rotated = q_conj * v * q
    t = 2.0 * np.cross(np.array([-x, -y, -z]), vec)
    return vec + w * t + np.cross(np.array([-x, -y, -z]), t)
```

## Appendix C: Comparison with Psi0 Deployment Architecture

| Aspect | Psi0 (VLA, loco-manipulation) | This Project (RL locomotion) |
|---|---|---|
| Policy type | Diffusion VLA (Qwen3-VL + flow matching) | MLP actor-critic (PPO) |
| Inference latency | ~100-300ms (heavy GPU) | ~0.1-0.5ms (lightweight MLP) |
| Architecture | Server-client (WebSocket/HTTP) | Single-process (direct SDK) |
| Control split | System-0 RL tracker + VLA upper body | End-to-end joint position control |
| Observation | RGB camera + 32-D proprio | 920-D proprioceptive only |
| Action space | 36-D (hands + arms + torso + locomotion) | 23-D (all joints, position targets) |
| Control rate | 30Hz VLA → 60Hz tracking controller | 50Hz direct control |

**Key difference:** Psi0 uses a hierarchical architecture where an RL-based "System-0" tracking
controller handles low-level locomotion while the VLA controls upper body and high-level motion.
Our roller-skating policy is a single end-to-end controller that directly outputs all 23 joint
targets — there is no separate locomotion controller underneath.

This means our deployment is simpler (no IK solver, no separate tracking policy), but the
RL policy must handle all aspects of balance and locomotion directly, making the sim-to-real
gap more critical to bridge carefully.
