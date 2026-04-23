"""Deployment configuration for the G1 roller-skating policy.

This module holds all constants and mappings required to bridge the trained
policy with the physical Unitree G1 robot.  Every value here has a direct
counterpart either in the training environment configuration or in the
Unitree G1 SDK.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Joint definitions — order MUST match the policy (see control_spec.py)
# ---------------------------------------------------------------------------

CONTROLLED_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

NUM_JOINTS: int = len(CONTROLLED_JOINT_NAMES)  # 23

# Standing reference pose (radians) — matches STANDING_SKATE_CONTROLLED_JOINT_POS
STANDING_JOINT_POS: np.ndarray = np.array([
    -0.05,  0.0,  0.0,  0.45, -0.25, 0.0,     # left leg
    -0.05,  0.0,  0.0,  0.45, -0.25, 0.0,     # right leg
     0.0,   0.0,  0.05,                        # waist
    -0.10,  0.45, -0.20, 1.10,                 # left arm
    -0.10, -0.45,  0.20, 1.10,                 # right arm
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Unitree G1 motor ID mapping
#
# IMPORTANT: These motor IDs must be verified against your specific G1 EDU
# firmware version.  The mapping below follows the standard G1-29DOF layout.
# Run the joint-verification procedure (Stage 3 in DEPLOYMENT_PIPELINE.md)
# before trusting these values.
# ---------------------------------------------------------------------------

UNITREE_MOTOR_IDS: dict[str, int] = {
    "left_hip_pitch_joint":     0,
    "left_hip_roll_joint":      1,
    "left_hip_yaw_joint":       2,
    "left_knee_joint":          3,
    "left_ankle_pitch_joint":   4,
    "left_ankle_roll_joint":    5,
    "right_hip_pitch_joint":    6,
    "right_hip_roll_joint":     7,
    "right_hip_yaw_joint":      8,
    "right_knee_joint":         9,
    "right_ankle_pitch_joint":  10,
    "right_ankle_roll_joint":   11,
    "waist_yaw_joint":          12,
    "waist_roll_joint":         13,
    "waist_pitch_joint":        14,
    "left_shoulder_pitch_joint":  15,
    "left_shoulder_roll_joint":   16,
    "left_shoulder_yaw_joint":    17,
    "left_elbow_joint":           18,
    "right_shoulder_pitch_joint": 19,
    "right_shoulder_roll_joint":  20,
    "right_shoulder_yaw_joint":   21,
    "right_elbow_joint":          22,
}

POLICY_TO_MOTOR_IDS: list[int] = [
    UNITREE_MOTOR_IDS[name] for name in CONTROLLED_JOINT_NAMES
]

# ---------------------------------------------------------------------------
# Actuator PD gains — must match the sim actuator configuration.
#
# Values are derived from g1.py motor constants:
#   natural_freq = 10 * 2π ≈ 62.832 rad/s
#   damping_ratio = 2.0
#   Kp = armature × ω²
#   Kd = 2 × damping_ratio × armature × ω
# ---------------------------------------------------------------------------

_OMEGA = 10.0 * 2.0 * np.pi

# Reflected inertias per motor class (from g1.py two-stage-planetary computation)
# These are approximate — loaded from metadata at runtime when available.
_ARMATURE_5020 = 0.0122
_ARMATURE_7520_14 = 0.0893
_ARMATURE_7520_22 = 0.1619

_KP_5020 = _ARMATURE_5020 * _OMEGA ** 2
_KD_5020 = 2.0 * 2.0 * _ARMATURE_5020 * _OMEGA
_KP_7520_14 = _ARMATURE_7520_14 * _OMEGA ** 2
_KD_7520_14 = 2.0 * 2.0 * _ARMATURE_7520_14 * _OMEGA
_KP_7520_22 = _ARMATURE_7520_22 * _OMEGA ** 2
_KD_7520_22 = 2.0 * 2.0 * _ARMATURE_7520_22 * _OMEGA

_ACTUATOR_CLASS: dict[str, str] = {
    "left_hip_pitch_joint":     "7520_14",
    "left_hip_roll_joint":      "7520_22",
    "left_hip_yaw_joint":       "7520_14",
    "left_knee_joint":          "7520_22",
    "left_ankle_pitch_joint":   "ankle",
    "left_ankle_roll_joint":    "ankle",
    "right_hip_pitch_joint":    "7520_14",
    "right_hip_roll_joint":     "7520_22",
    "right_hip_yaw_joint":      "7520_14",
    "right_knee_joint":         "7520_22",
    "right_ankle_pitch_joint":  "ankle",
    "right_ankle_roll_joint":   "ankle",
    "waist_yaw_joint":          "7520_14",
    "waist_roll_joint":         "waist",
    "waist_pitch_joint":        "waist",
    "left_shoulder_pitch_joint":  "5020",
    "left_shoulder_roll_joint":   "5020",
    "left_shoulder_yaw_joint":    "5020",
    "left_elbow_joint":           "5020",
    "right_shoulder_pitch_joint": "5020",
    "right_shoulder_roll_joint":  "5020",
    "right_shoulder_yaw_joint":   "5020",
    "right_elbow_joint":          "5020",
}

_CLASS_GAINS = {
    "5020":    (_KP_5020,     _KD_5020),
    "7520_14": (_KP_7520_14,  _KD_7520_14),
    "7520_22": (_KP_7520_22,  _KD_7520_22),
    "ankle":   (_KP_5020 * 2, _KD_5020 * 2),
    "waist":   (_KP_5020 * 2, _KD_5020 * 2),
}


def get_pd_gains() -> tuple[np.ndarray, np.ndarray]:
    """Return (kp, kd) arrays of shape (23,) in policy joint order."""
    kp = np.zeros(NUM_JOINTS)
    kd = np.zeros(NUM_JOINTS)
    for i, name in enumerate(CONTROLLED_JOINT_NAMES):
        cls = _ACTUATOR_CLASS[name]
        kp[i], kd[i] = _CLASS_GAINS[cls]
    return kp, kd


# ---------------------------------------------------------------------------
# Observation configuration
# ---------------------------------------------------------------------------

OBS_FRAME_DIM: int = 92
OBS_HISTORY_LENGTH: int = 10
OBS_TOTAL_DIM: int = OBS_FRAME_DIM * OBS_HISTORY_LENGTH  # 920

ANG_VEL_SCALE: float = 0.25
JOINT_VEL_SCALE: float = 0.05

# ---------------------------------------------------------------------------
# Control configuration
# ---------------------------------------------------------------------------

CONTROL_FREQ_HZ: int = 50
CONTROL_DT: float = 1.0 / CONTROL_FREQ_HZ  # 0.02s

ACTION_BETA: float = 1.0  # final curriculum stage

# Approximate joint limits (radians) from MJCF — used for safety clamping
# Format: (lower, upper) per joint in policy order
JOINT_LIMITS: np.ndarray = np.array([
    [-2.53,  2.88],   # left_hip_pitch
    [-0.52,  2.97],   # left_hip_roll
    [-2.76,  2.76],   # left_hip_yaw
    [-0.09,  2.53],   # left_knee
    [-0.87,  0.52],   # left_ankle_pitch
    [-0.26,  0.26],   # left_ankle_roll
    [-2.53,  2.88],   # right_hip_pitch
    [-2.97,  0.52],   # right_hip_roll
    [-2.76,  2.76],   # right_hip_yaw
    [-0.09,  2.53],   # right_knee
    [-0.87,  0.52],   # right_ankle_pitch
    [-0.26,  0.26],   # right_ankle_roll
    [-2.76,  2.76],   # waist_yaw
    [-0.26,  0.26],   # waist_roll
    [-0.26,  0.26],   # waist_pitch
    [-3.11,  2.19],   # left_shoulder_pitch
    [-1.58,  2.27],   # left_shoulder_roll
    [-2.62,  2.62],   # left_shoulder_yaw
    [ 0.00,  2.53],   # left_elbow
    [-3.11,  2.19],   # right_shoulder_pitch
    [-2.27,  1.58],   # right_shoulder_roll
    [-2.62,  2.62],   # right_shoulder_yaw
    [ 0.00,  2.53],   # right_elbow
], dtype=np.float64)

SOFT_JOINT_LIMIT_FACTOR: float = 0.9
SOFT_JOINT_LIMITS: np.ndarray = JOINT_LIMITS * SOFT_JOINT_LIMIT_FACTOR

# Maximum position change per control step (rad) — safety rate limit
MAX_JOINT_DELTA_PER_STEP: float = 0.5

# Orientation safety: max tilt angle (rad) before emergency stop
MAX_TILT_ANGLE_RAD: float = np.deg2rad(70.0)

# Standing height target (meters) — used for height monitoring
STANDING_HEIGHT: float = 0.80

# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

NETWORK_INTERFACE: str = "192.168.123.123"
CYCLONEDDS_URI: str = (
    '<CycloneDDS><Domain><General>'
    f'<NetworkInterfaceAddress>{NETWORK_INTERFACE}</NetworkInterfaceAddress>'
    '</General></Domain></CycloneDDS>'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class DeploymentConfig:
    """Aggregated config that can be loaded from metadata JSON or defaults."""

    joint_names: list[str] = field(default_factory=lambda: list(CONTROLLED_JOINT_NAMES))
    default_pos: np.ndarray = field(default_factory=lambda: STANDING_JOINT_POS.copy())
    action_scale: np.ndarray | None = None
    action_beta: float = ACTION_BETA
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    obs_frame_dim: int = OBS_FRAME_DIM
    obs_history_length: int = OBS_HISTORY_LENGTH
    control_freq_hz: int = CONTROL_FREQ_HZ
    joint_limits: np.ndarray = field(default_factory=lambda: SOFT_JOINT_LIMITS.copy())
    max_delta_per_step: float = MAX_JOINT_DELTA_PER_STEP
    max_tilt_rad: float = MAX_TILT_ANGLE_RAD
    motor_ids: list[int] = field(default_factory=lambda: list(POLICY_TO_MOTOR_IDS))

    def __post_init__(self):
        if self.kp is None or self.kd is None:
            self.kp, self.kd = get_pd_gains()

    @classmethod
    def from_metadata(cls, metadata_path: str) -> DeploymentConfig:
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        cfg = cls()

        if "default_joint_pos" in meta:
            cfg.default_pos = np.array(meta["default_joint_pos"], dtype=np.float64)
        if "action_scale" in meta:
            cfg.action_scale = np.array(meta["action_scale"], dtype=np.float64)
        if "action_beta" in meta:
            cfg.action_beta = float(meta["action_beta"])
        if "joint_names" in meta:
            cfg.joint_names = meta["joint_names"]

        if "joint_stiffness" in meta and "joint_damping" in meta:
            cfg.kp = np.array(meta["joint_stiffness"], dtype=np.float64)
            cfg.kd = np.array(meta["joint_damping"], dtype=np.float64)

        return cfg
