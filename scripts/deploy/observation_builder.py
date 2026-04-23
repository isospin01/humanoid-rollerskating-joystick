"""Build the 92-D observation vector and maintain the 10-frame history buffer.

The observation vector layout must *exactly* match what the policy saw during
training.  Any mismatch in ordering, scaling, or frame convention will cause
the deployed policy to produce nonsensical actions.
"""

from __future__ import annotations

import numpy as np

from config import (
    ANG_VEL_SCALE,
    JOINT_VEL_SCALE,
    NUM_JOINTS,
    OBS_FRAME_DIM,
    OBS_HISTORY_LENGTH,
    OBS_TOTAL_DIM,
)


def quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate *vec* by the inverse of a unit quaternion in (w, x, y, z) order."""
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    u = np.array([-x, -y, -z])
    t = 2.0 * np.cross(u, vec)
    return vec + w * t + np.cross(u, t)


class VelocityEstimator:
    """Simple complementary-filter velocity estimator from IMU data.

    For initial deployment this can be replaced with a constant zero vector
    or an external mocap system.  The policy was trained with ±0.1 m/s noise
    on this channel, giving some tolerance for estimation error.
    """

    def __init__(self, dt: float, alpha: float = 0.98):
        self._dt = dt
        self._alpha = alpha
        self._vel_world = np.zeros(3)
        self._gravity = np.array([0.0, 0.0, -9.81])
        self._initialized = False

    def update(self, accel_body: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """Integrate accelerometer and return body-frame velocity."""
        accel_world = quat_apply_inverse(
            np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]]),
            accel_body,
        )
        accel_world -= self._gravity

        if not self._initialized:
            self._vel_world = np.zeros(3)
            self._initialized = True

        self._vel_world = self._alpha * (self._vel_world + accel_world * self._dt)

        vel_body = quat_apply_inverse(quat_wxyz, self._vel_world)
        return vel_body

    def reset(self):
        self._vel_world = np.zeros(3)
        self._initialized = False

    def set_velocity(self, vel_body: np.ndarray):
        """Override with an externally measured velocity (e.g. from mocap)."""
        self._vel_world = vel_body  # approximate; ignores body→world rotation


class ForwardKinematicsSolver:
    """Compute skate separation from joint positions using the MJCF kinematic chain.

    This is a simplified implementation that computes the vector from the
    left skate center to the right skate center in world frame.

    For a production deployment, load the MJCF model into mujoco and use
    ``mj_kinematics`` for exact FK.
    """

    def __init__(self):
        self._mj_model = None
        self._mj_data = None

    def initialize_from_xml(self, xml_path: str):
        """Load MJCF model for FK computation."""
        import mujoco
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)

    def compute_skate_separation(
        self,
        joint_pos_all: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Return world-frame separation vector (right_center - left_center).

        If the MJCF model is loaded, uses mujoco FK.  Otherwise, returns a
        constant approximation based on the standing pose.
        """
        if self._mj_model is not None:
            return self._compute_mujoco_fk(joint_pos_all, base_quat_wxyz)

        # Fallback: approximate from hip width + joint positions.
        # In the standing pose the skates are roughly 0.13m apart in Y.
        hip_width = 0.129
        return np.array([0.0, -hip_width, 0.0])

    def _compute_mujoco_fk(
        self,
        joint_pos: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ) -> np.ndarray:
        import mujoco

        self._mj_data.qpos[3:7] = base_quat_wxyz
        ctrl_joint_ids = []
        for name in [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
        ]:
            jid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ctrl_joint_ids.append(jid)

        for i, jid in enumerate(ctrl_joint_ids):
            qpos_adr = self._mj_model.jnt_qposadr[jid]
            self._mj_data.qpos[qpos_adr] = joint_pos[i]

        mujoco.mj_kinematics(self._mj_model, self._mj_data)

        marker_names = [
            "left_skate_front_marker", "left_skate_rear_marker",
            "right_skate_front_marker", "right_skate_rear_marker",
        ]
        positions = []
        for name in marker_names:
            sid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            positions.append(self._mj_data.site_xpos[sid].copy())

        left_center = (positions[0] + positions[1]) / 2.0
        right_center = (positions[2] + positions[3]) / 2.0
        return right_center - left_center


class ObservationBuilder:
    """Construct the 92-D per-frame observation and maintain the history buffer."""

    def __init__(
        self,
        standing_ref: np.ndarray,
        ang_vel_scale: float = ANG_VEL_SCALE,
        joint_vel_scale: float = JOINT_VEL_SCALE,
        history_length: int = OBS_HISTORY_LENGTH,
    ):
        self.standing_ref = standing_ref.copy()
        self.ang_vel_scale = ang_vel_scale
        self.joint_vel_scale = joint_vel_scale
        self.history_length = history_length

        self._history = np.zeros((history_length, OBS_FRAME_DIM), dtype=np.float32)
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    def reset(self, initial_joint_pos: np.ndarray | None = None):
        """Fill the history buffer with the initial standing observation."""
        frame = np.zeros(OBS_FRAME_DIM, dtype=np.float32)
        # projected_gravity at index 9:12 (after command[3] + lin_vel[3] + ang_vel[3])
        frame[9:12] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if initial_joint_pos is not None:
            # joint_pos_rel at index 12:35
            frame[12:12 + NUM_JOINTS] = (initial_joint_pos - self.standing_ref).astype(np.float32)
        # wheel_contacts at index 81:89 — default to all-contact
        frame[81:89] = 1.0
        self._history[:] = frame[np.newaxis, :]
        self._last_action[:] = 0.0

    def build_frame(
        self,
        command: np.ndarray,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        wheel_contacts: np.ndarray,
        skate_separation: np.ndarray,
    ) -> np.ndarray:
        """Assemble a single 92-D observation frame.

        Args:
            command: [vx, vy, wz] body-frame velocity command (3,)
            base_lin_vel_b: body-frame linear velocity (3,)
            base_ang_vel_b: body-frame angular velocity (3,) — raw, NOT pre-scaled
            projected_gravity: gravity in body frame (3,)
            joint_pos: absolute joint positions (23,)
            joint_vel: joint velocities (23,) — raw, NOT pre-scaled
            wheel_contacts: per-wheel contact flags (8,) — 1.0 or 0.0
            skate_separation: world-frame R-L skate center vector (3,)

        Returns:
            obs_frame: (92,) observation vector
        """
        frame = np.empty(OBS_FRAME_DIM, dtype=np.float32)
        idx = 0

        # command [3]
        frame[idx:idx + 3] = command
        idx += 3

        # base_lin_vel [3]
        frame[idx:idx + 3] = base_lin_vel_b
        idx += 3

        # base_ang_vel scaled [3]
        frame[idx:idx + 3] = base_ang_vel_b * self.ang_vel_scale
        idx += 3

        # projected_gravity [3]
        frame[idx:idx + 3] = projected_gravity
        idx += 3

        # joint_pos relative to standing [23]
        frame[idx:idx + NUM_JOINTS] = joint_pos - self.standing_ref
        idx += NUM_JOINTS

        # joint_vel scaled [23]
        frame[idx:idx + NUM_JOINTS] = joint_vel * self.joint_vel_scale
        idx += NUM_JOINTS

        # last_action [23]
        frame[idx:idx + NUM_JOINTS] = self._last_action
        idx += NUM_JOINTS

        # wheel_contact [8]
        frame[idx:idx + 8] = wheel_contacts
        idx += 8

        # skate_separation [3]
        frame[idx:idx + 3] = skate_separation
        idx += 3

        assert idx == OBS_FRAME_DIM, f"Frame assembly error: expected {OBS_FRAME_DIM}, got {idx}"

        return frame

    def push_and_get(
        self,
        command: np.ndarray,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        wheel_contacts: np.ndarray,
        skate_separation: np.ndarray,
    ) -> np.ndarray:
        """Build a frame, push onto history, return flattened (920,) observation."""
        frame = self.build_frame(
            command, base_lin_vel_b, base_ang_vel_b, projected_gravity,
            joint_pos, joint_vel, wheel_contacts, skate_separation,
        )

        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = frame

        return self._history.flatten()  # (920,)

    def set_last_action(self, action: np.ndarray):
        """Store the policy's output for inclusion in the next observation frame."""
        self._last_action[:] = action

    @property
    def obs_dim(self) -> int:
        return OBS_TOTAL_DIM
