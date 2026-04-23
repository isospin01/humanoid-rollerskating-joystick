"""Safety utilities for real-world deployment of the roller-skating policy.

Every target sent to the motors passes through this module's checks.
"""

from __future__ import annotations

import numpy as np

from config import (
    MAX_JOINT_DELTA_PER_STEP,
    MAX_TILT_ANGLE_RAD,
    NUM_JOINTS,
    SOFT_JOINT_LIMITS,
)


def clamp_joint_limits(
    targets: np.ndarray,
    limits: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp target positions to soft joint limits."""
    if limits is None:
        limits = SOFT_JOINT_LIMITS
    return np.clip(targets, limits[:, 0], limits[:, 1])


def rate_limit(
    targets: np.ndarray,
    prev_targets: np.ndarray,
    max_delta: float = MAX_JOINT_DELTA_PER_STEP,
) -> np.ndarray:
    """Limit per-step position change to prevent violent jumps."""
    delta = targets - prev_targets
    delta = np.clip(delta, -max_delta, max_delta)
    return prev_targets + delta


def is_orientation_safe(
    projected_gravity: np.ndarray,
    max_tilt_rad: float = MAX_TILT_ANGLE_RAD,
) -> bool:
    """Check if the torso tilt is within the safe range.

    ``projected_gravity`` is gravity expressed in the body frame.
    When upright it equals approximately [0, 0, -1].
    """
    cos_angle = -projected_gravity[2]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    tilt = np.arccos(cos_angle)
    return float(tilt) < max_tilt_rad


def compute_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0,0,-1] into the body frame."""
    from observation_builder import quat_apply_inverse
    return quat_apply_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))


class SafetyGuard:
    """Aggregate safety checks run on every control step.

    Attributes:
        triggered: True if any safety check has tripped.
        reason: Human-readable reason for the safety stop.
    """

    def __init__(
        self,
        joint_limits: np.ndarray | None = None,
        max_delta: float = MAX_JOINT_DELTA_PER_STEP,
        max_tilt_rad: float = MAX_TILT_ANGLE_RAD,
    ):
        self.joint_limits = joint_limits if joint_limits is not None else SOFT_JOINT_LIMITS
        self.max_delta = max_delta
        self.max_tilt_rad = max_tilt_rad

        self._prev_targets = None
        self.triggered = False
        self.reason = ""

    def reset(self, initial_pos: np.ndarray):
        self._prev_targets = initial_pos.copy()
        self.triggered = False
        self.reason = ""

    def process(
        self,
        raw_targets: np.ndarray,
        projected_gravity: np.ndarray,
    ) -> np.ndarray:
        """Apply all safety filters and return safe targets.

        If the orientation exceeds the limit, ``self.triggered`` is set and
        the returned targets are the last safe targets (hold position).
        """
        if self.triggered:
            return self._prev_targets.copy()

        if not is_orientation_safe(projected_gravity, self.max_tilt_rad):
            self.triggered = True
            self.reason = (
                f"Tilt exceeded {np.degrees(self.max_tilt_rad):.0f}° limit"
            )
            return self._prev_targets.copy()

        safe = clamp_joint_limits(raw_targets, self.joint_limits)

        if self._prev_targets is not None:
            safe = rate_limit(safe, self._prev_targets, self.max_delta)

        self._prev_targets = safe.copy()
        return safe
