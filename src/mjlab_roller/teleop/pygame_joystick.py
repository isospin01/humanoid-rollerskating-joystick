"""Joystick input helpers for sim teleop."""

from __future__ import annotations

import math

import numpy as np


def apply_deadzone(value: float, deadzone: float) -> float:
  if deadzone <= 0.0:
    return value
  magnitude = abs(value)
  if magnitude <= deadzone:
    return 0.0
  scaled = (magnitude - deadzone) / max(1.0 - deadzone, 1e-6)
  return math.copysign(scaled, value)


class PygameJoystickCommandInput:
  """Read a single USB gamepad and convert it to `[vx, vy, wz]` commands."""

  def __init__(
    self,
    *,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    deadzone: float = 0.10,
    smoothing: float = 0.20,
  ) -> None:
    try:
      import pygame
    except ModuleNotFoundError as exc:
      raise ModuleNotFoundError(
        "Joystick teleop requires the `pygame` package. Install project dependencies "
        "in the new repo before using `--command-source joystick`."
      ) from exc

    self._pygame = pygame
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
      raise RuntimeError("No joystick/gamepad detected for pygame teleop input.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    self._joystick = joystick
    self.max_vx = max_vx
    self.max_vy = max_vy
    self.max_wz = max_wz
    self.deadzone = deadzone
    self.smoothing = smoothing
    self._smoothed = np.zeros(3, dtype=np.float32)

  def set_limits(self, *, max_vx: float, max_vy: float, max_wz: float) -> None:
    self.max_vx = max(0.0, float(max_vx))
    self.max_vy = max(0.0, float(max_vy))
    self.max_wz = max(0.0, float(max_wz))

  def _axis(self, index: int) -> float:
    if index >= self._joystick.get_numaxes():
      return 0.0
    return float(self._joystick.get_axis(index))

  def read_command(self) -> np.ndarray:
    self._pygame.event.pump()

    # Xbox-style default mapping:
    # left stick Y -> forward, left stick X -> lateral, right stick X -> yaw.
    axis_forward = apply_deadzone(-self._axis(1), self.deadzone)
    axis_lateral = apply_deadzone(self._axis(0), self.deadzone)
    yaw_axis_index = 3 if self._joystick.get_numaxes() > 3 else 2
    axis_yaw = apply_deadzone(self._axis(yaw_axis_index), self.deadzone)

    raw = np.array(
      [
        max(0.0, axis_forward) * self.max_vx,
        axis_lateral * self.max_vy,
        axis_yaw * self.max_wz,
      ],
      dtype=np.float32,
    )
    if self.smoothing <= 0.0:
      self._smoothed = raw
    else:
      alpha = float(np.clip(self.smoothing, 0.0, 1.0))
      self._smoothed = (1.0 - alpha) * self._smoothed + alpha * raw
    return self._smoothed.copy()

  def close(self) -> None:
    if getattr(self, "_joystick", None) is None:
      return
    try:
      self._joystick.quit()
    finally:
      self._pygame.joystick.quit()
      self._pygame.quit()
      self._joystick = None
