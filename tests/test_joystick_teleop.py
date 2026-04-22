from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.teleop.pygame_joystick import (  # type: ignore
  PygameJoystickCommandInput,
  apply_deadzone,
)


class _FakeJoystick:
  def __init__(self, axes: list[float]) -> None:
    self.axes = axes
    self.closed = False

  def init(self) -> None:
    return None

  def get_numaxes(self) -> int:
    return len(self.axes)

  def get_axis(self, index: int) -> float:
    return self.axes[index]

  def quit(self) -> None:
    self.closed = True


class _FakeJoystickModule:
  def __init__(self, joystick: _FakeJoystick) -> None:
    self._joystick = joystick
    self.closed = False

  def init(self) -> None:
    return None

  def get_count(self) -> int:
    return 1

  def Joystick(self, index: int) -> _FakeJoystick:
    assert index == 0
    return self._joystick

  def quit(self) -> None:
    self.closed = True


class _FakePygame:
  def __init__(self, axes: list[float]) -> None:
    self.joystick = _FakeJoystickModule(_FakeJoystick(axes))
    self.event = SimpleNamespace(pump=lambda: None)
    self.closed = False

  def init(self) -> None:
    return None

  def quit(self) -> None:
    self.closed = True


class JoystickTeleopTests(unittest.TestCase):
  def test_apply_deadzone_zeroes_small_values(self) -> None:
    self.assertEqual(apply_deadzone(0.04, 0.1), 0.0)
    self.assertGreater(apply_deadzone(0.4, 0.1), 0.0)

  def test_pygame_mapping_matches_vx_vy_wz_contract(self) -> None:
    fake_pygame = _FakePygame([0.2, -0.4, 0.0, 0.3])
    with patch.dict(sys.modules, {"pygame": fake_pygame}):
      teleop = PygameJoystickCommandInput(
        max_vx=1.0,
        max_vy=2.0,
        max_wz=3.0,
        deadzone=0.0,
        smoothing=1.0,
      )
      command = teleop.read_command()
      np.testing.assert_allclose(command, np.array([0.4, 0.4, 0.9], dtype=np.float32))
      teleop.close()
      self.assertTrue(fake_pygame.closed)

  def test_set_limits_updates_scaling(self) -> None:
    fake_pygame = _FakePygame([0.0, -1.0, 0.0, 1.0])
    with patch.dict(sys.modules, {"pygame": fake_pygame}):
      teleop = PygameJoystickCommandInput(
        max_vx=0.3,
        max_vy=0.2,
        max_wz=0.25,
        deadzone=0.0,
        smoothing=1.0,
      )
      teleop.set_limits(max_vx=1.5, max_vy=0.2, max_wz=1.0)
      command = teleop.read_command()
      np.testing.assert_allclose(command, np.array([1.5, 0.0, 1.0], dtype=np.float32))
      teleop.close()


if __name__ == "__main__":
  unittest.main()
