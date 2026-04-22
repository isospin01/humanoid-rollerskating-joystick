from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.tasks.bootstrap import bootstrap_task_registry  # type: ignore
from mjlab_roller.tasks.registry import list_tasks, load_env_cfg  # type: ignore
from mjlab_roller.tasks.skater.mdp.command import SkaterJoystickCommandCfg  # type: ignore


class TaskRegistryEntryTests(unittest.TestCase):
  def test_registry_contains_legacy_and_skater_tasks(self) -> None:
    bootstrap_task_registry()
    tasks = list_tasks()
    self.assertIn("Mjlab-Roller-Flat-Unitree-G1", tasks)
    self.assertIn("Mjlab-Roller-Joystick-Flat-Unitree-G1", tasks)

  def test_skater_task_uses_joystick_command_cfg(self) -> None:
    bootstrap_task_registry()
    env_cfg = load_env_cfg("Mjlab-Roller-Joystick-Flat-Unitree-G1")
    self.assertIsInstance(env_cfg.commands["skate"], SkaterJoystickCommandCfg)


if __name__ == "__main__":
  unittest.main()
