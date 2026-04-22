from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.tasks.bootstrap import bootstrap_task_registry  # type: ignore
from mjlab_roller.tasks.registry import load_env_cfg  # type: ignore
from mjlab_roller.tasks.skater.mdp.events import (  # type: ignore
  randomize_actuator_and_wheel_damping,
)


class _FakeSim:
  def __init__(self) -> None:
    self.model = SimpleNamespace(
      actuator_gainprm=torch.zeros((3, 2, 10), dtype=torch.float32),
      actuator_biasprm=torch.zeros((3, 2, 10), dtype=torch.float32),
      dof_damping=torch.zeros((3, 5), dtype=torch.float32),
      nu=2,
    )
    self.mj_model = SimpleNamespace(jnt_dofadr=np.asarray([0, 1, 3], dtype=np.int32))
    self._defaults = {
      "actuator_gainprm": torch.zeros((2, 10), dtype=torch.float32),
      "actuator_biasprm": torch.zeros((2, 10), dtype=torch.float32),
      "dof_damping": torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50], dtype=torch.float32),
    }
    self._defaults["actuator_gainprm"][:, 0] = torch.tensor([10.0, 20.0])
    self._defaults["actuator_biasprm"][:, 2] = torch.tensor([1.0, 2.0])
    self.model.actuator_gainprm[:] = self._defaults["actuator_gainprm"]
    self.model.actuator_biasprm[:] = self._defaults["actuator_biasprm"]
    self.model.dof_damping[:] = self._defaults["dof_damping"]

  def get_default_field(self, field: str) -> torch.Tensor:
    return self._defaults[field]


class _FakeEnv:
  def __init__(self) -> None:
    self.device = "cpu"
    self.num_envs = 3
    self.sim = _FakeSim()
    self.wheel_joint_ids = [0, 2]


class SkaterDomainRandomizationTests(unittest.TestCase):
  def test_paper_randomization_terms_run_on_reset(self) -> None:
    bootstrap_task_registry()
    env_cfg = load_env_cfg("Mjlab-Roller-Joystick-Flat-Unitree-G1")

    for name in (
      "link_mass",
      "torso_com",
      "left_skate_com",
      "right_skate_com",
      "wheel_static_friction",
      "wheel_dynamic_friction",
      "actuator_and_wheel_damping",
    ):
      self.assertEqual(env_cfg.events[name].mode, "reset")

    self.assertEqual(env_cfg.events["link_mass"].params["ranges"], (0.9, 1.1))
    self.assertEqual(
      env_cfg.events["wheel_static_friction"].params["ranges"],
      (0.1, 0.8),
    )
    self.assertEqual(
      env_cfg.events["wheel_dynamic_friction"].params["ranges"],
      (0.1, 0.4),
    )
    self.assertTrue(env_cfg.events["wheel_static_friction"].params["shared_random"])
    self.assertTrue(env_cfg.events["wheel_dynamic_friction"].params["shared_random"])
    self.assertEqual(
      env_cfg.events["actuator_and_wheel_damping"].params["actuator_stiffness_scale"],
      (0.9, 1.1),
    )
    self.assertEqual(
      env_cfg.events["actuator_and_wheel_damping"].params["actuator_damping_scale"],
      (0.9, 1.1),
    )
    self.assertEqual(
      env_cfg.events["actuator_and_wheel_damping"].params["wheel_joint_damping"],
      (0.002, 0.005),
    )

  def test_actuator_and_wheel_damping_respects_env_ids(self) -> None:
    torch.manual_seed(0)
    env = _FakeEnv()

    randomize_actuator_and_wheel_damping(env, env_ids=torch.tensor([1]))

    default_gain = env.sim.get_default_field("actuator_gainprm")
    default_bias = env.sim.get_default_field("actuator_biasprm")
    default_damping = env.sim.get_default_field("dof_damping")
    wheel_dof_ids = torch.tensor([0, 3], dtype=torch.long)

    self.assertTrue(torch.allclose(env.sim.model.actuator_gainprm[0], default_gain))
    self.assertTrue(torch.allclose(env.sim.model.actuator_gainprm[2], default_gain))
    self.assertTrue(torch.allclose(env.sim.model.actuator_biasprm[0], default_bias))
    self.assertTrue(torch.allclose(env.sim.model.actuator_biasprm[2], default_bias))
    self.assertTrue(torch.allclose(env.sim.model.dof_damping[0], default_damping))
    self.assertTrue(torch.allclose(env.sim.model.dof_damping[2], default_damping))

    scaled_gain = env.sim.model.actuator_gainprm[1, :, 0]
    scaled_bias = env.sim.model.actuator_biasprm[1, :, 2]
    self.assertTrue(torch.all(scaled_gain >= default_gain[:, 0] * 0.9))
    self.assertTrue(torch.all(scaled_gain <= default_gain[:, 0] * 1.1))
    self.assertTrue(torch.all(scaled_bias >= default_bias[:, 2] * 0.9))
    self.assertTrue(torch.all(scaled_bias <= default_bias[:, 2] * 1.1))

    self.assertTrue(torch.all(env.sim.model.dof_damping[1, wheel_dof_ids] >= 0.002))
    self.assertTrue(torch.all(env.sim.model.dof_damping[1, wheel_dof_ids] <= 0.005))
    self.assertTrue(
      torch.allclose(
        env.sim.model.dof_damping[1, torch.tensor([1, 2, 4], dtype=torch.long)],
        default_damping[torch.tensor([1, 2, 4], dtype=torch.long)],
      )
    )


if __name__ == "__main__":
  unittest.main()
