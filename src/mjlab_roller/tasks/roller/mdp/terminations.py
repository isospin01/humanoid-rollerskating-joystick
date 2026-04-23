from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab_roller.envs import G1RollerManagerBasedRlEnv


def illegal_contact(env: G1RollerManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return torch.any(sensor.data.found, dim=-1)


def bad_skate_contact_loss(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  contact = env._get_skate_contact()
  return torch.sum(contact, dim=-1) == 0


def excessive_boot_scrape(env: G1RollerManagerBasedRlEnv, threshold_s: float) -> torch.Tensor:
  return torch.any(env._get_boot_scrape_time() > threshold_s, dim=-1)


def excessive_lateral_slip(env: G1RollerManagerBasedRlEnv, threshold: float) -> torch.Tensor:
  left_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[0], :]
  right_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[1], :]
  left_local = quat_apply_inverse(env.robot.data.body_link_quat_w[:, env.skate_body_ids[0], :], left_vel)
  right_local = quat_apply_inverse(env.robot.data.body_link_quat_w[:, env.skate_body_ids[1], :], right_vel)
  left_local = left_local[:, 1].abs()
  right_local = right_local[:, 1].abs()
  return torch.maximum(left_local, right_local) > threshold
