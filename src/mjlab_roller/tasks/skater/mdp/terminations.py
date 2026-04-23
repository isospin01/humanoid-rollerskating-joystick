from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_roller.envs import G1SkaterJoystickManagerBasedRlEnv


def illegal_contact(env: G1SkaterJoystickManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return torch.any(sensor.data.found, dim=-1)


def bad_skate_contact_loss(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(env.get_skate_contact(), dim=-1) == 0


def excessive_boot_scrape(
  env: G1SkaterJoystickManagerBasedRlEnv,
  threshold_s: float,
) -> torch.Tensor:
  return torch.any(env.get_boot_scrape_time() > threshold_s, dim=-1)


def excessive_lateral_slip(
  env: G1SkaterJoystickManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  lateral_speed = torch.abs(env.get_skate_body_vel_local()[..., 1])
  return torch.max(lateral_speed, dim=-1).values > threshold
