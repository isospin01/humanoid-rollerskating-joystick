from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_roller.envs import G1SkaterJoystickManagerBasedRlEnv


def base_lin_vel_b(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_base_lin_vel_b()


def base_ang_vel_b(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_base_ang_vel_b()


def joint_pos_rel_reference(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_controlled_joint_pos() - env.get_reference_joint_pos()


def joint_vel_rel_controlled(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_controlled_joint_vel()


def left_skate_vel_local(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_skate_body_vel_local()[:, 0, :]


def right_skate_vel_local(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_skate_body_vel_local()[:, 1, :]


def left_skate_ang_vel_local(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_skate_body_ang_vel_local()[:, 0, :]


def right_skate_ang_vel_local(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_skate_body_ang_vel_local()[:, 1, :]


def contact_forces(
  env: G1SkaterJoystickManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  forces_flat = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def wheel_contact_summary(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.wheel_contact_filt.float()


def skate_separation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return env.get_skate_separation()
