from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab_roller.envs import G1SkaterJoystickManagerBasedRlEnv


def linear_velocity_track(
  env: G1SkaterJoystickManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(torch.square(command[:, :2] - env.get_base_lin_vel_b()[:, :2]), dim=-1)
  return torch.exp(-error / (std**2))


def angular_velocity_track(
  env: G1SkaterJoystickManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - env.get_base_ang_vel_b()[:, 2])
  return torch.exp(-error / (std**2))


def alive_reward(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def base_ang_vel_xy_l2(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.get_base_ang_vel_b()[:, :2]), dim=-1)


def controlled_joint_vel_l2(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.get_controlled_joint_vel()), dim=-1)


def controlled_joint_acc_l2(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  joint_acc = env.get_controlled_joint_acc()
  if joint_acc is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(joint_acc), dim=-1)


def controlled_joint_pos_limits(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  joint_limits = env.get_controlled_soft_joint_pos_limits()
  if joint_limits is None:
    return torch.zeros(env.num_envs, device=env.device)
  joint_pos = env.get_controlled_joint_pos()
  lower = joint_limits[..., 0]
  upper = joint_limits[..., 1]
  lower_violation = torch.clamp(lower - joint_pos, min=0.0)
  upper_violation = torch.clamp(joint_pos - upper, min=0.0)
  return torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=-1)


def energy_consumption(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  joint_torque = env.get_controlled_joint_torque()
  if joint_torque is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.abs(joint_torque * env.get_controlled_joint_vel()), dim=-1)


def arms_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_reference_joint_pos()
  return torch.sum(torch.abs(diff[:, env.arm_obs_ids]), dim=-1)


def waist_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_reference_joint_pos()
  return torch.sum(torch.abs(diff[:, env.waist_obs_ids]), dim=-1)


def ankle_roll_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_reference_joint_pos()
  return torch.sum(torch.abs(diff[:, env.ankle_roll_obs_ids]), dim=-1)


def flat_orientation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
  projected_gravity = quat_apply_inverse(env.robot.data.root_link_quat_w, gravity_w)
  return torch.sum(torch.square(projected_gravity - gravity_w), dim=-1)


def base_height_error(
  env: G1SkaterJoystickManagerBasedRlEnv,
  target_height: float,
) -> torch.Tensor:
  return torch.square(env.get_base_height() - target_height)


def feet_too_near(
  env: G1SkaterJoystickManagerBasedRlEnv,
  min_distance: float,
) -> torch.Tensor:
  return (env.get_skate_center_distance() < min_distance).float()


def feet_too_far(
  env: G1SkaterJoystickManagerBasedRlEnv,
  max_distance: float,
) -> torch.Tensor:
  return (env.get_skate_center_distance() > max_distance).float()


def wheel_axial_slip(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.abs(env.get_wheel_axial_slip()), dim=-1)


def wheel_air_time_penalty(
  env: G1SkaterJoystickManagerBasedRlEnv,
  min_contact_wheels: int,
) -> torch.Tensor:
  return (env.get_wheel_contact_count() < float(min_contact_wheels)).float()


def leg_symmetry(
  env: G1SkaterJoystickManagerBasedRlEnv,
  pos_weight: float,
  vel_weight: float,
) -> torch.Tensor:
  joint_pos = env.get_controlled_joint_pos()
  joint_vel = env.get_controlled_joint_vel()
  left_pos = joint_pos[:, env.left_leg_obs_ids]
  right_pos = joint_pos[:, env.right_leg_obs_ids] * env.leg_symmetry_signs
  left_vel = joint_vel[:, env.left_leg_obs_ids]
  right_vel = joint_vel[:, env.right_leg_obs_ids] * env.leg_symmetry_signs
  return -(
    pos_weight * torch.mean(torch.square(left_pos - right_pos), dim=-1)
    + vel_weight * torch.mean(torch.square(left_vel - right_vel), dim=-1)
  )


def arm_symmetry(
  env: G1SkaterJoystickManagerBasedRlEnv,
  pos_weight: float,
  vel_weight: float,
) -> torch.Tensor:
  joint_pos = env.get_controlled_joint_pos()
  joint_vel = env.get_controlled_joint_vel()
  left_pos = joint_pos[:, env.left_arm_obs_ids]
  right_pos = joint_pos[:, env.right_arm_obs_ids] * env.arm_symmetry_signs
  left_vel = joint_vel[:, env.left_arm_obs_ids]
  right_vel = joint_vel[:, env.right_arm_obs_ids] * env.arm_symmetry_signs
  return -(
    pos_weight * torch.mean(torch.square(left_pos - right_pos), dim=-1)
    + vel_weight * torch.mean(torch.square(left_vel - right_vel), dim=-1)
  )


def undesired_contacts(
  env: G1SkaterJoystickManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  illegal = torch.any(sensor.data.found, dim=-1)
  scrape = torch.any(env.get_boot_scrape(), dim=-1)
  return torch.logical_or(illegal, scrape).float()


def bad_skate_double_air_penalty(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  """1.0 when no skate wheels contact terrain (same predicate as old hard termination).

  AMP v2: weight **-2.0** recommended (softer than -5; harsh penalties freeze exploration).
  """
  return (torch.sum(env.get_skate_contact(), dim=-1) == 0).float()


def excessive_lateral_slip_penalty(
  env: G1SkaterJoystickManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  """1.0 when max lateral skate speed exceeds threshold (same predicate as old hard termination)."""
  lateral_speed = torch.abs(env.get_skate_body_vel_local()[..., 1])
  return (torch.max(lateral_speed, dim=-1).values > threshold).float()
