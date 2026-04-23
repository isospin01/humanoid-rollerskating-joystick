from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab_roller.envs.g1_skater_joystick_rl_env import G1SkaterJoystickManagerBasedRlEnv


def track_base_linear_velocity_xy(
  env: G1SkaterJoystickManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.robot.data.root_link_lin_vel_b[:, :2]
  error = torch.sum(torch.square(command[:, :2] - actual), dim=-1)
  return torch.exp(-error / std**2)


def track_base_yaw_rate(
  env: G1SkaterJoystickManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - env.robot.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def alive_bonus(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def base_ang_vel_xy_l2(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.robot.data.root_link_ang_vel_b[:, :2]), dim=-1)


def energy_consumption(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  joint_torque = env.get_controlled_joint_torque()
  if joint_torque is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.abs(joint_torque * env.get_controlled_joint_vel()), dim=-1)


def arms_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_controlled_default_joint_pos()
  return torch.sum(torch.abs(diff[:, env.arm_joint_ids]), dim=-1)


def waist_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_controlled_default_joint_pos()
  return torch.sum(torch.abs(diff[:, env.waist_joint_ids]), dim=-1)


def ankle_roll_deviation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  diff = env.get_controlled_joint_pos() - env.get_controlled_default_joint_pos()
  return torch.sum(torch.abs(diff[:, env.ankle_roll_joint_ids]), dim=-1)


def flat_orientation(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
  projected_gravity = quat_apply_inverse(env.robot.data.root_link_quat_w, gravity)
  target = gravity
  return torch.sum(torch.square(projected_gravity - target), dim=-1)


def base_height_error(env: G1SkaterJoystickManagerBasedRlEnv, target_height: float) -> torch.Tensor:
  return torch.square(env.robot.data.root_link_pos_w[:, 2] - target_height)


def feet_too_near(env: G1SkaterJoystickManagerBasedRlEnv, min_distance: float) -> torch.Tensor:
  feet_distance = env.get_skate_distance()
  return (feet_distance < min_distance).float()


def feet_too_far(env: G1SkaterJoystickManagerBasedRlEnv, max_distance: float) -> torch.Tensor:
  feet_distance = env.get_skate_distance()
  return (feet_distance > max_distance).float()


def wheel_axial_slip(env: G1SkaterJoystickManagerBasedRlEnv) -> torch.Tensor:
  left_skate_quat = env.robot.data.body_link_quat_w[:, env.skate_body_ids[0], :].unsqueeze(1).repeat(1, 4, 1)
  right_skate_quat = env.robot.data.body_link_quat_w[:, env.skate_body_ids[1], :].unsqueeze(1).repeat(1, 4, 1)
  left_vel = env.robot.data.body_link_lin_vel_w[:, env.left_wheel_body_ids, :]
  right_vel = env.robot.data.body_link_lin_vel_w[:, env.right_wheel_body_ids, :]
  left_local = quat_apply_inverse(left_skate_quat.reshape(-1, 4), left_vel.reshape(-1, 3)).view(env.num_envs, 4, 3)
  right_local = quat_apply_inverse(right_skate_quat.reshape(-1, 4), right_vel.reshape(-1, 3)).view(env.num_envs, 4, 3)
  return torch.sum(torch.abs(left_local[:, :, 1]), dim=-1) + torch.sum(torch.abs(right_local[:, :, 1]), dim=-1)


def wheel_air_time_penalty(env: G1SkaterJoystickManagerBasedRlEnv, min_contact_wheels: int) -> torch.Tensor:
  contact_count = torch.sum(env.wheel_contact_filt.float(), dim=-1)
  return (contact_count < float(min_contact_wheels)).float()


def leg_symmetry(
  env: G1SkaterJoystickManagerBasedRlEnv,
  pos_weight: float,
  vel_weight: float,
) -> torch.Tensor:
  left_pos = env.get_controlled_joint_pos()[:, env.left_leg_joint_ids]
  right_pos = env.get_controlled_joint_pos()[:, env.right_leg_joint_ids] * env.leg_symmetry_signs
  left_vel = env.get_controlled_joint_vel()[:, env.left_leg_joint_ids]
  right_vel = env.get_controlled_joint_vel()[:, env.right_leg_joint_ids] * env.leg_symmetry_signs
  return -(
    pos_weight * torch.mean(torch.square(left_pos - right_pos), dim=-1)
    + vel_weight * torch.mean(torch.square(left_vel - right_vel), dim=-1)
  )


def arm_symmetry(
  env: G1SkaterJoystickManagerBasedRlEnv,
  pos_weight: float,
  vel_weight: float,
) -> torch.Tensor:
  left_pos = env.get_controlled_joint_pos()[:, env.left_arm_joint_ids]
  right_pos = env.get_controlled_joint_pos()[:, env.right_arm_joint_ids] * env.arm_symmetry_signs
  left_vel = env.get_controlled_joint_vel()[:, env.left_arm_joint_ids]
  right_vel = env.get_controlled_joint_vel()[:, env.right_arm_joint_ids] * env.arm_symmetry_signs
  return -(
    pos_weight * torch.mean(torch.square(left_pos - right_pos), dim=-1)
    + vel_weight * torch.mean(torch.square(left_vel - right_vel), dim=-1)
  )


def undesired_contacts(env: G1SkaterJoystickManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  illegal = torch.any(sensor.data.found, dim=-1)
  scrape = torch.any(env._get_boot_scrape(), dim=-1)
  return torch.logical_or(illegal, scrape).float()
