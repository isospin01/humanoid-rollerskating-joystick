from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_mul,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab_roller.envs import G1RollerManagerBasedRlEnv


def push_skate_lin_vel(
  env: G1RollerManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.robot.data.root_link_lin_vel_b
  lin_vel_error = (
    torch.square(command[:, 0] - actual[:, 0])
    + torch.square(actual[:, 1])
    + torch.square(actual[:, 2])
  )
  return torch.exp(-lin_vel_error / std**2)


def push_yaw_align(env: G1RollerManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.robot.data.heading_w
  if target_w is not None:
    yaw_diff = torch.abs(wrap_to_pi(target_w - heading_w))
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    yaw_diff = torch.abs(wrap_to_pi(command[:, 1] - heading_w))
  return torch.exp(-yaw_diff / std**2)


def push_single_support_air_time(
  env: G1RollerManagerBasedRlEnv,
  threshold_min: float,
  threshold_max: float,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  air_time = env._get_skate_air_time()
  in_range = (air_time > threshold_min) & (air_time < threshold_max)
  single_support = torch.sum(in_range, dim=-1) == 1
  contact = env._get_skate_contact()
  reward = single_support.float() * (torch.sum(contact, dim=-1) >= 1).float()
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (command[:, 0] > command_threshold).float()


def push_stance_skate_contact(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  contact = env._get_skate_contact()
  scrape = env._get_boot_scrape()
  return (torch.sum(contact, dim=-1) >= 1).float() * (~torch.any(scrape, dim=-1)).float()


def glide_dual_skate_contact(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  contact = env._get_skate_contact()
  scrape = env._get_boot_scrape()
  return (torch.sum(contact, dim=-1) == 2).float() * (~torch.any(scrape, dim=-1)).float()


def glide_speed_retention(
  env: G1RollerManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.robot.data.root_link_lin_vel_b[:, 0]
  return torch.exp(-torch.square(command[:, 0] - actual) / std**2)


def glide_skate_spacing(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  separation = env._get_skate_separation()
  target = torch.tensor([0.0, -0.22, 0.0], device=env.device)
  error = torch.sum(torch.square(separation - target), dim=-1)
  return torch.exp(-error / std**2)


def glide_lateral_stability(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  left_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[0], :]
  right_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[1], :]
  left_local = quat_apply_inverse(env.robot.data.root_link_quat_w, left_vel)
  right_local = quat_apply_inverse(env.robot.data.root_link_quat_w, right_vel)
  lateral_error = torch.square(left_local[:, 1]) + torch.square(right_local[:, 1])
  return torch.exp(-lateral_error / std**2)


def steer_dual_skate_contact(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  contact = env._get_skate_contact()
  scrape = env._get_boot_scrape()
  return (torch.sum(contact, dim=-1) == 2).float() * (~torch.any(scrape, dim=-1)).float()


def steer_joint_pos(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  dof_error = torch.mean(torch.square(env.get_controlled_joint_pos() - env.steer_init_pos), dim=1)
  return torch.exp(-dof_error / std**2)


def steer_track_heading(env: G1RollerManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.robot.data.heading_w
  if target_w is not None:
    error = wrap_to_pi(heading_w - target_w)
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = wrap_to_pi(heading_w - command[:, 1])
  return torch.exp(-torch.abs(error) / std**2)


def steer_lean_guide(env: G1RollerManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.robot.data.heading_w
  if target_w is not None:
    delta_theta = wrap_to_pi(target_w - heading_w)
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    delta_theta = wrap_to_pi(command[:, 1] - heading_w)
  vx = env.robot.data.root_link_lin_vel_b[:, 0]
  remaining_steps = env._steer_remaining_steps()
  delta_t = (remaining_steps * env.step_dt).clamp(min=0.5)
  lean_ref = torch.clamp((0.35 * delta_theta) / (vx * delta_t + 1e-6), -0.25, 0.25)
  skate_quat = env.robot.data.body_link_quat_w[:, env.skate_body_ids, :]
  roll, _, _ = euler_xyz_from_quat(skate_quat.reshape(-1, 4))
  roll = roll.view(env.num_envs, -1)
  lean = roll.mean(dim=-1)
  return torch.exp(-torch.abs(lean - lean_ref) / std**2)


def transition_body_pos_tracking(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  target_pos_b, _, in_transition = env._get_transition_target_b()
  body_pos_w = env.robot.data.body_link_pos_w[:, :, :3]
  root_pos_w = env.robot.data.root_link_pos_w[:, :3][:, None, :].repeat(1, env.robot.num_bodies, 1)
  root_quat_w = env.robot.data.root_link_quat_w[:, None, :].repeat(1, env.robot.num_bodies, 1)
  current_body_pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
  pos_error = (current_body_pos_b - target_pos_b)[:, env.beizer_ids, :]
  pos_error_norm = torch.sum(torch.square(pos_error), dim=-1)
  reward = torch.exp(-pos_error_norm.mean(dim=-1) / std**2)
  return torch.where(in_transition, reward, torch.zeros_like(reward))


def transition_body_rot_tracking(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  _, target_quat_b, in_transition = env._get_transition_target_b()
  body_quat_w = env.robot.data.body_link_quat_w[:, :, :4]
  root_quat_w = env.robot.data.root_link_quat_w[:, None, :].repeat(1, env.robot.num_bodies, 1)
  target_quat_w = quat_mul(root_quat_w, target_quat_b)
  quat_error = torch.square(
    quat_error_magnitude(target_quat_w[:, env.slerp_ids, :], body_quat_w[:, env.slerp_ids, :])
  )
  reward = torch.exp(-quat_error.mean(dim=-1) / std**2)
  return torch.where(in_transition, reward, torch.zeros_like(reward))


def transition_penalty_contact(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  contact = env._get_skate_contact()
  return (torch.sum(contact, dim=-1) > 0).float()


def reg_wheel_contact_coverage(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return (torch.sum(env.wheel_contact_filt, dim=1) >= 6).float()


def self_collision_cost(env: G1RollerManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.squeeze(-1)


def lean_flat(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  skate_quat = env.robot.data.body_link_quat_w[:, env.skate_body_ids, :]
  roll, _, _ = euler_xyz_from_quat(skate_quat.reshape(-1, 4))
  roll = roll.view(env.num_envs, -1)
  diff = torch.mean(torch.abs(roll), dim=-1)
  reward = torch.exp(-diff / std**2)
  return torch.where(env.steer_phase_mask, torch.zeros_like(reward), reward)


def stand_still(env: G1RollerManagerBasedRlEnv, std: float) -> torch.Tensor:
  dof_error = torch.mean(
    torch.square(env.get_controlled_joint_pos() - env.get_controlled_default_joint_pos()), dim=1
  )
  reward = torch.exp(-dof_error / std**2)
  return reward * env.still.float()


def boot_scrape_penalty(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return torch.any(env._get_boot_scrape(), dim=-1).float()


def controlled_joint_vel_l2(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(env.get_controlled_joint_vel()), dim=1)


def controlled_joint_acc_l2(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  joint_acc = env.get_controlled_joint_acc()
  if joint_acc is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(joint_acc), dim=1)


def controlled_joint_torques_l2(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  joint_torque = env.get_controlled_joint_torque()
  if joint_torque is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(joint_torque), dim=1)


def controlled_joint_pos_limits(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  joint_limits = env.get_controlled_soft_joint_pos_limits()
  if joint_limits is None:
    return torch.zeros(env.num_envs, device=env.device)
  joint_pos = env.get_controlled_joint_pos()
  lower = joint_limits[..., 0]
  upper = joint_limits[..., 1]
  lower_violation = torch.clamp(lower - joint_pos, min=0.0)
  upper_violation = torch.clamp(joint_pos - upper, min=0.0)
  return torch.sum(lower_violation + upper_violation, dim=1)
