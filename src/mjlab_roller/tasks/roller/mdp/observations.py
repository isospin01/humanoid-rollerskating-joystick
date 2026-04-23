from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab_roller.envs import G1RollerManagerBasedRlEnv


def heading(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env.robot.data.heading_w.unsqueeze(-1)


def heading_error(env: G1RollerManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  target_w = env.get_heading_target_w(command_name)
  heading_w = env.robot.data.heading_w
  if target_w is not None:
    error = wrap_to_pi(target_w - heading_w)
  else:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = wrap_to_pi(command[:, 1] - heading_w)
  return error.unsqueeze(-1)


def contact_phase(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env.contact_phase.clone()


def phase(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env._get_phase().unsqueeze(-1)


def joint_pos_rel_controlled(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env.get_controlled_joint_pos() - env.get_controlled_default_joint_pos()


def joint_vel_rel_controlled(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env.get_controlled_joint_vel()


def left_skate_vel_local(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  skate_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[0], :]
  return quat_apply_inverse(env.robot.data.root_link_quat_w, skate_vel)


def right_skate_vel_local(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  skate_vel = env.robot.data.body_link_lin_vel_w[:, env.skate_body_ids[1], :]
  return quat_apply_inverse(env.robot.data.root_link_quat_w, skate_vel)


def left_skate_ang_vel_local(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  skate_ang_vel = env.robot.data.body_link_ang_vel_w[:, env.skate_body_ids[0], :]
  return quat_apply_inverse(env.robot.data.root_link_quat_w, skate_ang_vel)


def right_skate_ang_vel_local(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  skate_ang_vel = env.robot.data.body_link_ang_vel_w[:, env.skate_body_ids[1], :]
  return quat_apply_inverse(env.robot.data.root_link_quat_w, skate_ang_vel)


def skate_lean(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  quats = env.robot.data.body_link_quat_w[:, env.skate_body_ids, :]
  roll, _, _ = euler_xyz_from_quat(quats.reshape(-1, 4))
  return roll.view(env.num_envs, -1)


def trans_target_pos_b(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  target_pos_b, _, _ = env._get_transition_target_b()
  return target_pos_b[:, env.beizer_ids, :].view(env.num_envs, -1)


def trans_target_quat_b(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  _, target_quat_b, _ = env._get_transition_target_b()
  return target_quat_b[:, env.slerp_ids, :].view(env.num_envs, -1)


def contact_forces(env: G1RollerManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  forces_flat = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def wheel_contact_summary(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env.wheel_contact_filt.float()


def skate_marker_sep(env: G1RollerManagerBasedRlEnv) -> torch.Tensor:
  return env._get_skate_separation()
