from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab_roller.envs.g1_skater_joystick_rl_env import (
    G1SkaterJoystickManagerBasedRlEnv,
    SkaterCurriculumStageCfg,
  )


@dataclass(kw_only=True)
class SkaterJoystickCommandCfg(CommandTermCfg):
  command_source: Literal["sampled", "joystick"] = "sampled"
  allow_reverse: bool = False

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 1.15
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: G1SkaterJoystickManagerBasedRlEnv) -> "SkaterJoystickCommand":
    return SkaterJoystickCommand(self, env)


class SkaterJoystickCommand(CommandTerm):
  cfg: SkaterJoystickCommandCfg

  def __init__(self, cfg: SkaterJoystickCommandCfg, env: G1SkaterJoystickManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.robot
    self.env = env
    self.command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.current_ranges = cfg.ranges
    self.metrics["error_lin_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw_rate"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def set_stage(self, stage: SkaterCurriculumStageCfg) -> None:
    self.current_ranges = SkaterJoystickCommandCfg.Ranges(
      lin_vel_x=stage.lin_vel_x,
      lin_vel_y=stage.lin_vel_y,
      ang_vel_z=stage.ang_vel_z,
    )
    self.cfg.resampling_time_range = stage.resampling_time_range

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max(max_command_time / self._env.step_dt, 1.0)
    lin_error = torch.norm(
      self.command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    )
    yaw_error = torch.abs(
      self.command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self.metrics["error_lin_xy"] += lin_error / max_command_step
    self.metrics["error_yaw_rate"] += yaw_error / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    command = torch.empty(len(env_ids), 3, device=self.device)
    command[:, 0].uniform_(*self.current_ranges.lin_vel_x)
    command[:, 1].uniform_(*self.current_ranges.lin_vel_y)
    command[:, 2].uniform_(*self.current_ranges.ang_vel_z)
    if not self.cfg.allow_reverse:
      command[:, 0].clamp_(min=0.0)
    self.command_b[env_ids] = command

  def _update_command(self) -> None:
    if self.cfg.command_source != "joystick" or self.env.command_source != "joystick":
      return
    external = self.env.external_command_buffer
    self.command_b[:, 0] = max(float(external[0].item()), 0.0)
    self.command_b[:, 1] = float(external[1].item())
    self.command_b[:, 2] = float(external[2].item())
