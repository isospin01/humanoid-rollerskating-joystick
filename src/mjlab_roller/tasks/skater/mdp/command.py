from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_roller.teleop import PygameJoystickCommandInput

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer
  from mjlab_roller.envs import G1SkaterJoystickManagerBasedRlEnv


@dataclass(kw_only=True)
class SkaterJoystickCommandCfg(CommandTermCfg):
  command_source: Literal["sampled", "joystick"] = "sampled"
  joystick_backend: str = "pygame"
  joystick_deadzone: float = 0.10
  joystick_smoothing: float = 0.20
  clamp_negative_vx: bool = True

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
    self.env = env
    self.robot = env.robot
    self.command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_lin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self._teleop_input: PygameJoystickCommandInput | None = None
    if cfg.command_source == "joystick":
      if self.num_envs != 1:
        raise ValueError("Joystick command source only supports `num_envs=1`.")
      if cfg.joystick_backend != "pygame":
        raise ValueError(f"Unsupported joystick backend: {cfg.joystick_backend}")
      self._teleop_input = PygameJoystickCommandInput(
        max_vx=max(cfg.ranges.lin_vel_x[1], 0.0),
        max_vy=max(abs(cfg.ranges.lin_vel_y[0]), abs(cfg.ranges.lin_vel_y[1])),
        max_wz=max(abs(cfg.ranges.ang_vel_z[0]), abs(cfg.ranges.ang_vel_z[1])),
        deadzone=cfg.joystick_deadzone,
        smoothing=cfg.joystick_smoothing,
      )

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def set_stage(
    self,
    *,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    resampling_time_range: tuple[float, float],
  ) -> None:
    self.cfg.ranges = self.cfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=lin_vel_y,
      ang_vel_z=ang_vel_z,
    )
    self.cfg.resampling_time_range = resampling_time_range
    if self._teleop_input is not None:
      self._teleop_input.set_limits(
        max_vx=max(lin_vel_x[1], 0.0),
        max_vy=max(abs(lin_vel_y[0]), abs(lin_vel_y[1])),
        max_wz=max(abs(ang_vel_z[0]), abs(ang_vel_z[1])),
      )

  def _update_metrics(self):
    max_command_time = max(self.cfg.resampling_time_range[1], self.env.step_dt)
    max_command_step = max_command_time / self.env.step_dt
    lin_vel = self.env.get_base_lin_vel_b()[:, :2]
    ang_vel = self.env.get_base_ang_vel_b()[:, 2]
    self.metrics["error_lin"] += (
      torch.norm(self.command_b[:, :2] - lin_vel, dim=-1) / max_command_step
    )
    self.metrics["error_yaw"] += (
      torch.abs(self.command_b[:, 2] - ang_vel) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.command_source == "joystick":
      self.command_b[env_ids] = 0.0
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self._clamp_command(env_ids)

  def _update_command(self):
    if self.cfg.command_source != "joystick":
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
      self._clamp_command(env_ids)
      return
    assert self._teleop_input is not None
    command = torch.tensor(
      self._teleop_input.read_command(), device=self.device, dtype=torch.float32
    )
    self.command_b[0] = command
    self._clamp_command(torch.tensor([0], device=self.device, dtype=torch.long))

  def _clamp_command(self, env_ids: torch.Tensor) -> None:
    x_min, x_max = self.cfg.ranges.lin_vel_x
    y_min, y_max = self.cfg.ranges.lin_vel_y
    wz_min, wz_max = self.cfg.ranges.ang_vel_z
    self.command_b[env_ids, 0] = torch.clamp(self.command_b[env_ids, 0], x_min, x_max)
    self.command_b[env_ids, 1] = torch.clamp(self.command_b[env_ids, 1], y_min, y_max)
    self.command_b[env_ids, 2] = torch.clamp(self.command_b[env_ids, 2], wz_min, wz_max)
    if self.cfg.clamp_negative_vx:
      self.command_b[env_ids, 0] = torch.clamp_min(self.command_b[env_ids, 0], 0.0)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return
    pelvis_pos = self.robot.data.root_link_pos_w[batch].detach().cpu().numpy()
    goal_pos = pelvis_pos + torch.tensor(
      [
        self.command_b[batch, 0].item(),
        self.command_b[batch, 1].item(),
        0.0,
      ]
    ).cpu().numpy()
    visualizer.add_sphere(
      center=goal_pos,
      radius=0.08,
      color=(0.2, 0.8, 0.2, 1.0),
      label="joystick_command",
    )

  def close(self) -> None:
    if self._teleop_input is not None:
      self._teleop_input.close()
      self._teleop_input = None
