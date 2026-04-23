from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import torch
import warp as wp
from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES
from mjlab_roller.tasks.skater.mdp.command import SkaterJoystickCommand


@dataclass(frozen=True)
class CurriculumStageCfg:
  start_iteration: int
  beta: float
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  resampling_time_range: tuple[float, float]


@dataclass(kw_only=True)
class G1SkaterJoystickManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):
  rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  standing_joint_pos: list[float] = field(default_factory=list)
  standing_height: float = 0.80
  curriculum_stages: tuple[CurriculumStageCfg, ...] = field(default_factory=tuple)
  curriculum_rollout_horizon: int = 24
  command_name: str = "skate"
  command_axes: tuple[str, str, str] = ("vx", "vy", "wz")
  command_frame: str = "body"
  joystick_mapping: tuple[str, str, str] = ("left_y:vx", "left_x:vy", "right_x:wz")
  action_beta_max: float = 1.0
  eval_mode: bool = False


class G1SkaterJoystickManagerBasedRlEnv(ManagerBasedRlEnv):
  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: G1SkaterJoystickManagerBasedRlEnvCfg  # type: ignore[assignment]

  def __init__(
    self,
    cfg: G1SkaterJoystickManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    del kwargs
    self.cfg = cfg  # type: ignore[assignment]
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )
    self.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm", "dof_damping"))
    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    print_info("")
    table = PrettyTable()
    table.title = "Skater Joystick Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    self.robot = self.scene["robot"]
    self._init_buffers()

    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore[assignment]

    self.load_managers()
    self.setup_manager_visualizers()
    if self.cfg.eval_mode and self.cfg.curriculum_stages:
      self.set_curriculum_stage_by_index(len(self.cfg.curriculum_stages) - 1)
    else:
      self.set_curriculum_stage_by_index(0)

  def _init_buffers(self) -> None:
    self._init_ids_buffers()
    self.reference_joint_pos = torch.tensor(
      self.cfg.standing_joint_pos, device=self.device, dtype=torch.float32
    ).repeat(self.num_envs, 1)
    self.left_wheel_contact_filt = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.right_wheel_contact_filt = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.last_left_wheel_contacts = torch.zeros_like(self.left_wheel_contact_filt)
    self.last_right_wheel_contacts = torch.zeros_like(self.right_wheel_contact_filt)
    self.wheel_contact_filt = torch.zeros(
      self.num_envs, 8, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.left_boot_scrape = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.right_boot_scrape = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.last_left_boot_scrape = torch.zeros_like(self.left_boot_scrape)
    self.last_right_boot_scrape = torch.zeros_like(self.right_boot_scrape)
    self.left_boot_scrape_time = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.right_boot_scrape_time = torch.zeros_like(self.left_boot_scrape_time)
    self.current_action_beta = self.cfg.action_beta_max
    self.curriculum_stage = 0
    self.curriculum_iteration = 0
    self.amp_active_mask = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )

    self.command_axes = tuple(self.cfg.command_axes)
    self.command_frame = self.cfg.command_frame
    self.joystick_mapping = tuple(self.cfg.joystick_mapping)

    self.left_leg_obs_ids = (0, 1, 2, 3, 4, 5)
    self.right_leg_obs_ids = (6, 7, 8, 9, 10, 11)
    self.waist_obs_ids = (12, 13, 14)
    self.left_arm_obs_ids = (15, 16, 17, 18)
    self.right_arm_obs_ids = (19, 20, 21, 22)
    self.arm_obs_ids = self.left_arm_obs_ids + self.right_arm_obs_ids
    self.ankle_roll_obs_ids = (5, 11)
    self.leg_symmetry_signs = torch.tensor(
      [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
      device=self.device,
      dtype=torch.float32,
    ).view(1, -1)
    self.arm_symmetry_signs = torch.tensor(
      [1.0, -1.0, -1.0, 1.0],
      device=self.device,
      dtype=torch.float32,
    ).view(1, -1)

  def _init_ids_buffers(self) -> None:
    self.controlled_joint_names = list(CONTROLLED_JOINT_NAMES)
    self.controlled_joint_ids, _ = self.robot.find_joints(
      name_keys=self.controlled_joint_names, preserve_order=True
    )
    self.skate_body_ids, _ = self.robot.find_bodies(
      name_keys=["left_inline_skate", "right_inline_skate"],
      preserve_order=True,
    )
    self.left_wheel_body_ids, _ = self.robot.find_bodies(
      name_keys=[
        "left_skate_wheel_1",
        "left_skate_wheel_2",
        "left_skate_wheel_3",
        "left_skate_wheel_4",
      ],
      preserve_order=True,
    )
    self.right_wheel_body_ids, _ = self.robot.find_bodies(
      name_keys=[
        "right_skate_wheel_1",
        "right_skate_wheel_2",
        "right_skate_wheel_3",
        "right_skate_wheel_4",
      ],
      preserve_order=True,
    )
    self.marker_site_ids, _ = self.robot.find_sites(
      name_keys=[
        "left_skate_front_marker",
        "left_skate_rear_marker",
        "right_skate_front_marker",
        "right_skate_rear_marker",
      ],
      preserve_order=True,
    )
    self.wheel_joint_ids, _ = self.robot.find_joints(
      name_keys=[
        "left_skate_wheel_1_joint",
        "left_skate_wheel_2_joint",
        "left_skate_wheel_3_joint",
        "left_skate_wheel_4_joint",
        "right_skate_wheel_1_joint",
        "right_skate_wheel_2_joint",
        "right_skate_wheel_3_joint",
        "right_skate_wheel_4_joint",
      ],
      preserve_order=True,
    )

  def load_managers(self) -> None:
    super().load_managers()
    self.reward_manager = RewardManager(
      self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reward_manager}")

  def set_curriculum_rollout_horizon(self, num_steps_per_env: int) -> None:
    self.cfg.curriculum_rollout_horizon = max(1, int(num_steps_per_env))

  def _resolve_curriculum_stage(self, learning_iteration: int) -> int:
    stage_index = 0
    for index, stage in enumerate(self.cfg.curriculum_stages):
      if learning_iteration >= stage.start_iteration:
        stage_index = index
    return stage_index

  def set_curriculum_stage(self, learning_iteration: int) -> None:
    self.curriculum_iteration = int(learning_iteration)
    # Keep internal progress counter aligned when restoring from checkpoints.
    # Otherwise `_update_curriculum_from_progress()` can immediately overwrite
    # the resumed stage based on a stale `common_step_counter`.
    self.common_step_counter = self.curriculum_iteration * max(
      1, self.cfg.curriculum_rollout_horizon
    )
    self.set_curriculum_stage_by_index(self._resolve_curriculum_stage(learning_iteration))

  def set_curriculum_stage_by_index(self, stage_index: int) -> None:
    if not self.cfg.curriculum_stages:
      self.curriculum_stage = 0
      self.current_action_beta = self.cfg.action_beta_max
      return
    stage_index = int(max(0, min(stage_index, len(self.cfg.curriculum_stages) - 1)))
    self.curriculum_stage = stage_index
    stage = self.cfg.curriculum_stages[stage_index]
    self.current_action_beta = stage.beta
    terms = getattr(self.command_manager, "_terms", None) or getattr(
      self.command_manager, "terms", None
    )
    if terms is None:
      return
    command_term = terms.get(self.cfg.command_name)
    if isinstance(command_term, SkaterJoystickCommand):
      command_term.set_stage(
        lin_vel_x=stage.lin_vel_x,
        lin_vel_y=stage.lin_vel_y,
        ang_vel_z=stage.ang_vel_z,
        resampling_time_range=stage.resampling_time_range,
      )

  def _update_curriculum_from_progress(self) -> None:
    if self.cfg.eval_mode or not self.cfg.curriculum_stages:
      return
    learning_iteration = self.common_step_counter // max(
      1, self.cfg.curriculum_rollout_horizon
    )
    if learning_iteration != self.curriculum_iteration:
      self.set_curriculum_stage(learning_iteration)

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    self._update_curriculum_from_progress()
    self.action_manager.process_action(
      (self.current_action_beta * action).to(self.device)
    )
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self._compute_contact()

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(self.step_dt)

    self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(self.reset_env_ids) > 0:
      self._reset_idx(self.reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)
    self.obs_buf = self.observation_manager.compute(update_history=True)
    self.extras.setdefault("log", {})
    self.extras["log"]["Curriculum/stage"] = float(self.curriculum_stage)
    self.extras["log"]["Curriculum/beta"] = float(self.current_action_beta)
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    if env_ids is None:
      return
    self.left_wheel_contact_filt[env_ids] = False
    self.right_wheel_contact_filt[env_ids] = False
    self.last_left_wheel_contacts[env_ids] = False
    self.last_right_wheel_contacts[env_ids] = False
    self.wheel_contact_filt[env_ids] = False
    self.left_boot_scrape[env_ids] = False
    self.right_boot_scrape[env_ids] = False
    self.last_left_boot_scrape[env_ids] = False
    self.last_right_boot_scrape[env_ids] = False
    self.left_boot_scrape_time[env_ids] = 0.0
    self.right_boot_scrape_time[env_ids] = 0.0

  def _compute_contact(self) -> None:
    left_sensor = self.scene.sensors["left_skate_contact"]
    right_sensor = self.scene.sensors["right_skate_contact"]
    left_force = torch.norm(left_sensor.data.force, dim=-1)
    right_force = torch.norm(right_sensor.data.force, dim=-1)
    left_wheel_contact = left_force > 1.0
    right_wheel_contact = right_force > 1.0
    self.left_wheel_contact_filt = torch.logical_or(
      left_wheel_contact, self.last_left_wheel_contacts
    )
    self.right_wheel_contact_filt = torch.logical_or(
      right_wheel_contact, self.last_right_wheel_contacts
    )
    self.last_left_wheel_contacts = left_wheel_contact
    self.last_right_wheel_contacts = right_wheel_contact
    self.wheel_contact_filt = torch.cat(
      [self.left_wheel_contact_filt, self.right_wheel_contact_filt], dim=1
    )

    left_scrape_sensor = self.scene.sensors["left_boot_scrape"]
    right_scrape_sensor = self.scene.sensors["right_boot_scrape"]
    left_scrape = torch.any(left_scrape_sensor.data.found, dim=-1)
    right_scrape = torch.any(right_scrape_sensor.data.found, dim=-1)
    self.left_boot_scrape = torch.logical_or(left_scrape, self.last_left_boot_scrape)
    self.right_boot_scrape = torch.logical_or(right_scrape, self.last_right_boot_scrape)
    self.last_left_boot_scrape = left_scrape
    self.last_right_boot_scrape = right_scrape
    self.left_boot_scrape_time = torch.where(
      left_scrape,
      self.left_boot_scrape_time + self.step_dt,
      torch.zeros_like(self.left_boot_scrape_time),
    )
    self.right_boot_scrape_time = torch.where(
      right_scrape,
      self.right_boot_scrape_time + self.step_dt,
      torch.zeros_like(self.right_boot_scrape_time),
    )

  def get_export_metadata(self) -> dict[str, list | str | float]:
    max_beta = self.cfg.action_beta_max
    if self.cfg.curriculum_stages:
      max_beta = max(stage.beta for stage in self.cfg.curriculum_stages)
    return {
      "command_axes": list(self.command_axes),
      "command_frame": self.command_frame,
      "joystick_mapping": list(self.joystick_mapping),
      "action_beta_max": float(max_beta),
    }

  def get_amp_observations(self) -> torch.Tensor:
    return self.get_controlled_joint_pos()

  def get_reference_joint_pos(self) -> torch.Tensor:
    return self.reference_joint_pos

  def get_controlled_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self.controlled_joint_ids]

  def get_controlled_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self.controlled_joint_ids]

  def get_controlled_default_joint_pos(self) -> torch.Tensor:
    return self.get_reference_joint_pos()

  def get_controlled_joint_acc(self) -> torch.Tensor | None:
    joint_acc = getattr(self.robot.data, "joint_acc", None)
    if joint_acc is None:
      return None
    return joint_acc[:, self.controlled_joint_ids]

  def get_controlled_joint_torque(self) -> torch.Tensor | None:
    for field_name in ("joint_torque", "joint_torques", "applied_torque"):
      try:
        joint_torque = getattr(self.robot.data, field_name)
      except (AttributeError, NotImplementedError):
        joint_torque = None
      if joint_torque is not None:
        return joint_torque[:, self.controlled_joint_ids]
    return None

  def get_controlled_soft_joint_pos_limits(self) -> torch.Tensor | None:
    for field_name in ("soft_joint_pos_limits", "joint_pos_limits"):
      joint_limits = getattr(self.robot.data, field_name, None)
      if joint_limits is not None:
        return joint_limits[:, self.controlled_joint_ids]
    return None

  def get_base_lin_vel_b(self) -> torch.Tensor:
    for field_name in ("root_link_lin_vel_b", "root_lin_vel_b"):
      value = getattr(self.robot.data, field_name, None)
      if value is not None:
        return value
    raise AttributeError("Robot data does not expose a base linear velocity field.")

  def get_base_ang_vel_b(self) -> torch.Tensor:
    for field_name in ("root_link_ang_vel_b", "root_ang_vel_b", "root_link_ang_vel_w"):
      value = getattr(self.robot.data, field_name, None)
      if value is not None:
        return value
    raise AttributeError("Robot data does not expose a base angular velocity field.")

  def get_base_height(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w[:, 2]

  def get_skate_contact(self) -> torch.Tensor:
    left_contact = torch.any(self.left_wheel_contact_filt, dim=-1)
    right_contact = torch.any(self.right_wheel_contact_filt, dim=-1)
    return torch.stack([left_contact, right_contact], dim=-1)

  def _get_skate_contact(self) -> torch.Tensor:
    return self.get_skate_contact()

  def get_wheel_contact_count(self) -> torch.Tensor:
    return torch.sum(self.wheel_contact_filt, dim=1)

  def get_boot_scrape(self) -> torch.Tensor:
    return torch.stack([self.left_boot_scrape, self.right_boot_scrape], dim=-1)

  def _get_boot_scrape(self) -> torch.Tensor:
    return self.get_boot_scrape()

  def get_boot_scrape_time(self) -> torch.Tensor:
    return torch.stack([self.left_boot_scrape_time, self.right_boot_scrape_time], dim=-1)

  def _get_boot_scrape_time(self) -> torch.Tensor:
    return self.get_boot_scrape_time()

  def get_skate_marker_positions(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.marker_site_ids, :3]

  def get_skate_separation(self) -> torch.Tensor:
    markers = self.get_skate_marker_positions()
    left_center = markers[:, [0, 1], :].mean(dim=1)
    right_center = markers[:, [2, 3], :].mean(dim=1)
    return right_center - left_center

  def get_skate_center_distance(self) -> torch.Tensor:
    return torch.norm(self.get_skate_separation(), dim=-1)

  def get_skate_distance(self) -> torch.Tensor:
    return self.get_skate_center_distance()

  def get_skate_body_vel_local(self) -> torch.Tensor:
    skate_vel = self.robot.data.body_link_lin_vel_w[:, self.skate_body_ids, :]
    root_quat_w = self.robot.data.root_link_quat_w[:, None, :].repeat(1, 2, 1)
    return quat_apply_inverse(root_quat_w, skate_vel)

  def get_skate_body_ang_vel_local(self) -> torch.Tensor:
    skate_ang_vel = self.robot.data.body_link_ang_vel_w[:, self.skate_body_ids, :]
    root_quat_w = self.robot.data.root_link_quat_w[:, None, :].repeat(1, 2, 1)
    return quat_apply_inverse(root_quat_w, skate_ang_vel)

  def get_skate_roll(self) -> torch.Tensor:
    skate_quat = self.robot.data.body_link_quat_w[:, self.skate_body_ids, :]
    roll, _, _ = euler_xyz_from_quat(skate_quat.reshape(-1, 4))
    return roll.view(self.num_envs, 2)

  def get_wheel_axial_slip(self) -> torch.Tensor:
    left_wheel_vel = self.robot.data.body_link_lin_vel_w[:, self.left_wheel_body_ids, :]
    right_wheel_vel = self.robot.data.body_link_lin_vel_w[:, self.right_wheel_body_ids, :]
    left_quat = self.robot.data.body_link_quat_w[:, self.skate_body_ids[0], :][
      :, None, :
    ].repeat(1, 4, 1)
    right_quat = self.robot.data.body_link_quat_w[:, self.skate_body_ids[1], :][
      :, None, :
    ].repeat(1, 4, 1)
    left_local = quat_apply_inverse(left_quat, left_wheel_vel)
    right_local = quat_apply_inverse(right_quat, right_wheel_vel)
    return torch.cat([left_local[..., 1], right_local[..., 1]], dim=1)

  def close(self) -> None:
    terms = getattr(self.command_manager, "_terms", None) or getattr(
      self.command_manager, "terms", None
    )
    if terms is not None:
      for term in terms.values():
        close_fn = getattr(term, "close", None)
        if callable(close_fn):
          close_fn()
    if self._offline_renderer is not None:
      self._offline_renderer.close()
      self._offline_renderer = None
    super().close()
