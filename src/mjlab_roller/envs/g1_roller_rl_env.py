from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.utils.lab_api.math import subtract_frame_transforms
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from prettytable import PrettyTable

from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES
from mjlab_roller.core.project_paths import data_path


@dataclass(kw_only=True)
class G1RollerManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):
  push_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  glide_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  steer_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  transition_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  regularization_rewards: dict[str, RewardTermCfg] = field(default_factory=dict)

  cycle_time: float = 6.0
  phase_ratios: list[float] = field(default_factory=list)
  beizer_names: list[str] = field(default_factory=list)
  slerp_names: list[str] = field(default_factory=list)
  transition_bezier_offset: float = 0.05
  glide_init_pos: list[float] = field(default_factory=list)
  steer_init_pos: list[float] = field(default_factory=list)
  eval_mode: bool = False
  eval_output_dir: str | None = None


class G1RollerManagerBasedRlEnv(ManagerBasedRlEnv):
  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: G1RollerManagerBasedRlEnvCfg  # type: ignore[assignment]

  def __init__(
    self,
    cfg: G1RollerManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
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
    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
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

    self.cycle_time = self.cfg.cycle_time
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
    self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore

    self.load_managers()
    self.setup_manager_visualizers()

  def _init_buffers(self):
    self._init_ids_buffers()
    self.phase_ratios = torch.tensor(
      self.cfg.phase_ratios, device=self.device, dtype=torch.float32
    ).repeat(self.num_envs, 1)
    self.glide_init_pos = torch.tensor(
      self.cfg.glide_init_pos, device=self.device, dtype=torch.float32
    ).repeat(self.num_envs, 1)
    self.steer_init_pos = torch.tensor(
      self.cfg.steer_init_pos, device=self.device, dtype=torch.float32
    ).repeat(self.num_envs, 1)

    self.last_left_wheel_contacts = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.last_right_wheel_contacts = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.left_wheel_contact_filt = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.right_wheel_contact_filt = torch.zeros(
      self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.wheel_contact_filt = torch.zeros(
      self.num_envs, 8, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.last_left_boot_scrape = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.last_right_boot_scrape = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.contact_phase = torch.zeros(
      self.num_envs, 6, dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.last_contact_phase = torch.zeros(
      self.num_envs, 6, dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.amp_active_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.steer_phase_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.transition_active_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_entered_push2glide = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_entered_glide2steer = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_entered_steer2push = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_exited_push2glide = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_exited_glide2steer = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.just_exited_steer2push = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )
    self.left_boot_scrape_time = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.right_boot_scrape_time = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.phase_length_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
    )
    self.still = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
    )

    transition_body_count = len(self.transition_body_ids)

    def _load_transition_ref_pose(filename: str) -> torch.Tensor:
      pose = torch.from_numpy(np.load(data_path("ref_pose", filename))).to(
        self.device, dtype=torch.float32
      )
      if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(
          f"Transition reference pose {filename} must have shape (num_bodies, 7), got {tuple(pose.shape)}"
        )
      if pose.shape[0] != transition_body_count:
        raise ValueError(
          "Transition reference pose body count does not match the robot subset "
          f"used for transition targets: expected {transition_body_count}, got {pose.shape[0]} "
          f"for {filename}"
        )
      return pose.repeat(self.num_envs, 1, 1)

    # Transition reference poses exclude the inline skates and wheels, so we only
    # keep a matching 30-body subset in the cached start/goal pose buffers.
    push_init_body_pose = _load_transition_ref_pose("push_start_pose_b.npy")
    glide_init_body_pose = _load_transition_ref_pose("glide_start_pose_b.npy")
    steer_init_body_pose = _load_transition_ref_pose("steer_start_pose_b.npy")
    self.push_init_body_pos_b = push_init_body_pose[..., :3]
    self.glide_init_body_pos_b = glide_init_body_pose[..., :3]
    self.steer_init_body_pos_b = steer_init_body_pose[..., :3]
    self.push_init_body_quat_b = push_init_body_pose[..., 3:]
    self.glide_init_body_quat_b = glide_init_body_pose[..., 3:]
    self.steer_init_body_quat_b = steer_init_body_pose[..., 3:]
    self.body_bezier_buffers = {
      "push2glide_start_pos_b": torch.zeros(
        self.num_envs, transition_body_count, 3, device=self.device, requires_grad=False
      ),
      "glide2steer_start_pos_b": torch.zeros(
        self.num_envs, transition_body_count, 3, device=self.device, requires_grad=False
      ),
      "steer2push_start_pos_b": torch.zeros(
        self.num_envs, transition_body_count, 3, device=self.device, requires_grad=False
      ),
      "push2glide_start_quat_b": torch.zeros(
        self.num_envs, transition_body_count, 4, device=self.device, requires_grad=False
      ),
      "glide2steer_start_quat_b": torch.zeros(
        self.num_envs, transition_body_count, 4, device=self.device, requires_grad=False
      ),
      "steer2push_start_quat_b": torch.zeros(
        self.num_envs, transition_body_count, 4, device=self.device, requires_grad=False
      ),
    }

  def _init_ids_buffers(self):
    self.controlled_joint_names = list(CONTROLLED_JOINT_NAMES)
    self.controlled_joint_ids, _ = self.robot.find_joints(
      name_keys=self.controlled_joint_names, preserve_order=True
    )
    self.feet_body_ids, _ = self.robot.find_bodies(
      name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
      preserve_order=True,
    )
    self.skate_body_ids, _ = self.robot.find_bodies(
      name_keys=["left_inline_skate", "right_inline_skate"],
      preserve_order=True,
    )
    self.left_wheel_body_ids, _ = self.robot.find_bodies(
      name_keys=["left_skate_wheel_1", "left_skate_wheel_2", "left_skate_wheel_3", "left_skate_wheel_4"],
      preserve_order=True,
    )
    self.right_wheel_body_ids, _ = self.robot.find_bodies(
      name_keys=["right_skate_wheel_1", "right_skate_wheel_2", "right_skate_wheel_3", "right_skate_wheel_4"],
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
    self.beizer_ids, _ = self.robot.find_bodies(
      name_keys=self.cfg.beizer_names, preserve_order=True
    )
    self.slerp_ids, _ = self.robot.find_bodies(
      name_keys=self.cfg.slerp_names, preserve_order=True
    )
    excluded_body_ids = {int(body_id) for body_id in self.skate_body_ids}
    excluded_body_ids.update(int(body_id) for body_id in self.left_wheel_body_ids)
    excluded_body_ids.update(int(body_id) for body_id in self.right_wheel_body_ids)
    self.transition_body_ids = [
      body_id
      for body_id in range(self.robot.num_bodies)
      if body_id not in excluded_body_ids
    ]

  def load_managers(self) -> None:
    super().load_managers()
    self.push_reward_manager = RewardManager(
      self.cfg.push_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.push_reward_manager}")
    self.glide_reward_manager = RewardManager(
      self.cfg.glide_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.glide_reward_manager}")
    self.steer_reward_manager = RewardManager(
      self.cfg.steer_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.steer_reward_manager}")
    self.transition_reward_manager = RewardManager(
      self.cfg.transition_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.transition_reward_manager}")
    self.reg_reward_manager = RewardManager(
      self.cfg.regularization_rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reg_reward_manager}")

  def get_heading_target_w(self, command_name: str) -> torch.Tensor | None:
    terms = getattr(self.command_manager, "_terms", None) or getattr(self.command_manager, "terms", None)
    if terms is None:
      return None
    term = terms.get(command_name)
    if term is None:
      return None
    return getattr(term, "target_heading_w", None)

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    self.action_manager.process_action(action.to(self.device))
    self.still = self.command_manager.get_command("skate")[:, 0] < 0.1  # pyright: ignore[reportOptionalSubscript]
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.phase_length_buf += 1
    self.common_step_counter += 1
    self._compute_contact()

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    push_reward_buf = self.push_reward_manager.compute(self.step_dt) * self.contact_phase[:, 0]
    glide_reward_buf = self.glide_reward_manager.compute(self.step_dt) * self.contact_phase[:, 2]
    steer_reward_buf = self.steer_reward_manager.compute(self.step_dt) * self.contact_phase[:, 4]
    transition_reward_buf = (
      self.transition_reward_manager.compute(self.step_dt)
      * (
        self.contact_phase[:, 1]
        + self.contact_phase[:, 3]
        + self.contact_phase[:, 5]
      ).clamp(max=1.0)
    )
    reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)
    self.reward_buf = (
      push_reward_buf
      + glide_reward_buf
      + steer_reward_buf
      + transition_reward_buf
      + reg_reward_buf
    )

    self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(self.reset_env_ids) > 0:
      self._reset_idx(self.reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.obs_buf = self.observation_manager.compute(update_history=True)
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    super()._reset_idx(env_ids)
    info = self.push_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.glide_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.steer_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.reg_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    info = self.transition_reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    self.phase_length_buf[env_ids] = 0
    self.amp_active_mask[env_ids] = False
    self.steer_phase_mask[env_ids] = False
    self.transition_active_mask[env_ids] = False
    self.left_boot_scrape_time[env_ids] = 0.0
    self.right_boot_scrape_time[env_ids] = 0.0
    for buf in self.body_bezier_buffers.values():
      buf[env_ids] = 0

  def _compute_contact(self):
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
    self._resample_contact_phases()

  def _resample_contact_phases(self):
    self.last_contact_phase = self.contact_phase.clone()
    phase = self._get_phase()

    push_phase = (phase >= self.phase_ratios[:, 0]) & (phase < self.phase_ratios[:, 1]) & ~self.still
    push2glide = (phase >= self.phase_ratios[:, 1]) & (phase < self.phase_ratios[:, 2]) & ~self.still
    glide_phase = (phase >= self.phase_ratios[:, 2]) & (phase < self.phase_ratios[:, 3]) & ~self.still
    glide2steer = (phase >= self.phase_ratios[:, 3]) & (phase < self.phase_ratios[:, 4]) & ~self.still
    steer_phase = (phase >= self.phase_ratios[:, 4]) & (phase < self.phase_ratios[:, 5]) & ~self.still
    steer2push = (phase >= self.phase_ratios[:, 5]) & (phase <= self.phase_ratios[:, 6]) & ~self.still

    self.contact_phase[:, 0] = push_phase.float()
    self.contact_phase[:, 1] = push2glide.float()
    self.contact_phase[:, 2] = glide_phase.float()
    self.contact_phase[:, 3] = glide2steer.float()
    self.contact_phase[:, 4] = steer_phase.float()
    self.contact_phase[:, 5] = steer2push.float()

    self.amp_active_mask = push_phase
    self.steer_phase_mask = steer_phase
    self.transition_active_mask = push2glide | glide2steer | steer2push

    self.just_entered_push2glide = push2glide & (self.last_contact_phase[:, 1] < 0.5)
    self.just_entered_glide2steer = glide2steer & (self.last_contact_phase[:, 3] < 0.5)
    self.just_entered_steer2push = steer2push & (self.last_contact_phase[:, 5] < 0.5)
    self.just_exited_push2glide = (self.last_contact_phase[:, 1] > 0.5) & ~push2glide
    self.just_exited_glide2steer = (self.last_contact_phase[:, 3] > 0.5) & ~glide2steer
    self.just_exited_steer2push = (self.last_contact_phase[:, 5] > 0.5) & ~steer2push

    body_pos_w = self.robot.data.body_link_pos_w
    body_quat_w = self.robot.data.body_link_quat_w
    root_pos_w = self.robot.data.root_link_pos_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    root_quat_w = self.robot.data.root_link_quat_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    body_pos_b, body_quat_b = subtract_frame_transforms(
      root_pos_w, root_quat_w, body_pos_w, body_quat_w
    )
    transition_body_pos_b = body_pos_b[:, self.transition_body_ids, :]
    transition_body_quat_b = body_quat_b[:, self.transition_body_ids, :]
    if self.just_entered_push2glide.any():
      self.body_bezier_buffers["push2glide_start_pos_b"][self.just_entered_push2glide] = transition_body_pos_b[self.just_entered_push2glide]
      self.body_bezier_buffers["push2glide_start_quat_b"][self.just_entered_push2glide] = transition_body_quat_b[self.just_entered_push2glide]
    if self.just_entered_glide2steer.any():
      self.body_bezier_buffers["glide2steer_start_pos_b"][self.just_entered_glide2steer] = transition_body_pos_b[self.just_entered_glide2steer]
      self.body_bezier_buffers["glide2steer_start_quat_b"][self.just_entered_glide2steer] = transition_body_quat_b[self.just_entered_glide2steer]
    if self.just_entered_steer2push.any():
      self.body_bezier_buffers["steer2push_start_pos_b"][self.just_entered_steer2push] = transition_body_pos_b[self.just_entered_steer2push]
      self.body_bezier_buffers["steer2push_start_quat_b"][self.just_entered_steer2push] = transition_body_quat_b[self.just_entered_steer2push]

  def _get_phase(self):
    self.phase_length_buf[self.still] = 0
    phase = ((self.phase_length_buf * self.step_dt / self.cycle_time)) % 1.0
    return torch.clamp(phase, 0.0, 1.0)

  def _steer_remaining_steps(self):
    phase = self._get_phase()
    steer_end_phase = self.phase_ratios[:, 5]
    remaining_phase = torch.where(
      phase < steer_end_phase,
      steer_end_phase - phase,
      1.0 - phase + steer_end_phase,
    )
    return remaining_phase * self.cycle_time / self.step_dt

  def _get_skate_contact(self) -> torch.Tensor:
    left_contact = torch.any(self.left_wheel_contact_filt, dim=-1)
    right_contact = torch.any(self.right_wheel_contact_filt, dim=-1)
    return torch.stack([left_contact, right_contact], dim=-1)

  def _get_boot_scrape(self) -> torch.Tensor:
    return torch.stack([self.left_boot_scrape, self.right_boot_scrape], dim=-1)

  def _get_boot_scrape_time(self) -> torch.Tensor:
    return torch.stack([self.left_boot_scrape_time, self.right_boot_scrape_time], dim=-1)

  def _get_skate_air_time(self) -> torch.Tensor:
    left_sensor = self.scene.sensors["left_skate_contact"]
    right_sensor = self.scene.sensors["right_skate_contact"]
    left_air_time = left_sensor.data.current_air_time
    right_air_time = right_sensor.data.current_air_time
    assert left_air_time is not None
    assert right_air_time is not None
    return torch.stack([left_air_time.mean(dim=1), right_air_time.mean(dim=1)], dim=-1)

  def _get_skate_marker_positions(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.marker_site_ids, :3]

  def _get_skate_separation(self) -> torch.Tensor:
    markers = self._get_skate_marker_positions()
    left_center = markers[:, [0, 1], :].mean(dim=1)
    right_center = markers[:, [2, 3], :].mean(dim=1)
    return right_center - left_center

  def _get_transition_target_b(self):
    phase = self._get_phase()
    push2glide = (phase > self.phase_ratios[:, 1]) & (phase < self.phase_ratios[:, 2]) & ~self.still
    glide2steer = (phase > self.phase_ratios[:, 3]) & (phase < self.phase_ratios[:, 4]) & ~self.still
    steer2push = (phase > self.phase_ratios[:, 5]) & (phase < self.phase_ratios[:, 6]) & ~self.still
    in_transition = push2glide | glide2steer | steer2push

    body_pos_w = self.robot.data.body_link_pos_w
    body_quat_w = self.robot.data.body_link_quat_w
    root_pos_w = self.robot.data.root_link_pos_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    root_quat_w = self.robot.data.root_link_quat_w[:, None, :].repeat(1, self.robot.num_bodies, 1)
    current_body_pos_b, current_body_quat_b = subtract_frame_transforms(
      root_pos_w, root_quat_w, body_pos_w, body_quat_w
    )
    target_pos_b = current_body_pos_b.clone()
    target_quat_b = current_body_quat_b.clone()

    t = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    if push2glide.any():
      t[push2glide] = self._transition_progress(push2glide, 1, 2)
      push2glide_target_pos = bezier_curve(
        self.body_bezier_buffers["push2glide_start_pos_b"][push2glide],
        self.glide_init_body_pos_b[push2glide],
        t[push2glide],
        offset=self.cfg.transition_bezier_offset,
      )
      push2glide_target_quat = quaternion_slerp(
        self.body_bezier_buffers["push2glide_start_quat_b"][push2glide],
        self.glide_init_body_quat_b[push2glide],
        t[push2glide],
      )
      push2glide_pos_view = target_pos_b[push2glide].clone()
      push2glide_pos_view[:, self.transition_body_ids, :] = push2glide_target_pos
      target_pos_b[push2glide] = push2glide_pos_view
      push2glide_quat_view = target_quat_b[push2glide].clone()
      push2glide_quat_view[:, self.transition_body_ids, :] = push2glide_target_quat
      target_quat_b[push2glide] = push2glide_quat_view
    if glide2steer.any():
      t[glide2steer] = self._transition_progress(glide2steer, 3, 4)
      glide2steer_target_pos = bezier_curve(
        self.body_bezier_buffers["glide2steer_start_pos_b"][glide2steer],
        self.steer_init_body_pos_b[glide2steer],
        t[glide2steer],
        offset=self.cfg.transition_bezier_offset,
      )
      glide2steer_target_quat = quaternion_slerp(
        self.body_bezier_buffers["glide2steer_start_quat_b"][glide2steer],
        self.steer_init_body_quat_b[glide2steer],
        t[glide2steer],
      )
      glide2steer_pos_view = target_pos_b[glide2steer].clone()
      glide2steer_pos_view[:, self.transition_body_ids, :] = glide2steer_target_pos
      target_pos_b[glide2steer] = glide2steer_pos_view
      glide2steer_quat_view = target_quat_b[glide2steer].clone()
      glide2steer_quat_view[:, self.transition_body_ids, :] = glide2steer_target_quat
      target_quat_b[glide2steer] = glide2steer_quat_view
    if steer2push.any():
      t[steer2push] = self._transition_progress(steer2push, 5, 6)
      steer2push_target_pos = bezier_curve(
        self.body_bezier_buffers["steer2push_start_pos_b"][steer2push],
        self.push_init_body_pos_b[steer2push],
        t[steer2push],
        offset=self.cfg.transition_bezier_offset,
      )
      steer2push_target_quat = quaternion_slerp(
        self.body_bezier_buffers["steer2push_start_quat_b"][steer2push],
        self.push_init_body_quat_b[steer2push],
        t[steer2push],
      )
      steer2push_pos_view = target_pos_b[steer2push].clone()
      steer2push_pos_view[:, self.transition_body_ids, :] = steer2push_target_pos
      target_pos_b[steer2push] = steer2push_pos_view
      steer2push_quat_view = target_quat_b[steer2push].clone()
      steer2push_quat_view[:, self.transition_body_ids, :] = steer2push_target_quat
      target_quat_b[steer2push] = steer2push_quat_view
    return target_pos_b, target_quat_b, in_transition

  def _transition_progress(self, mask: torch.Tensor, start_idx: int, end_idx: int) -> torch.Tensor:
    phase = self._get_phase()
    duration = (self.phase_ratios[:, end_idx] - self.phase_ratios[:, start_idx]).clamp(min=1e-5)
    progress = (phase - self.phase_ratios[:, start_idx]) / duration
    return torch.clamp(progress[mask], 0.0, 1.0)

  def get_amp_observations(self):
    return self.get_controlled_joint_pos()

  def get_controlled_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos[:, self.controlled_joint_ids]

  def get_controlled_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel[:, self.controlled_joint_ids]

  def get_controlled_default_joint_pos(self) -> torch.Tensor:
    return self.robot.data.default_joint_pos[:, self.controlled_joint_ids]

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


def bezier_curve(start_p, end_p, t, offset=0.05):
  middle_p = (start_p + end_p) / 2.0
  middle_p[..., 2] += offset
  t = torch.clamp(t, 0.0, 1.0).view(-1, 1, 1)
  return (1 - t) ** 2 * start_p + 2 * (1 - t) * t * middle_p + t ** 2 * end_p


def quaternion_slerp(q0, q1, t, shortestpath=True):
  if t.dim() == 1:
    t = t.view(-1, 1, 1)
  if t.shape[1] == 1 and q0.shape[1] > 1:
    t = t.repeat(1, q0.shape[1], 1)

  eps = 1e-6
  d = torch.sum(q0 * q1, dim=-1, keepdim=True)
  zero_mask = torch.isclose(t, torch.zeros_like(t), atol=eps)
  ones_mask = torch.isclose(t, torch.ones_like(t), atol=eps)
  dist_mask = torch.abs(torch.abs(d) - 1.0) < eps
  out = torch.zeros_like(q0)
  out[zero_mask.squeeze(-1)] = q0[zero_mask.squeeze(-1)]
  out[ones_mask.squeeze(-1)] = q1[ones_mask.squeeze(-1)]
  out[dist_mask.squeeze(-1)] = q0[dist_mask.squeeze(-1)]
  if shortestpath:
    q1 = torch.where(d < 0, -q1, q1)
    d = torch.abs(d)
  angle = torch.acos(torch.clamp(d, -1.0 + eps, 1.0 - eps))
  angle_mask = torch.abs(angle) < eps
  out[angle_mask.squeeze(-1)] = q0[angle_mask.squeeze(-1)]
  final_mask = ~(zero_mask | ones_mask | dist_mask | angle_mask)
  final_mask = final_mask.squeeze(-1)
  sin_angle = torch.sin(angle)
  weight0 = torch.sin((1.0 - t) * angle) / (sin_angle + eps)
  weight1 = torch.sin(t * angle) / (sin_angle + eps)
  result = weight0 * q0 + weight1 * q1
  result = result / (torch.norm(result, dim=-1, keepdim=True) + eps)
  out[final_mask] = result[final_mask]
  return out
