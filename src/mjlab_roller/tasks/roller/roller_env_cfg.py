import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES
from mjlab_roller.envs import G1RollerManagerBasedRlEnvCfg
from mjlab_roller.tasks.roller import mdp
from mjlab_roller.tasks.roller.mdp import SkateUniformVelocityCommandCfg


def make_g1_roller_env_cfg() -> G1RollerManagerBasedRlEnvCfg:
  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "skate"},
      scale=(2.0, 1.0),
    ),
    "heading": ObservationTermCfg(func=mdp.heading, scale=1.0 / math.pi),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel_controlled,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel_controlled,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(func=mdp.phase),
  }

  critic_terms = {
    **policy_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "heading_error": ObservationTermCfg(
      func=mdp.heading_error,
      params={"command_name": "skate"},
      scale=1.0 / math.pi,
    ),
    "left_skate_vel_local": ObservationTermCfg(func=mdp.left_skate_vel_local),
    "right_skate_vel_local": ObservationTermCfg(func=mdp.right_skate_vel_local),
    "left_skate_ang_vel_local": ObservationTermCfg(func=mdp.left_skate_ang_vel_local),
    "right_skate_ang_vel_local": ObservationTermCfg(func=mdp.right_skate_ang_vel_local),
    "skate_lean": ObservationTermCfg(func=mdp.skate_lean),
    "skate_marker_sep": ObservationTermCfg(func=mdp.skate_marker_sep),
    "trans_target_pos_b": ObservationTermCfg(func=mdp.trans_target_pos_b),
    "trans_target_quat_b": ObservationTermCfg(func=mdp.trans_target_quat_b),
    "left_skate_forces": ObservationTermCfg(
      func=mdp.contact_forces,
      params={"sensor_name": "left_skate_contact"},
    ),
    "right_skate_forces": ObservationTermCfg(
      func=mdp.contact_forces,
      params={"sensor_name": "right_skate_contact"},
    ),
    "left_boot_scrape_force": ObservationTermCfg(
      func=mdp.contact_forces,
      params={"sensor_name": "left_boot_scrape"},
    ),
    "right_boot_scrape_force": ObservationTermCfg(
      func=mdp.contact_forces,
      params={"sensor_name": "right_boot_scrape"},
    ),
    "wheel_contact_summary": ObservationTermCfg(func=mdp.wheel_contact_summary),
    "contact_phase": ObservationTermCfg(func=mdp.contact_phase),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "skate": SkateUniformVelocityCommandCfg(
      resampling_time_range=(20.0, 20.0),
      rel_standing_envs=0.0,
      rel_heading_envs=1.0,
      heading_command=True,
      debug_vis=True,
      ranges=SkateUniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.5),
        heading=(-math.pi / 4, math.pi / 4),
      ),
    )
  }

  events = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 10.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.25, 0.25)}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.01, 0.01),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    ),
    "left_skate_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("left_inline_skate",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.015, 0.015), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "right_skate_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("right_inline_skate",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.015, 0.015), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "robot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
        "operation": "scale",
        "field": "geom_friction",
        "ranges": (0.5, 1.5),
      },
    ),
    "skate_frame_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_skate_frame_collision",)),
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.4, 1.2),
      },
    ),
    "wheel_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_skate_wheel_.*_collision",)),
        "operation": "scale",
        "field": "geom_friction",
        "axes": [0, 2],
        "ranges": (0.6, 1.4),
      },
    ),
  }

  push_rewards = {
    "push_skate_lin_vel": RewardTermCfg(
      func=mdp.push_skate_lin_vel,
      weight=3.0,
      params={"command_name": "skate", "std": math.sqrt(0.25)},
    ),
    "push_yaw_align": RewardTermCfg(
      func=mdp.push_yaw_align,
      weight=1.0,
      params={"command_name": "skate", "std": math.sqrt(0.1)},
    ),
    "push_single_support_air_time": RewardTermCfg(
      func=mdp.push_single_support_air_time,
      weight=2.0,
      params={
        "threshold_min": 0.05,
        "threshold_max": 0.35,
        "command_name": "skate",
        "command_threshold": 0.1,
      },
    ),
    "push_stance_skate_contact": RewardTermCfg(func=mdp.push_stance_skate_contact, weight=1.0),
  }

  glide_rewards = {
    "glide_dual_skate_contact": RewardTermCfg(func=mdp.glide_dual_skate_contact, weight=2.5),
    "glide_speed_retention": RewardTermCfg(
      func=mdp.glide_speed_retention,
      weight=2.0,
      params={"command_name": "skate", "std": math.sqrt(0.2)},
    ),
    "glide_skate_spacing": RewardTermCfg(
      func=mdp.glide_skate_spacing,
      weight=1.5,
      params={"std": math.sqrt(0.05)},
    ),
    "glide_lateral_stability": RewardTermCfg(
      func=mdp.glide_lateral_stability,
      weight=1.0,
      params={"std": math.sqrt(0.03)},
    ),
  }

  steer_rewards = {
    "steer_dual_skate_contact": RewardTermCfg(func=mdp.steer_dual_skate_contact, weight=2.0),
    "steer_joint_pos": RewardTermCfg(
      func=mdp.steer_joint_pos,
      weight=1.5,
      params={"std": math.sqrt(0.2)},
    ),
    "steer_track_heading": RewardTermCfg(
      func=mdp.steer_track_heading,
      weight=5.0,
      params={"command_name": "skate", "std": math.sqrt(0.02)},
    ),
    "steer_lean_guide": RewardTermCfg(
      func=mdp.steer_lean_guide,
      weight=4.0,
      params={"command_name": "skate", "std": math.sqrt(0.02)},
    ),
  }

  transition_rewards = {
    "transition_body_pos_tracking": RewardTermCfg(
      func=mdp.transition_body_pos_tracking,
      weight=8.0,
      params={"std": math.sqrt(0.2)},
    ),
    "transition_body_rot_tracking": RewardTermCfg(
      func=mdp.transition_body_rot_tracking,
      weight=8.0,
      params={"std": math.sqrt(0.4)},
    ),
    "transition_penalty_contact": RewardTermCfg(
      func=mdp.transition_penalty_contact,
      weight=-0.1,
    ),
  }

  regularization_rewards = {
    "reg_wheel_contact_coverage": RewardTermCfg(func=mdp.reg_wheel_contact_coverage, weight=0.5),
    "dof_pos_limits": RewardTermCfg(func=mdp.controlled_joint_pos_limits, weight=-5.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.1),
    "joint_vel_l2": RewardTermCfg(func=mdp.controlled_joint_vel_l2, weight=-1e-3),
    "joint_acc_l2": RewardTermCfg(func=mdp.controlled_joint_acc_l2, weight=-2.5e-7),
    "joint_torques_l2": RewardTermCfg(func=mdp.controlled_joint_torques_l2, weight=-1e-6),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "robot_collision"},
    ),
    "lean_flat": RewardTermCfg(
      func=mdp.lean_flat,
      weight=2.0,
      params={"std": math.sqrt(0.03)},
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=1.0,
      params={"std": math.sqrt(0.1)},
    ),
    "boot_scrape_penalty": RewardTermCfg(func=mdp.boot_scrape_penalty, weight=-1.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "bad_skate_contact_loss": TerminationTermCfg(func=mdp.bad_skate_contact_loss),
    "excessive_boot_scrape": TerminationTermCfg(
      func=mdp.excessive_boot_scrape,
      params={"threshold_s": 0.2},
    ),
    "excessive_lateral_slip": TerminationTermCfg(
      func=mdp.excessive_lateral_slip,
      params={"threshold": 0.8},
    ),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "illegal_contact"},
    ),
  }

  return G1RollerManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane", terrain_generator=None),
      entities={},
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    terminations=terminations,
    curriculum={},
    push_rewards=push_rewards,
    glide_rewards=glide_rewards,
    steer_rewards=steer_rewards,
    transition_rewards=transition_rewards,
    regularization_rewards=regularization_rewards,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=4.0,
      elevation=-10.0,
      azimuth=210.0,
    ),
    sim=SimulationCfg(
      nconmax=128,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
    transition_bezier_offset=0.05,
  )
