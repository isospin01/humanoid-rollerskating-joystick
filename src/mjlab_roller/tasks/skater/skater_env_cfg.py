from __future__ import annotations

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

from mjlab_roller.assets.robots.roller.g1 import STANDING_SKATE_CONTROLLED_JOINT_POS
from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES
from mjlab_roller.envs import (
  CurriculumStageCfg,
  G1SkaterJoystickManagerBasedRlEnvCfg,
)
from mjlab_roller.tasks.skater import mdp
from mjlab_roller.tasks.skater.mdp.command import SkaterJoystickCommandCfg


def make_g1_skater_env_cfg() -> G1SkaterJoystickManagerBasedRlEnvCfg:
  policy_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "skate"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel_b,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_b,
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel_reference,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel_controlled,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "wheel_contact_summary": ObservationTermCfg(func=mdp.wheel_contact_summary),
    "skate_separation": ObservationTermCfg(func=mdp.skate_separation),
  }

  critic_terms = {
    **policy_terms,
    "left_skate_vel_local": ObservationTermCfg(func=mdp.left_skate_vel_local),
    "right_skate_vel_local": ObservationTermCfg(func=mdp.right_skate_vel_local),
    "left_skate_ang_vel_local": ObservationTermCfg(func=mdp.left_skate_ang_vel_local),
    "right_skate_ang_vel_local": ObservationTermCfg(func=mdp.right_skate_ang_vel_local),
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
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,
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
    "skate": SkaterJoystickCommandCfg(
      resampling_time_range=(4.0, 6.0),
      debug_vis=True,
      ranges=SkaterJoystickCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.3),
        lin_vel_y=(-0.05, 0.05),
        ang_vel_z=(-0.25, 0.25),
      ),
    )
  }

  events = {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.01, 0.01),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
      },
    ),
    "link_mass": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "scale",
        "field": "body_mass",
        "ranges": (0.9, 1.1),
      },
    ),
    "torso_com": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "left_skate_com": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("left_inline_skate",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "right_skate_com": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("right_inline_skate",)),
        "operation": "add",
        "field": "body_ipos",
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    "wheel_static_friction": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_skate_wheel_.*_collision",)),
        "operation": "abs",
        "field": "geom_friction",
        "axes": [0],
        "ranges": (0.1, 0.8),
        "shared_random": True,
      },
    ),
    "wheel_dynamic_friction": EventTermCfg(
      mode="reset",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_skate_wheel_.*_collision",)),
        "operation": "abs",
        "field": "geom_friction",
        "axes": [2],
        "ranges": (0.1, 0.4),
        "shared_random": True,
      },
    ),
    "actuator_and_wheel_damping": EventTermCfg(
      mode="reset",
      func=mdp.randomize_actuator_and_wheel_damping,
      domain_randomization=False,
      params={
        "actuator_stiffness_scale": (0.9, 1.1),
        "actuator_damping_scale": (0.9, 1.1),
        "wheel_joint_damping": (0.002, 0.005),
      },
    ),
  }

  rewards = {
    "linear_velocity_track": RewardTermCfg(
      func=mdp.linear_velocity_track,
      weight=3.2,
      params={"command_name": "skate", "std": 0.5},
    ),
    "angular_velocity_track": RewardTermCfg(
      func=mdp.angular_velocity_track,
      weight=1.2,
      params={"command_name": "skate", "std": 0.5},
    ),
    "alive_reward": RewardTermCfg(func=mdp.alive_reward, weight=0.15),
    "base_ang_vel_xy_l2": RewardTermCfg(func=mdp.base_ang_vel_xy_l2, weight=-0.05),
    "controlled_joint_vel_l2": RewardTermCfg(
      func=mdp.controlled_joint_vel_l2,
      weight=-0.001,
    ),
    "controlled_joint_acc_l2": RewardTermCfg(
      func=mdp.controlled_joint_acc_l2,
      weight=-2.5e-7,
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "controlled_joint_pos_limits": RewardTermCfg(
      func=mdp.controlled_joint_pos_limits,
      weight=-5.0,
    ),
    "energy_consumption": RewardTermCfg(
      func=mdp.energy_consumption,
      weight=-2e-5,
    ),
    "arms_deviation": RewardTermCfg(func=mdp.arms_deviation, weight=-0.4),
    "waist_deviation": RewardTermCfg(func=mdp.waist_deviation, weight=-2.0),
    "ankle_roll_deviation": RewardTermCfg(
      func=mdp.ankle_roll_deviation,
      weight=-0.2,
    ),
    "flat_orientation": RewardTermCfg(func=mdp.flat_orientation, weight=-7.0),
    "base_height_error": RewardTermCfg(
      func=mdp.base_height_error,
      weight=-2.0,
      params={"target_height": 0.80},
    ),
    "feet_too_near": RewardTermCfg(
      func=mdp.feet_too_near,
      weight=-1.0,
      params={"min_distance": 0.2},
    ),
    "feet_too_far": RewardTermCfg(
      func=mdp.feet_too_far,
      weight=-5.0,
      params={"max_distance": 0.5},
    ),
    "wheel_axial_slip": RewardTermCfg(func=mdp.wheel_axial_slip, weight=-0.1),
    "wheel_air_time_penalty": RewardTermCfg(
      func=mdp.wheel_air_time_penalty,
      weight=-1.0,
      params={"min_contact_wheels": 4},
    ),
    "leg_symmetry": RewardTermCfg(
      func=mdp.leg_symmetry,
      weight=0.5,
      params={"pos_weight": 1.0, "vel_weight": 0.05},
    ),
    "arm_symmetry": RewardTermCfg(
      func=mdp.arm_symmetry,
      weight=0.5,
      params={"pos_weight": 1.0, "vel_weight": 0.05},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "illegal_contact"},
    ),
    # Soft penalties (AMP v2): same predicates as removed hard terminations; weight -2.0/step when active.
    "bad_skate_double_air_penalty": RewardTermCfg(
      func=mdp.bad_skate_double_air_penalty,
      weight=-2.0,
    ),
    "excessive_lateral_slip_penalty": RewardTermCfg(
      func=mdp.excessive_lateral_slip_penalty,
      weight=-2.0,
      params={"threshold": 1.5},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "excessive_boot_scrape": TerminationTermCfg(
      func=mdp.excessive_boot_scrape,
      params={"threshold_s": 0.5},
    ),
  }

  curriculum_stages = (
    CurriculumStageCfg(
      start_iteration=0,
      beta=0.35,
      lin_vel_x=(0.0, 0.3),
      lin_vel_y=(-0.05, 0.05),
      ang_vel_z=(-0.25, 0.25),
      resampling_time_range=(4.0, 6.0),
    ),
    CurriculumStageCfg(
      start_iteration=5_000,
      beta=0.55,
      lin_vel_x=(0.0, 0.7),
      lin_vel_y=(-0.10, 0.10),
      ang_vel_z=(-0.50, 0.50),
      resampling_time_range=(3.0, 5.0),
    ),
    CurriculumStageCfg(
      start_iteration=15_000,
      beta=0.80,
      lin_vel_x=(0.0, 1.1),
      lin_vel_y=(-0.15, 0.15),
      ang_vel_z=(-0.75, 0.75),
      resampling_time_range=(2.0, 4.0),
    ),
    CurriculumStageCfg(
      start_iteration=30_000,
      beta=1.00,
      lin_vel_x=(0.0, 1.5),
      lin_vel_y=(-0.20, 0.20),
      ang_vel_z=(-1.00, 1.00),
      resampling_time_range=(1.5, 3.0),
    ),
  )

  return G1SkaterJoystickManagerBasedRlEnvCfg(
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
    rewards=rewards,
    standing_joint_pos=list(STANDING_SKATE_CONTROLLED_JOINT_POS),
    standing_height=0.80,
    curriculum_stages=curriculum_stages,
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
  )
