"""Unitree G1 joystick-skater environment configurations."""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_roller.assets.robots.roller.g1 import (
  G1_23Dof_ACTION_SCALE,
  STANDING_SKATE_INIT_KEYFRAME,
  get_g1_23dof_robot_cfg,
)
from mjlab_roller.envs import CurriculumStageCfg, G1SkaterJoystickManagerBasedRlEnvCfg
from mjlab_roller.tasks.skater import mdp
from mjlab_roller.tasks.skater.mdp.command import SkaterJoystickCommandCfg
from mjlab_roller.tasks.skater.skater_env_cfg import make_g1_skater_env_cfg


def unitree_g1_skater_amp_env_cfg(play: bool = False) -> G1SkaterJoystickManagerBasedRlEnvCfg:
  """Env config for AMP training with velocity curriculum (beta always 1.0).

  Stage thresholds are absolute training iterations. For resume ~59.5k + lerp warmup, stage 0 (0.3 m/s)
  must apply until ~70k so AMP+style is not thrown straight into stage-4 speeds (v2 lesson learned).
  """
  cfg = unitree_g1_skater_env_cfg(play=play)
  cfg.curriculum_stages = (
    CurriculumStageCfg(
      start_iteration=0,
      beta=1.0,
      lin_vel_x=(0.0, 0.3),
      lin_vel_y=(-0.05, 0.05),
      ang_vel_z=(-0.25, 0.25),
      resampling_time_range=(4.0, 6.0),
    ),
    CurriculumStageCfg(
      start_iteration=70_000,
      beta=1.0,
      lin_vel_x=(0.0, 0.5),
      lin_vel_y=(-0.08, 0.08),
      ang_vel_z=(-0.35, 0.35),
      resampling_time_range=(3.5, 5.5),
    ),
    CurriculumStageCfg(
      start_iteration=78_000,
      beta=1.0,
      lin_vel_x=(0.0, 0.8),
      lin_vel_y=(-0.12, 0.12),
      ang_vel_z=(-0.55, 0.55),
      resampling_time_range=(3.0, 5.0),
    ),
    CurriculumStageCfg(
      start_iteration=86_000,
      beta=1.0,
      lin_vel_x=(0.0, 1.1),
      lin_vel_y=(-0.15, 0.15),
      ang_vel_z=(-0.75, 0.75),
      resampling_time_range=(2.0, 4.0),
    ),
    CurriculumStageCfg(
      start_iteration=96_000,
      beta=1.0,
      lin_vel_x=(0.0, 1.5),
      lin_vel_y=(-0.20, 0.20),
      ang_vel_z=(-1.00, 1.00),
      resampling_time_range=(1.5, 3.0),
    ),
  )
  cfg.action_beta_max = 1.0
  return cfg


def unitree_g1_skater_env_cfg(play: bool = False) -> G1SkaterJoystickManagerBasedRlEnvCfg:
  cfg = make_g1_skater_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 128
  cfg.scene.entities = {
    "robot": get_g1_23dof_robot_cfg(init_state=STANDING_SKATE_INIT_KEYFRAME)
  }
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  left_skate_contact = ContactSensorCfg(
    name="left_skate_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^left_skate_wheel_[1-4]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  right_skate_contact = ContactSensorCfg(
    name="right_skate_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^right_skate_wheel_[1-4]_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    track_air_time=True,
  )
  left_boot_scrape = ContactSensorCfg(
    name="left_boot_scrape",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^left_skate_frame_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  right_boot_scrape = ContactSensorCfg(
    name="right_boot_scrape",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^right_skate_frame_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  illegal_contact_cfg = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_shin_collision|.*_linkage_brace_collision|.*_shoulder_yaw_collision|.*_elbow_yaw_collision|.*_wrist_collision|.*_hand_collision|pelvis_collision$",
      entity="robot",
    ),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (
    left_skate_contact,
    right_skate_contact,
    left_boot_scrape,
    right_boot_scrape,
    illegal_contact_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_23Dof_ACTION_SCALE

  skate_cmd = cfg.commands["skate"]
  assert isinstance(skate_cmd, SkaterJoystickCommandCfg)
  skate_cmd.viz.z_offset = 1.15

  if play:
    cfg.episode_length_s = 60.0
    cfg.eval_mode = True
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
  return cfg
