"""Unitree G1 roller-skating environment configurations."""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_roller.assets.robots.roller.g1 import (
  G1_23Dof_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
)
from mjlab_roller.envs import G1RollerManagerBasedRlEnvCfg
from mjlab_roller.tasks.roller import mdp
from mjlab_roller.tasks.roller.mdp import SkateUniformVelocityCommandCfg
from mjlab_roller.tasks.roller.roller_env_cfg import make_g1_roller_env_cfg


def unitree_g1_roller_env_cfg(play: bool = False) -> G1RollerManagerBasedRlEnvCfg:
  cfg = make_g1_roller_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  # Roller/skate contact geometry needs a larger contact budget than the old default.
  cfg.sim.nconmax = 128
  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}
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
  robot_collision_cfg = ContactSensorCfg(
    name="robot_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
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
    robot_collision_cfg,
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
  assert isinstance(skate_cmd, SkateUniformVelocityCommandCfg)
  skate_cmd.viz.z_offset = 1.15

  cfg.beizer_names = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  ]
  cfg.slerp_names = cfg.beizer_names
  cfg.phase_ratios = [0.0, 0.30, 0.40, 0.65, 0.75, 0.95, 1.0]
  cfg.glide_init_pos = [
    -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,
    -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,
    0.0, 0.0, 0.05,
    -0.1, 0.45, -0.2, 1.1,
    -0.1, -0.45, 0.2, 1.1,
  ]
  cfg.steer_init_pos = [
    -0.15, 0.08, 0.05, 0.6, -0.42, 0.0,
    -0.15, -0.08, 0.05, 0.6, -0.42, 0.0,
    0.0, 0.0, 0.1,
    0.0, 0.55, -0.25, 0.55,
    0.0, -0.55, 0.25, 0.55,
  ]

  if play:
    cfg.episode_length_s = 60
    cfg.eval_mode = True
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    cfg.events.pop("push_robot", None)
  return cfg
