"""Unitree G1 roller configuration and action scaling."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia_from_two_stage_planetary
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES

G1_XML: Path = Path(os.path.join(os.path.dirname(__file__), "xmls", "g1.xml"))
assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_g1_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_14, GEARS_7520_14)

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_22, GEARS_7520_22)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)

ACTUATOR_5020 = ElectricActuator(reflected_inertia=ARMATURE_5020, velocity_limit=37.0, effort_limit=25.0)
ACTUATOR_7520_14 = ElectricActuator(reflected_inertia=ARMATURE_7520_14, velocity_limit=32.0, effort_limit=88.0)
ACTUATOR_7520_22 = ElectricActuator(reflected_inertia=ARMATURE_7520_22, velocity_limit=20.0, effort_limit=139.0)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

PUSH_INIT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(-0.03, 0.1, 0.78),
  joint_pos={
    "left_knee_joint": 0.23,
    "left_ankle_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.7,
    "right_knee_joint": 1.17,
    "right_ankle_pitch_joint": -0.45,
    "left_shoulder_pitch_joint": -0.03,
    "left_shoulder_roll_joint": 0.45,
    "left_shoulder_yaw_joint": -0.21,
    "left_elbow_joint": 1.32,
    "right_shoulder_pitch_joint": -0.7,
    "right_shoulder_roll_joint": -0.845,
    "right_shoulder_yaw_joint": 0.83,
    "right_elbow_joint": 1.19,
  },
  joint_vel={".*": 0.0},
)

STANDING_SKATE_INIT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(-0.03, 0.0, 0.80),
  joint_pos={},
  joint_vel={".*": 0.0},
)

STANDING_SKATE_CONTROLLED_JOINT_POS: tuple[float, ...] = (
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,
  0.0, 0.0, 0.05,
  -0.10, 0.45, -0.20, 1.10,
  -0.10, -0.45, 0.20, 1.10,
)

STANDING_SKATE_INIT_KEYFRAME.joint_pos = dict(
  zip(CONTROLLED_JOINT_NAMES, STANDING_SKATE_CONTROLLED_JOINT_POS, strict=True)
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (1,)},
)

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_23dof_robot_cfg(
  init_state: EntityCfg.InitialStateCfg | None = None,
) -> EntityCfg:
  return EntityCfg(
    init_state=init_state or PUSH_INIT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_spec,
    articulation=G1_ARTICULATION,
  )


G1_23Dof_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in G1_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  assert actuator_cfg.effort_limit is not None
  for name in actuator_cfg.target_names_expr:
    G1_23Dof_ACTION_SCALE[name] = 0.25 * actuator_cfg.effort_limit / actuator_cfg.stiffness


__all__ = [
  "G1_23Dof_ACTION_SCALE",
  "PUSH_INIT_KEYFRAME",
  "STANDING_SKATE_CONTROLLED_JOINT_POS",
  "STANDING_SKATE_INIT_KEYFRAME",
  "get_g1_23dof_robot_cfg",
]
