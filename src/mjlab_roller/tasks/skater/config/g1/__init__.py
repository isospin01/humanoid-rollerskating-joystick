from mjlab_roller.envs import G1SkaterJoystickManagerBasedRlEnv
from mjlab_roller.tasks.registry import register_mjlab_task
from mjlab_roller.tasks.roller.rl import RollerOnPolicyRunner
from mjlab_roller.tasks.skater.rl import (
    SkaterAMPOnPolicyRunner,
    SkaterResidualAMPOnPolicyRunner,
)

from .env_cfgs import unitree_g1_skater_amp_env_cfg, unitree_g1_skater_env_cfg
from .rl_cfg import (
    unitree_g1_skater_amp_ppo_runner_cfg,
    unitree_g1_skater_ppo_runner_cfg,
    unitree_g1_skater_residual_amp_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Roller-Joystick-Flat-Unitree-G1",
  env_cls=G1SkaterJoystickManagerBasedRlEnv,
  env_cfg=unitree_g1_skater_env_cfg(),
  play_env_cfg=unitree_g1_skater_env_cfg(play=True),
  rl_cfg=unitree_g1_skater_ppo_runner_cfg(),
  runner_cls=RollerOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Roller-Joystick-AMP-Flat-Unitree-G1",
  env_cls=G1SkaterJoystickManagerBasedRlEnv,
  env_cfg=unitree_g1_skater_amp_env_cfg(),
  play_env_cfg=unitree_g1_skater_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_skater_amp_ppo_runner_cfg(),
  runner_cls=SkaterAMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Roller-Joystick-Residual-Amp-Flat-Unitree-G1",
  env_cls=G1SkaterJoystickManagerBasedRlEnv,
  env_cfg=unitree_g1_skater_amp_env_cfg(),
  play_env_cfg=unitree_g1_skater_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_skater_residual_amp_ppo_runner_cfg(),
  runner_cls=SkaterResidualAMPOnPolicyRunner,
)
