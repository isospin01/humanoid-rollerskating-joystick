from __future__ import annotations

import torch


def randomize_actuator_and_wheel_damping(
  env,
  env_ids=None,
  *,
  actuator_stiffness_scale: tuple[float, float] = (0.9, 1.1),
  actuator_damping_scale: tuple[float, float] = (0.9, 1.1),
  wheel_joint_damping: tuple[float, float] = (0.002, 0.005),
) -> None:
  model = env.sim.model
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
  if env_ids.numel() == 0:
    return

  default_gainprm = env.sim.get_default_field("actuator_gainprm").to(
    device=model.actuator_gainprm.device,
    dtype=model.actuator_gainprm.dtype,
  )
  default_biasprm = env.sim.get_default_field("actuator_biasprm").to(
    device=model.actuator_biasprm.device,
    dtype=model.actuator_biasprm.dtype,
  )
  default_dof_damping = env.sim.get_default_field("dof_damping").to(
    device=model.dof_damping.device,
    dtype=model.dof_damping.dtype,
  )

  def _sample_uniform(low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
    return low + (high - low) * torch.rand(
      shape, device=env.device, dtype=default_gainprm.dtype
    )

  nu = int(model.nu)
  if nu > 0:
    model.actuator_gainprm[env_ids] = default_gainprm.unsqueeze(0)
    model.actuator_biasprm[env_ids] = default_biasprm.unsqueeze(0)
    kp_scale = _sample_uniform(*actuator_stiffness_scale, (env_ids.numel(), nu))
    kd_scale = _sample_uniform(*actuator_damping_scale, (env_ids.numel(), nu))
    model.actuator_gainprm[env_ids, :, 0] = default_gainprm[:, 0].unsqueeze(0) * kp_scale
    model.actuator_biasprm[env_ids, :, 2] = default_biasprm[:, 2].unsqueeze(0) * kd_scale

  wheel_joint_ids = torch.as_tensor(
    env.wheel_joint_ids, device=env.device, dtype=torch.long
  )
  if wheel_joint_ids.numel() == 0:
    return
  wheel_dof_ids = torch.as_tensor(
    env.sim.mj_model.jnt_dofadr[wheel_joint_ids.cpu().numpy()],
    device=env.device,
    dtype=torch.long,
  )
  model.dof_damping[env_ids] = default_dof_damping.unsqueeze(0)
  model.dof_damping[env_ids.unsqueeze(1), wheel_dof_ids.unsqueeze(0)] = _sample_uniform(
    *wheel_joint_damping, (env_ids.numel(), wheel_dof_ids.numel())
  )
