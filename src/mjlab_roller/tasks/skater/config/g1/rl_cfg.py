"""RL configuration for the Unitree G1 skater joystick task."""

from mjlab_roller.rl.config import (
  RslRlAMPOnPolicyRunnerCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
  RslRlResidualActorCriticCfg,
  RslRlResidualAmpAlgorithmCfg,
  RslRlResidualAmpRunnerCfg,
)


def unitree_g1_skater_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="PPO",
    ),
    experiment_name="g1_roller_skater_ppo",
    logger="tensorboard",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
  )


def unitree_g1_skater_amp_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
  return RslRlAMPOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.3,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
      class_name="AMP_PPO",
    ),
    experiment_name="g1_roller_skater_amp_ppo",
    logger="tensorboard",
    save_interval=2000,
    num_steps_per_env=24,
    max_iterations=100_000,
    amp_num_obs=23,
    amp_num_frames=5,
    use_lerp=False,
    amp_task_reward_lerp=0.7,
    amp_reward_coef=1.0,
    amp_motion_files="dataset/rollerskating_amp",
    amp_num_preload_transitions=200000,
    amp_discr_hidden_dims=(256, 256),
    lerp_warmup_iters=5000,
    lerp_warmup_start=0.3,
    transition_lr_start=5e-6,
    transition_clip_start=0.1,
    reward_schedule_anchor_iter=59_500,
  )


def unitree_g1_skater_residual_amp_ppo_runner_cfg() -> RslRlResidualAmpRunnerCfg:
  """MoRE residual-expert AMP variant.

  Sits on top of a frozen base PPO checkpoint. Only the residual actor +
  critic + discriminator are trained. Discriminator consumes upper-body
  joints (waist + arms, 11 DoF) only so the lower-body skating rhythm is
  protected from the AMP style reward.

  Adjustments for the tiny 429-transition AMP dataset:
    - preload_transitions: 200k → 5k  (11.7× oversample instead of 465×)
    - discriminator hidden: (256, 256) → (128, 64)
    - grad-pen coef: 1.0 → 10.0
  """

  return RslRlResidualAmpRunnerCfg(
    policy=RslRlResidualActorCriticCfg(
      init_noise_std=0.3,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(256, 128),  # residual MLP
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
      # Residual-specific fields:
      base_ckpt_path="model_58999.pt",
      base_actor_hidden_dims=(512, 256, 128),
      residual_hidden_dims=(256, 128),
      # delta: defaults from dataclass (12 legs @0.05, 3 waist, 8 arms)
    ),
    algorithm=RslRlResidualAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
      grad_pen_coef=10.0,
    ),
    experiment_name="g1_roller_residual_amp",
    logger="tensorboard",
    save_interval=2000,
    num_steps_per_env=24,
    max_iterations=100_000,
    amp_num_obs=11,  # upper-body slice width (23 → 11)
    amp_num_frames=5,
    amp_state_slice=(12, 23),  # waist + arms
    use_lerp=False,
    amp_task_reward_lerp=0.7,
    amp_reward_coef=1.0,
    amp_motion_files="dataset/rollerskating_amp",
    amp_num_preload_transitions=5_000,  # 429 unique → 11.7× oversample
    amp_discr_hidden_dims=(128, 64),  # smaller disc to resist overfitting
    lerp_warmup_iters=5000,
    lerp_warmup_start=0.3,
    transition_lr_start=5e-6,
    transition_clip_start=0.1,
    reward_schedule_anchor_iter=0,
  )
