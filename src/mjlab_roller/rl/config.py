"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass
class RslRlPpoActorCriticCfg:
  """Config for the PPO actor-critic networks."""

  init_noise_std: float = 1.0
  """The initial noise standard deviation of the policy."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation for the policy. Default is scalar."""
  actor_obs_normalization: bool = False
  """Whether to normalize the observation for the actor network. Default is False."""
  critic_obs_normalization: bool = False
  """Whether to normalize the observation for the critic network. Default is False."""
  actor_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the critic network."""
  activation: str = "elu"
  """The activation function to use in the actor and critic networks."""
  class_name: str = "ActorCritic"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  class_name: str = "PPO"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"policy": ("policy",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """The experiment name."""
  run_name: str = ""
  """The run name. Default is empty string."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
  """The policy configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""

@dataclass
class RslRlAMPOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  amp_num_obs: int = 23
  amp_num_frames: int = 5
  use_lerp: bool = False
  amp_task_reward_lerp: float = 0.7
  amp_reward_coef: float = 5.0
  amp_motion_files: str = "dataset/roller_push"
  amp_num_preload_transitions: int = 200000
  amp_discr_hidden_dims: Tuple[int, ...] = (256, 256)
  min_normalized_std: Tuple[float, ...] = (0.05,) * 20
  lerp_warmup_iters: int = 0
  lerp_warmup_start: float = 0.3
  transition_lr_start: Optional[float] = None
  """If set (with lerp_warmup_iters > 0), use this LR and fixed schedule for the first
  lerp_warmup_iters iterations, then restore algorithm learning_rate and schedule."""
  transition_clip_start: Optional[float] = None
  """If set, use this clip_param during the same window as transition_lr_start."""
  reward_schedule_anchor_iter: Optional[int] = None
  """If set, lerp warmup and LR/clip transition are measured from this absolute iteration
  (not from resume `start_iter`), so mid-run resumes do not rewind schedules."""


# ---------------------------------------------------------------------------
# Residual MoRE variant
# ---------------------------------------------------------------------------


@dataclass
class RslRlResidualActorCriticCfg(RslRlPpoActorCriticCfg):
  """Actor-critic cfg for the MoRE residual variant.

  Extends the PPO cfg with the fields ``ResidualActorCritic`` needs:

  - ``base_ckpt_path``: path to the frozen base PPO checkpoint (.pt).
  - ``delta``: per-joint residual scale, length = num_actions (23 for G1).
  - ``residual_hidden_dims``: residual MLP hidden sizes (small, plan default (256, 128)).
  - ``base_actor_hidden_dims``: must match base ckpt's actor shape.
  """

  base_ckpt_path: str = "model_58999.pt"
  delta: Tuple[float, ...] = (
    # 12 legs @ 0.05 rad (~2.9°) — keep SKATER skate rhythm
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    # 3 waist (yaw, roll, pitch) — allow moderate torso twist
    0.15, 0.10, 0.10,
    # 4 left arm: sh_pitch, sh_roll, sh_yaw, elbow — human-scale swing
    0.30, 0.30, 0.30, 0.20,
    # 4 right arm (mirror)
    0.30, 0.30, 0.30, 0.20,
  )
  residual_hidden_dims: Tuple[int, ...] = (256, 128)
  base_actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
  class_name: str = "ResidualActorCritic"


@dataclass
class RslRlResidualAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Algorithm cfg for ResidualAMPPPO: same as PPO cfg + grad_pen_coef."""

  grad_pen_coef: float = 10.0
  """WGAN-GP gradient penalty coefficient. Repo baseline used 1.0
  (hardcoded). Residual-AMP recipe bumps to 10.0 per standard WGAN-GP
  practice."""
  class_name: str = "ResidualAMPPPO"


@dataclass
class RslRlResidualAmpRunnerCfg(RslRlAMPOnPolicyRunnerCfg):
  """Runner cfg for the MoRE residual AMP task.

  Inherits all AMP fields and layers on residual-specific ones plus an
  upper-body-only discriminator state slice.
  """

  policy: RslRlResidualActorCriticCfg = field(default_factory=RslRlResidualActorCriticCfg)
  algorithm: RslRlResidualAmpAlgorithmCfg = field(default_factory=RslRlResidualAmpAlgorithmCfg)
  amp_state_slice: Optional[Tuple[int, int]] = (12, 23)
  """Half-open (start, end) index into the 23-DoF AMP state vector; if set,
  the discriminator sees only this slice. Default (12, 23) = waist + arms
  (11 DoF). Set to None to disable upper-body masking."""
  class_name: str = "SkaterResidualAMPOnPolicyRunner"
