"""AMP on-policy runner for the skater joystick task.

Extends ``AMPOnPolicyRunner`` with graceful warm-start loading from a base
(non-AMP) PPO checkpoint: only the actor-critic weights are restored while
the discriminator and optimizer start fresh.

Also defines ``SkaterResidualAMPOnPolicyRunner`` — a MoRE-style residual
variant that wraps a frozen base policy with a trainable residual MLP and
uses an upper-body-only discriminator.
"""

from __future__ import annotations

import torch

from rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner


class SkaterAMPOnPolicyRunner(AMPOnPolicyRunner):

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
        reset_critic: bool = False,
    ):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        if reset_critic:
            full_sd = loaded_dict["model_state_dict"]
            actor_sd = {k: v for k, v in full_sd.items()
                        if not k.startswith(("critic.", "critic_obs_normalizer."))}
            self.alg.policy.load_state_dict(actor_sd, strict=False)
            print("[INFO] Loaded actor weights only — critic randomly re-initialized.")
        else:
            self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if "disc_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["disc_state_dict"], strict=False)
            print("[INFO] Loaded discriminator weights from checkpoint.")
        else:
            print("[INFO] No discriminator weights in checkpoint — starting fresh.")
        if "amp_normalizer" in loaded_dict:
            for k, v in loaded_dict["amp_normalizer"].items():
                setattr(self.alg.amp_normalizer, k, v)
            print("[INFO] Loaded AMP normalizer from checkpoint.")

        if reset_critic:
            self.current_learning_iteration = loaded_dict.get("iter", 0)
            print(f"[INFO] Optimizer state NOT loaded (critic reset). "
                  f"Resuming from iteration {self.current_learning_iteration}.")
        elif load_optimizer:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
                self.current_learning_iteration = loaded_dict["iter"]
                print(f"[INFO] Resumed AMP training from iteration {self.current_learning_iteration}")
            except (ValueError, RuntimeError, KeyError):
                print(
                    f"[INFO] Loaded actor-critic weights from base checkpoint "
                    f"(trained to iter {loaded_dict.get('iter', '?')}). "
                    f"Discriminator and optimizer start fresh; iteration reset to 0."
                )
                self.current_learning_iteration = 0

        return loaded_dict.get("infos")


class SkaterResidualAMPOnPolicyRunner(SkaterAMPOnPolicyRunner):
    """MoRE residual-expert variant.

    Two responsibilities beyond the parent:

    1. Build the policy as ``ResidualActorCritic`` (frozen base + trainable
       residual MLP) and the algorithm as ``ResidualAMPPPO`` (which stores
       raw residual samples in the rollout buffer but returns the env
       action ``base(s) + tanh(raw)·delta`` to the caller).
    2. Apply upper-body masking to the AMP data path: the discriminator sees
       only waist + arm joints (indices 12-22 of the 23-DoF state), keeping
       the lower-body gait-style untouched by the style reward.

    The masking is implemented as a post-``super().__init__`` rewrite of
    ``self.amp_data.preloaded_sequences`` and a monkey-patch of
    ``env.get_amp_observations`` so no base-class code needs modification.
    """

    def __init__(self, env, train_cfg, log_dir: str | None = None, device: str = "cpu"):
        # --- Slice spec: (start, end) into the 23-DoF AMP state vector ---
        slice_spec = train_cfg.get("amp_state_slice")
        if slice_spec is not None:
            if not (isinstance(slice_spec, (list, tuple)) and len(slice_spec) == 2):
                raise ValueError(
                    f"amp_state_slice must be a (start, end) pair, got {slice_spec!r}"
                )
            s_start, s_end = int(slice_spec[0]), int(slice_spec[1])
            expected = s_end - s_start
            declared = train_cfg.get("amp_num_obs", 0)
            if declared == 0 or declared != expected:
                raise ValueError(
                    f"amp_num_obs ({declared}) must equal slice width "
                    f"({expected}) when amp_state_slice is set."
                )

        # Let parent build everything (discriminator uses amp_num_obs, so set it
        # to the masked width BEFORE calling super).
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # --- Apply the slice post-hoc where raw data enters/exits ---
        if slice_spec is not None:
            self._install_amp_upper_body_mask(int(slice_spec[0]), int(slice_spec[1]))

    def _install_amp_upper_body_mask(self, s_start: int, s_end: int) -> None:
        # 1. Slice the preloaded expert transitions in the loader in-place.
        if getattr(self.amp_data, "preloaded_sequences", None) is not None:
            self.amp_data.preloaded_sequences = (
                self.amp_data.preloaded_sequences[..., s_start:s_end].contiguous()
            )
        # Record the advertised state dim for downstream sanity checks.
        self.amp_data._observation_dim = s_end - s_start

        # 2. Wrap env.get_amp_observations so the rollout loop stores masked obs.
        env = self.env
        original_get = env.get_amp_observations

        def _sliced_get():
            return original_get()[..., s_start:s_end]

        env.get_amp_observations = _sliced_get  # type: ignore[attr-defined]

        print(
            f"[SkaterResidualAMPOnPolicyRunner] AMP state masked to "
            f"[{s_start}:{s_end}] (width={s_end - s_start})"
        )

    # ------------------------------------------------------------------
    # Algorithm construction — uses ResidualActorCritic + ResidualAMPPPO.
    # ------------------------------------------------------------------
    def _construct_algorithm(self, obs):
        # Late imports to avoid circulars and to keep this class importable
        # even when the project is being read statically (e.g., by linters).
        from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
        import warnings

        from mjlab_roller.rl.residual_actor_critic import ResidualActorCritic
        from mjlab_roller.rl.residual_amp_ppo import ResidualAMPPPO

        # Mirror base class rnd/symmetry resolution + deprecation handling.
        self.alg_cfg = resolve_rnd_config(
            self.alg_cfg, obs, self.cfg["obs_groups"], self.env
        )
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "`empirical_normalization` is deprecated; use policy.actor_obs_normalization.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # class_name is set by the rl_cfg for documentation; drop it here because
        # we explicitly pick ResidualActorCritic / ResidualAMPPPO below.
        self.policy_cfg.pop("class_name", None)
        self.alg_cfg.pop("class_name", None)

        # --- Resolve per-joint delta tensor (list in cfg → tensor on device) ---
        delta_cfg = self.policy_cfg.pop("delta", None)
        if delta_cfg is None:
            raise ValueError(
                "policy_cfg is missing required field 'delta' (per-joint residual scale)."
            )
        if isinstance(delta_cfg, torch.Tensor):
            delta_tensor = delta_cfg.clone().to(self.device, dtype=torch.float32)
        else:
            delta_tensor = torch.tensor(list(delta_cfg), dtype=torch.float32, device=self.device)

        base_ckpt_path = self.policy_cfg.pop("base_ckpt_path", None)
        if not base_ckpt_path:
            raise ValueError("policy_cfg is missing required field 'base_ckpt_path'.")

        # --- Instantiate the residual actor-critic ---
        policy = ResidualActorCritic(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            base_ckpt_path=base_ckpt_path,
            delta=delta_tensor,
            **self.policy_cfg,
        ).to(self.device)

        # --- Instantiate the residual AMP-PPO algorithm ---
        alg = ResidualAMPPPO(
            policy,
            self.discriminator,
            self.amp_data,
            self.amp_normalizer,
            self.amp_num_frames,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg,
        )
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg
