"""Residual expert ActorCritic on top of a frozen base policy.

Implements the MoRE-style residual architecture described in the project's
AMP training plan:

  action = base_actor(s).detach() + tanh(raw) * delta
  raw   ~ Normal(residual_mean(s), residual_std)

- ``base_actor`` is loaded from an existing PPO checkpoint and frozen.
- ``residual_actor`` is a small MLP whose last layer is zero-initialised so
  that at iteration 0 the effective action equals the base action.
- ``delta`` is a per-joint scale (e.g. 0.05 rad for legs, 0.30 rad for arms).

PPO bookkeeping convention:
- ``act(obs)`` returns the raw residual sample. The enclosing algorithm
  stores this raw tensor in the rollout buffer so that ``get_actions_log_prob``
  computes ``Normal.log_prob(raw)`` during update.
- ``raw_to_action(obs, raw)`` must be called by the algorithm to produce the
  tensor passed to ``env.step``. It is NOT idempotent with ``act``.
- ``act_inference(obs)`` (used by ``play.py``) returns the full deterministic
  action directly (``a_base + tanh(raw_mu) * delta``).

The tanh squashing's Jacobian is *intentionally* omitted from log-prob because
it is constant in the policy parameters (``delta`` is fixed), so it cancels
out of the PPO importance ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization


class ResidualActorCritic(ActorCritic):
    """ActorCritic whose actor is (frozen base policy) + (trainable residual).

    Args:
        obs, obs_groups, num_actions: forwarded to ActorCritic.__init__.
        base_ckpt_path: path to a PPO checkpoint (``.pt``) from which the base
            actor weights and normalizer stats are loaded.
        delta: per-joint residual scale tensor of shape ``[num_actions]``.
        residual_hidden_dims: hidden sizes of the residual MLP. Defaults to
            ``[256, 128]`` per the MoRE plan.
        base_actor_hidden_dims: shape of the base actor in the checkpoint.
            Must match the checkpoint's ``actor`` layer layout.
        actor_hidden_dims: ignored (retained only so tyro-style config wiring
            does not choke); residual architecture is set via
            ``residual_hidden_dims``.
        init_noise_std: initial std of the residual Normal distribution.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        *,
        base_ckpt_path: str,
        delta: torch.Tensor,
        residual_hidden_dims: tuple[int, ...] = (256, 128),
        base_actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        critic_hidden_dims: tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.3,
        noise_std_type: str = "scalar",
        **kwargs: Any,
    ):
        # If the caller passed actor_hidden_dims via kwargs, prefer
        # residual_hidden_dims (the semantically-named field) and drop the
        # duplicate to avoid TypeError in super().__init__.
        kwargs.pop("actor_hidden_dims", None)

        # Build the residual ActorCritic shell via parent init — this creates
        # self.actor (we will repurpose it for the residual), self.critic, std,
        # actor_obs_normalizer / critic_obs_normalizer.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=list(residual_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=False,
            **kwargs,
        )

        # The parent's self.actor is now the residual trunk + head.
        # Zero-init the residual's last linear layer so initial output == 0
        # → action == a_base at iter 0.
        self._zero_init_last_linear(self.actor)

        # --- Build the frozen base actor (same input dim + output dim) ---
        num_actor_obs = self._infer_num_actor_obs(obs, obs_groups)
        self.base_actor = MLP(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dims=list(base_actor_hidden_dims),
            activation=activation,
        )

        # A separate normalizer for the base path (frozen after load).
        # We always create it so the code path is uniform; if the base ckpt
        # was trained without normalization the loaded state will be Identity-
        # like.
        self.base_actor_obs_normalizer: nn.Module
        if actor_obs_normalization:
            self.base_actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.base_actor_obs_normalizer = nn.Identity()

        # --- Load base weights from checkpoint ---
        self._load_base_from_ckpt(base_ckpt_path)

        # --- Freeze base actor + its normalizer ---
        for p in self.base_actor.parameters():
            p.requires_grad_(False)
        self.base_actor.eval()
        for p in self.base_actor_obs_normalizer.parameters():
            p.requires_grad_(False)

        # --- Register delta buffer ---
        if delta.shape != (num_actions,):
            raise ValueError(
                f"delta must have shape [{num_actions}], got {tuple(delta.shape)}"
            )
        self.register_buffer("delta", delta.clone().to(torch.float32))

        # Cache for shape validation and debugging.
        self._num_actor_obs = num_actor_obs
        self._num_actions = num_actions

        print(
            f"[ResidualActorCritic] base={base_actor_hidden_dims} (frozen)"
            f" + residual={residual_hidden_dims} (trainable)"
        )
        print(
            f"[ResidualActorCritic] delta (per-joint) min={self.delta.min().item():.3f}"
            f" max={self.delta.max().item():.3f}"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_num_actor_obs(obs, obs_groups) -> int:
        total = 0
        for g in obs_groups["policy"]:
            total += obs[g].shape[-1]
        return total

    @staticmethod
    def _zero_init_last_linear(seq: nn.Module) -> None:
        last = None
        for m in seq.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            raise RuntimeError("ResidualActorCritic: no Linear layer in residual actor")
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _load_base_from_ckpt(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ResidualActorCritic: base_ckpt_path not found: {path}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)

        actor_sd = {
            k[len("actor.") :]: v for k, v in sd.items() if k.startswith("actor.")
        }
        if not actor_sd:
            raise RuntimeError(
                f"ResidualActorCritic: checkpoint {path} has no 'actor.*' keys"
            )
        missing, unexpected = self.base_actor.load_state_dict(actor_sd, strict=False)
        if missing or unexpected:
            print(
                f"[ResidualActorCritic] base_actor load missing={missing}"
                f" unexpected={unexpected}"
            )

        if isinstance(self.base_actor_obs_normalizer, EmpiricalNormalization):
            norm_sd = {
                k[len("actor_obs_normalizer.") :]: v
                for k, v in sd.items()
                if k.startswith("actor_obs_normalizer.")
            }
            if norm_sd:
                m, u = self.base_actor_obs_normalizer.load_state_dict(
                    norm_sd, strict=False
                )
                if m or u:
                    print(
                        f"[ResidualActorCritic] base_normalizer load missing={m}"
                        f" unexpected={u}"
                    )
            else:
                print(
                    "[ResidualActorCritic] WARNING: checkpoint has no"
                    " actor_obs_normalizer.* keys — base normalizer left at"
                    " identity-init running stats."
                )

        print(f"[ResidualActorCritic] base loaded from {path} (iter={ckpt.get('iter', '?')})")

    # ------------------------------------------------------------------
    # Core forward — overrides ActorCritic
    # ------------------------------------------------------------------

    def _normalized_actor_obs(self, obs) -> torch.Tensor:
        """Apply the base (frozen) normalizer — residual and base share stats."""
        actor_obs = self.get_actor_obs(obs)
        return self.base_actor_obs_normalizer(actor_obs)

    def update_distribution(self, obs):
        """Populate self.distribution = Normal(residual_mean, residual_std).

        Called by ActorCritic.act with ``obs`` already normalized. But we
        replace that path in our ``act`` override below to use the base
        normalizer, so here ``obs`` is already normalized.
        """
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        """Return a raw residual sample. NOT the env action.

        The enclosing AMP_PPO / ResidualAMPPPO algorithm stores this tensor in
        the rollout buffer. ``raw_to_action(obs, raw)`` converts it into the
        actual env action.
        """
        normalized = self._normalized_actor_obs(obs)
        self.update_distribution(normalized)
        raw = self.distribution.sample()
        return raw

    def act_inference(self, obs):
        """Deterministic env action, used by ``play.py``.

        Returns a_base + tanh(residual_mean) * delta (no stochastic sampling).
        """
        normalized = self._normalized_actor_obs(obs)
        with torch.no_grad():
            a_base = self.base_actor(normalized)
            raw_mu = self.actor(normalized)
            residual = torch.tanh(raw_mu) * self.delta
            return a_base + residual

    def raw_to_action(self, obs, raw: torch.Tensor) -> torch.Tensor:
        """Convert a raw residual sample into the env action.

        Called by ResidualAMPPPO.act after ``policy.act`` has returned raw.
        The base pass runs under ``no_grad`` because base is frozen; the
        tanh(raw) * delta part is NOT wrapped in no_grad because raw carries
        no grad here (it is the .detach()'d sample stored in rollout buffer).
        """
        normalized = self._normalized_actor_obs(obs)
        with torch.no_grad():
            a_base = self.base_actor(normalized)
        residual = torch.tanh(raw) * self.delta
        return a_base + residual

    def evaluate(self, obs, **kwargs):
        """Standard critic forward, fully trainable (reset_critic=True expected)."""
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    # ------------------------------------------------------------------
    # Normalization updates
    # ------------------------------------------------------------------

    def update_normalization(self, obs):
        """Update critic-side running stats only.

        The base-actor obs normalizer stays frozen — keeping the base policy's
        input distribution identical to what it was trained with.
        """
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # ------------------------------------------------------------------
    # Parameter filtering
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Yield only the parameters the optimizer should see.

        Excludes base_actor and base_actor_obs_normalizer. The optimizer
        constructor in the runner should call ``list(policy.trainable_parameters())``
        rather than ``policy.parameters()`` to avoid Adam holding moment state
        for frozen weights.
        """
        for p in self.parameters():
            if p.requires_grad:
                yield p
