"""AMP-PPO variant for the MoRE residual architecture.

Two behavioural changes relative to ``rsl_rl.algorithms.AMP_PPO``:

1. ``act(obs, amp_obs)`` returns the *env action* (``base(obs) + tanh(raw)*delta``)
   while still storing the *raw* residual sample in the rollout buffer so
   that PPO's log-prob bookkeeping is computed on the Normal distribution
   over ``raw``. This is the clean way to separate what goes into
   ``env.step`` from what PPO sees at update time.

2. The optimizer is rebuilt to register only parameters with
   ``requires_grad=True``, preventing Adam from tracking moment state for
   the frozen base actor + normalizer.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rsl_rl.algorithms import AMP_PPO


class ResidualAMPPPO(AMP_PPO):
    """AMP-PPO that understands raw residual samples vs env actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rebuild_optimizer_trainable_only()

    # ------------------------------------------------------------------
    # Optimizer rebuilt to skip frozen params
    # ------------------------------------------------------------------
    def _rebuild_optimizer_trainable_only(self) -> None:
        # Pull current LR and weight_decay settings from the parent-built optimizer.
        learning_rate = self.learning_rate

        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        trunk_params = list(self.discriminator.trunk.parameters())
        head_params = list(self.discriminator.amp_linear.parameters())

        params = [
            {"params": policy_params, "name": "policy"},
            {"params": trunk_params, "weight_decay": 10e-4, "name": "amp_trunk"},
            {"params": head_params, "weight_decay": 10e-2, "name": "amp_head"},
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Report how much frozen weight we saved optimizer state for.
        total_policy = sum(p.numel() for p in self.policy.parameters())
        trainable_policy = sum(p.numel() for p in policy_params)
        frozen = total_policy - trainable_policy
        print(
            f"[ResidualAMPPPO] policy params — trainable={trainable_policy:,}"
            f" frozen={frozen:,} (total={total_policy:,})"
        )

    # ------------------------------------------------------------------
    # Rollout act: store raw, return env action
    # ------------------------------------------------------------------
    def act(self, obs, amp_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # `policy.act(obs)` on ResidualActorCritic returns a Normal *sample*
        # of the residual (raw). Detach to avoid grad leaking into rollout.
        raw = self.policy.act(obs).detach()

        # Separate env action computed from raw + frozen base pass.
        env_action = self.policy.raw_to_action(obs, raw).detach()

        # Store RAW in the buffer so update-time log_prob is consistent.
        self.transition.actions = raw
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = (
            self.policy.get_actions_log_prob(raw).detach()
        )
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.amp_transition.observations = amp_obs

        return env_action
