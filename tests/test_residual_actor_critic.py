"""Unit tests for ResidualActorCritic.

Covers:
  1. Shape correctness of .act / .act_inference / .raw_to_action outputs.
  2. At initialization, env action == base action (zero-init residual).
  3. Base actor parameters receive no gradient after backward on residual loss.
  4. Residual actor + log_std DO receive gradient.
  5. raw_to_action round-trip: action = base(obs) + tanh(raw) * delta.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Repo root on sys.path for direct import without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Avoid MuJoCo GL init under headless CI.
os.environ.setdefault("MUJOCO_GL", "egl")

from rsl_rl.modules import ActorCritic  # noqa: E402

from mjlab_roller.rl.residual_actor_critic import ResidualActorCritic  # noqa: E402


ACTOR_OBS_DIM = 920
CRITIC_OBS_DIM = 134
NUM_ACTIONS = 23
BASE_HIDDEN = (512, 256, 128)
RESIDUAL_HIDDEN = (256, 128)
BATCH = 4


def _make_obs(batch: int = BATCH, device: str = "cpu"):
    return {
        "policy": torch.randn(batch, ACTOR_OBS_DIM, device=device),
        "critic": torch.randn(batch, CRITIC_OBS_DIM, device=device),
    }


def _make_obs_groups():
    return {"policy": ["policy"], "critic": ["critic"]}


def _make_delta() -> torch.Tensor:
    # 12 legs @ 0.05, 3 waist (0.15,0.10,0.10), 4 left arm (0.30,0.30,0.30,0.20),
    # 4 right arm (same).
    return torch.tensor(
        [0.05] * 12 + [0.15, 0.10, 0.10] + [0.30, 0.30, 0.30, 0.20] * 2,
        dtype=torch.float32,
    )


def _fake_base_ckpt(tmpdir: Path) -> Path:
    """Build a minimal valid ActorCritic checkpoint with the expected hidden dims."""
    obs = _make_obs(batch=1)
    obs_groups = _make_obs_groups()
    base = ActorCritic(
        obs,
        obs_groups,
        NUM_ACTIONS,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=list(BASE_HIDDEN),
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
    )
    # Randomize so zero vs non-zero effects are visible.
    for p in base.parameters():
        nn.init.normal_(p, std=0.1) if p.dim() >= 2 else nn.init.zeros_(p)

    ckpt = {"model_state_dict": base.state_dict(), "iter": 42}
    path = tmpdir / "fake_base.pt"
    torch.save(ckpt, path)
    return path


class ResidualActorCriticTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.ckpt_path = _fake_base_ckpt(self.tmp_path)
        self.obs = _make_obs()
        self.policy = ResidualActorCritic(
            obs=self.obs,
            obs_groups=_make_obs_groups(),
            num_actions=NUM_ACTIONS,
            base_ckpt_path=str(self.ckpt_path),
            delta=_make_delta(),
            residual_hidden_dims=RESIDUAL_HIDDEN,
            base_actor_hidden_dims=BASE_HIDDEN,
            critic_hidden_dims=(256, 256, 256),
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            activation="elu",
            init_noise_std=0.3,
            noise_std_type="scalar",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_01_act_returns_raw_with_correct_shape(self):
        raw = self.policy.act(self.obs)
        self.assertEqual(raw.shape, (BATCH, NUM_ACTIONS))
        self.assertTrue(torch.isfinite(raw).all())

    def test_02_act_inference_shape_and_finite(self):
        act = self.policy.act_inference(self.obs)
        self.assertEqual(act.shape, (BATCH, NUM_ACTIONS))
        self.assertTrue(torch.isfinite(act).all())

    def test_03_initial_action_equals_base_action(self):
        """Zero-init residual head → tanh(0)*delta = 0 → action == base."""
        # Compute base action directly.
        with torch.no_grad():
            actor_obs = self.policy.get_actor_obs(self.obs)
            normalized = self.policy.base_actor_obs_normalizer(actor_obs)
            a_base_expected = self.policy.base_actor(normalized)
        a_inf = self.policy.act_inference(self.obs)
        self.assertTrue(
            torch.allclose(a_inf, a_base_expected, atol=1e-6),
            msg=f"act_inference deviates from base at init: max |diff| = "
                f"{(a_inf - a_base_expected).abs().max().item():.2e}",
        )

    def test_04_raw_to_action_matches_formula(self):
        """raw_to_action(obs, raw) == base(obs) + tanh(raw)*delta."""
        raw = self.policy.act(self.obs).detach()
        action = self.policy.raw_to_action(self.obs, raw)
        with torch.no_grad():
            actor_obs = self.policy.get_actor_obs(self.obs)
            normalized = self.policy.base_actor_obs_normalizer(actor_obs)
            a_base = self.policy.base_actor(normalized)
            expected = a_base + torch.tanh(raw) * self.policy.delta
        self.assertTrue(torch.allclose(action, expected, atol=1e-6))

    def test_05_base_params_have_no_grad(self):
        """Backward through the PPO-style residual loss must not touch base params."""
        # Mimic the PPO update path: act() populates self.distribution, then
        # we compute log_prob + value as a proxy loss. This is where real
        # gradients to residual params (through mu, std) and to critic flow.
        raw = self.policy.act(self.obs).detach()  # stored in buffer, grad-free
        # New forward pass (like update step) — re-populates distribution.
        self.policy.act(self.obs)
        log_prob = self.policy.get_actions_log_prob(raw)
        value = self.policy.evaluate(self.obs)
        loss = -log_prob.mean() + value.pow(2).mean()
        loss.backward()

        for name, p in self.policy.base_actor.named_parameters():
            self.assertFalse(p.requires_grad, f"base_actor param {name} requires_grad unexpectedly")
            self.assertIsNone(p.grad, f"base_actor param {name} has grad: {p.grad}")

    def test_06_residual_params_receive_grad(self):
        """Residual trunk/head/std + critic must receive gradient."""
        raw = self.policy.act(self.obs).detach()
        self.policy.act(self.obs)
        log_prob = self.policy.get_actions_log_prob(raw)
        value = self.policy.evaluate(self.obs)
        loss = -log_prob.mean() + value.pow(2).mean()
        loss.backward()

        trainable_with_grad = 0
        trainable_missing_grad = []
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("base_actor") or name.startswith("base_actor_obs_normalizer"):
                continue
            if p.grad is None:
                trainable_missing_grad.append(name)
            else:
                trainable_with_grad += 1

        self.assertGreater(trainable_with_grad, 0, "No residual param received grad.")
        self.assertEqual(
            trainable_missing_grad,
            [],
            msg=f"Residual/critic params missing grad: {trainable_missing_grad}",
        )

    def test_07_trainable_parameters_excludes_base(self):
        trainable = list(self.policy.trainable_parameters())
        trainable_ids = {id(p) for p in trainable}
        for _, p in self.policy.base_actor.named_parameters():
            self.assertNotIn(id(p), trainable_ids, "base_actor param leaked into trainable_parameters")
        self.assertGreater(len(trainable), 0)

    def test_08_evaluate_returns_value_shape(self):
        v = self.policy.evaluate(self.obs)
        self.assertEqual(v.shape, (BATCH, 1))


if __name__ == "__main__":
    unittest.main()
