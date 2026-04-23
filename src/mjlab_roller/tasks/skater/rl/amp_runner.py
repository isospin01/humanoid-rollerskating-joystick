"""AMP on-policy runner for the skater joystick task.

Extends ``AMPOnPolicyRunner`` with graceful warm-start loading from a base
(non-AMP) PPO checkpoint: only the actor-critic weights are restored while
the discriminator and optimizer start fresh.
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
