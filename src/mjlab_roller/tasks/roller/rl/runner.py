import os

from mjlab_roller.rl.vecenv_wrapper import RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

from mjlab_roller.tasks.roller.rl.exporter import (
  attach_onnx_metadata,
  export_roller_policy_as_onnx,
)


class RollerOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      try:
        import wandb
      except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
          "W&B logging is enabled for this run, but the `wandb` package is not installed."
        ) from exc
      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      normalizer = self.alg.policy.actor_obs_normalizer if self.alg.policy.actor_obs_normalization else None
      export_roller_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
