import os

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_roller_policy_as_onnx(
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  from mjlab_roller.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_export_metadata,
  )

  metadata = get_export_metadata(env, run_path)
  attach_metadata_to_onnx(os.path.join(path, filename), metadata)
