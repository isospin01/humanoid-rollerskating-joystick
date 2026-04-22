from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.rl.exporter_utils import get_export_metadata  # type: ignore


class _DummyEnv:
  def get_export_metadata(self) -> dict[str, list | str | float]:
    return {
      "command_axes": ["vx", "vy", "wz"],
      "command_frame": "body",
      "action_beta_max": 1.0,
    }


class ExporterMetadataTests(unittest.TestCase):
  def test_extra_metadata_is_merged(self) -> None:
    with patch(
      "mjlab_roller.rl.exporter_utils.get_base_metadata",
      return_value={"run_path": "demo/run", "joint_names": ["foo_joint"]},
    ):
      metadata = get_export_metadata(_DummyEnv(), "demo/run")
    self.assertEqual(metadata["run_path"], "demo/run")
    self.assertEqual(metadata["joint_names"], ["foo_joint"])
    self.assertEqual(metadata["command_axes"], ["vx", "vy", "wz"])
    self.assertEqual(metadata["command_frame"], "body")
    self.assertEqual(metadata["action_beta_max"], 1.0)


if __name__ == "__main__":
  unittest.main()
