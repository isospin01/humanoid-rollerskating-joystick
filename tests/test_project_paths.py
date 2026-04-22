from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.core.project_paths import data_path, project_root, resolve_project_path  # type: ignore


class ProjectPathTests(unittest.TestCase):
  def test_project_root_is_repo_root(self) -> None:
    self.assertEqual(project_root(), ROOT)

  def test_data_path_points_into_dataset_dir(self) -> None:
    target = data_path("ref_pose", "push_start_pose_b.npy")
    self.assertTrue(target.is_file())

  def test_resolve_project_path_handles_relative_inputs(self) -> None:
    target = resolve_project_path("dataset/ref_pose/push_start_pose_b.npy")
    self.assertTrue(target.is_file())


if __name__ == "__main__":
  unittest.main()
