from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
import shutil
import unittest
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mjlab_roller.data.amp_dataset import CONTROLLED_JOINT_COUNT  # type: ignore
from mjlab_roller.data.amp_dataset import make_manifest_entry, normalize_amp_clip, validate_amp_dataset_dir, write_manifest  # type: ignore
loader_spec = importlib.util.spec_from_file_location(
  "motion_loader_g1",
  ROOT / "rsl_rl" / "utils" / "motion_loader_g1.py",
)
assert loader_spec is not None and loader_spec.loader is not None
loader_module = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(loader_module)
G1_AMPLoader = loader_module.G1_AMPLoader


class AmpDatasetTests(unittest.TestCase):
  def _make_test_dir(self) -> Path:
    path = ROOT / ".test_artifacts" / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=False)
    self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path

  def test_normalize_amp_clip_projects_legacy_shape(self) -> None:
    legacy = np.arange(5 * 36, dtype=np.float32).reshape(5, 36)
    projected = normalize_amp_clip(legacy)
    self.assertEqual(projected.shape, (5, CONTROLLED_JOINT_COUNT))

  def test_loader_yields_expected_amp_batch_shape(self) -> None:
    root = self._make_test_dir()
    clip = np.linspace(0.0, 1.0, 6 * CONTROLLED_JOINT_COUNT, dtype=np.float32).reshape(6, CONTROLLED_JOINT_COUNT)
    np.save(root / "clip.npy", clip)
    write_manifest(
      root,
      [make_manifest_entry(clip_id="clip", file_name="clip.npy", num_frames=6, source_kind="unit_test")],
    )
    loader = G1_AMPLoader("cpu", time_between_frames=1 / 50.0, motion_files=str(root), preload_transitions=True, num_preload_transitions=8, num_frames=5)
    batch = next(loader.feed_forward_generator_23dof_multi(1, 4))
    self.assertEqual(batch.shape, (4, 5, CONTROLLED_JOINT_COUNT))
    self.assertEqual(batch.dtype, torch.float32)

  def test_validate_amp_dataset_dir_requires_minimum_num_frames(self) -> None:
    root = self._make_test_dir()
    np.save(root / "short.npy", np.ones((3, CONTROLLED_JOINT_COUNT), dtype=np.float32))
    with self.assertRaises(ValueError):
      validate_amp_dataset_dir(root, num_frames=5)


if __name__ == "__main__":
  unittest.main()
