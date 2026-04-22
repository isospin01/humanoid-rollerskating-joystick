from __future__ import annotations

import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mjlab_roller.core.project_paths import data_path  # type: ignore


def _robot_body_names() -> list[str]:
  tree = ET.parse(ROOT / "src" / "mjlab_roller" / "assets" / "robots" / "roller" / "xmls" / "g1.xml")
  worldbody = tree.getroot().find("worldbody")
  assert worldbody is not None

  names: list[str] = []

  def walk(node: ET.Element) -> None:
    for body in node.findall("body"):
      names.append(body.attrib["name"])
      walk(body)

  walk(worldbody)
  return names


class TransitionReferenceAlignmentTests(unittest.TestCase):
  def test_reference_pose_count_matches_non_skate_non_wheel_bodies(self) -> None:
    body_names = _robot_body_names()
    transition_body_names = [
      name
      for name in body_names
      if "inline_skate" not in name and "wheel" not in name
    ]
    self.assertEqual(len(transition_body_names), 30)

    for filename in (
      "push_start_pose_b.npy",
      "glide_start_pose_b.npy",
      "steer_start_pose_b.npy",
    ):
      pose = np.load(data_path("ref_pose", filename))
      self.assertEqual(pose.shape, (len(transition_body_names), 7))


if __name__ == "__main__":
  unittest.main()
