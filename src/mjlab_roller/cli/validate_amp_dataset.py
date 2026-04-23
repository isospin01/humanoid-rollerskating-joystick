"""Validate the roller AMP dataset contract."""

from __future__ import annotations

import argparse
import json

from mjlab_roller.core.project_paths import resolve_project_path
from mjlab_roller.data.amp_dataset import validate_amp_dataset_dir


def main() -> None:
  parser = argparse.ArgumentParser(description="Validate the roller AMP dataset.")
  parser.add_argument("--dataset-dir", default="dataset/roller_push")
  parser.add_argument("--num-frames", type=int, default=5)
  args = parser.parse_args()

  summary = validate_amp_dataset_dir(resolve_project_path(args.dataset_dir), args.num_frames)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
