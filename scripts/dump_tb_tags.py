"""List scalar tags in a TensorBoard log directory."""
from __future__ import annotations

import argparse
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("logdir")
  args = p.parse_args()
  ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags().get("scalars", [])
  for t in sorted(tags):
    print(t)
  print(f"# total: {len(tags)}", file=sys.stderr)


if __name__ == "__main__":
  main()
