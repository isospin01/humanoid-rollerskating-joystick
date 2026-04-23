"""Plot TensorBoard scalar windows (same data as TB) to PNG for offline review."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _series(ea: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
  try:
    events = ea.Scalars(tag)
  except KeyError:
    return [], []
  return [e.step for e in events], [e.value for e in events]


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("logdir", type=Path)
  p.add_argument("--step-lo", type=int, required=True)
  p.add_argument("--step-hi", type=int, required=True)
  p.add_argument("-o", "--output", type=Path, required=True)
  p.add_argument("--title", type=str, default="")
  p.add_argument("--vline", type=int, action="append", default=[], help="Vertical line(s) at step")
  p.add_argument(
    "--include-learning-rate",
    action="store_true",
    help="Third panel: Loss/learning_rate (for LR schedule change checks)",
  )
  args = p.parse_args()

  ea = EventAccumulator(str(args.logdir), size_guidance={"scalars": 0})
  ea.Reload()

  tags = [
    ("Episode_Reward/linear_velocity_track", "linear_velocity_track"),
    ("Loss/value_function", "value_function loss"),
  ]
  if args.include_learning_rate:
    tags.append(("Loss/learning_rate", "learning rate"))
  n = len(tags)
  fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n + 1), sharex=True)
  if n == 1:
    axes = [axes]
  lo, hi = args.step_lo, args.step_hi

  for ax, (tag, label) in zip(axes, tags):
    steps, vals = _series(ea, tag)
    steps_f = [s for s, v in zip(steps, vals) if lo <= s <= hi]
    vals_f = [v for s, v in zip(steps, vals) if lo <= s <= hi]
    ax.plot(steps_f, vals_f, linewidth=1.2, color="#1f77b4")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    for vl in args.vline:
      ax.axvline(vl, color="tab:red", linestyle="--", linewidth=1, alpha=0.7)

  axes[-1].set_xlabel("iteration")
  ttl = args.title or f"Scalars {lo}–{hi} (TensorBoard data)"
  fig.suptitle(ttl, fontsize=12)
  fig.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=150)
  plt.close(fig)
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
