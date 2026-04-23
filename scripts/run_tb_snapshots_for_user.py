"""Regenerate TB snapshot PNGs (run from repo root with env_isaaclab)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
SCRIPT = ROOT / "scripts" / "plot_tb_scalar_window.py"
LOGDIR = ROOT / "logs/rsl_rl/g1_roller_skater_amp_ppo/2026-04-12_15-48-39_amp_lerp_lr_transition_v1"
OUT = ROOT / "artifacts" / "tb_snapshots"
STATUS = OUT / "_snapshot_status.txt"


def run(args: list[str]) -> None:
  r = subprocess.run([str(PY), str(SCRIPT), *args], capture_output=True, text=True)
  OUT.mkdir(parents=True, exist_ok=True)
  STATUS.write_text(
    f"args={args}\nreturncode={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}\n",
    encoding="utf-8",
  )
  r.check_returncode()


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  run(
    [
      str(LOGDIR),
      "--step-lo",
      "61000",
      "--step-hi",
      "62000",
      "-o",
      str(OUT / "amp_lr_trans_t1_iter61000_62000.png"),
      "--title",
      "amp_lerp_lr_transition_v1 | TB scalars 61000–62000 (transition early, full window)",
    ]
  )
  run(
    [
      str(LOGDIR),
      "--step-lo",
      "63600",
      "--step-hi",
      "65500",
      "--vline",
      "64500",
      "--include-learning-rate",
      "-o",
      str(OUT / "amp_lr_trans_t2_iter64500_transition_end.png"),
      "--title",
      "amp_lerp_lr_transition_v1 | around 64500 (red = lr/clip restore)",
    ]
  )
  STATUS.write_text(STATUS.read_text(encoding="utf-8") + "\nOK both plots\n", encoding="utf-8")


if __name__ == "__main__":
  main()
