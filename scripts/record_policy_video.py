#!/usr/bin/env python3
"""Headless rollout + MP4 recording (no GUI). Uses OffscreenRenderer via rgb_array."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.utils.wrappers import VideoRecorder
from mjlab_roller.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab_roller.tasks.bootstrap import bootstrap_task_registry
from mjlab_roller.tasks.registry import load_env_cfg, load_env_cls, load_rl_cfg, load_runner_cls
from mjlab_roller.tasks.skater.mdp.command import SkaterJoystickCommandCfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner


def _latest_amp_checkpoint(exp_name: str = "g1_roller_skater_amp_ppo") -> Path:
  root = Path("logs") / "rsl_rl" / exp_name
  best: tuple[int, Path] | None = None
  for p in root.rglob("model_*.pt"):
    if "base_policy" in p.parts:
      continue
    m = re.match(r"model_(\d+)\.pt$", p.name)
    if not m:
      continue
    it = int(m.group(1))
    if best is None or it > best[0]:
      best = (it, p)
  if best is None:
    raise FileNotFoundError(f"No model_*.pt under {root.resolve()}")
  return best[1]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--task",
    default="Mjlab-Roller-Joystick-AMP-Flat-Unitree-G1",
    help="Registered task id",
  )
  parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to .pt checkpoint (default: latest under g1_roller_skater_amp_ppo)",
  )
  parser.add_argument("--duration-sec", type=float, default=30.0)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/videos"),
    help="Directory for the MP4 file",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--vx-range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override forward speed command sampling (m/s), e.g. 0.0 1.5",
  )
  parser.add_argument(
    "--resample-range",
    type=float,
    nargs=2,
    metavar=("MIN_S", "MAX_S"),
    default=None,
    help="Override command resampling interval in seconds",
  )
  parser.add_argument(
    "--name-prefix",
    default="amp_skater_rollout",
    help="MP4 filename prefix (before -step-0.mp4)",
  )
  args = parser.parse_args()

  configure_torch_backends()
  bootstrap_task_registry()

  ckpt = Path(args.checkpoint) if args.checkpoint else _latest_amp_checkpoint()
  if not ckpt.is_file():
    raise FileNotFoundError(ckpt)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)

  skate = env_cfg.commands.get("skate")
  if isinstance(skate, SkaterJoystickCommandCfg):
    if args.vx_range is not None:
      vx_lo, vx_hi = args.vx_range
      skate.ranges = SkaterJoystickCommandCfg.Ranges(
        lin_vel_x=(vx_lo, vx_hi),
        lin_vel_y=skate.ranges.lin_vel_y,
        ang_vel_z=skate.ranges.ang_vel_z,
      )
      print(f"[INFO] Command lin_vel_x range: ({vx_lo}, {vx_hi})")
    if args.resample_range is not None:
      r_lo, r_hi = args.resample_range
      skate.resampling_time_range = (r_lo, r_hi)
      print(f"[INFO] Command resampling_time_range: ({r_lo}, {r_hi})")

  render_mode = "rgb_array"
  env_cls = load_env_cls(args.task)
  base_env = env_cls(cfg=env_cfg, device=device, render_mode=render_mode)
  base_env.seed(args.seed)

  video_folder = args.output_dir.resolve()
  video_folder.mkdir(parents=True, exist_ok=True)

  fps = float(base_env.metadata.get("render_fps", 50))
  n_frames = max(1, int(round(args.duration_sec * fps)))

  wrapped = VideoRecorder(
    base_env,
    video_folder=video_folder,
    step_trigger=lambda step: step == 0,
    video_length=n_frames,
    name_prefix=args.name_prefix,
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(wrapped, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_optimizer=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(f"[INFO] Checkpoint: {ckpt}")
  print(f"[INFO] Recording ~{args.duration_sec}s at fps={fps} -> {n_frames} frames -> {video_folder}")

  with torch.no_grad():
    for _ in range(n_frames):
      obs = env.get_observations().to(device)
      actions = policy(obs)
      env.step(actions)

  env.close()
  print("[INFO] Done.")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
