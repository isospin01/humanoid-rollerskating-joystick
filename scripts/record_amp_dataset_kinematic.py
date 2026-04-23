#!/usr/bin/env python3
"""Replay plain AMP joint clips (N, 23) on the roller G1 MuJoCo model and export MP4.

AMP files must match CONTROLLED_JOINT_NAMES order (same as training). Root pose is
held at the standing-skate default; only the 23 actuated joints follow the clip —
adequate for previewing discriminator reference motion.

Usage (repo root):
  PYTHONPATH=src:. MUJOCO_GL=egl python scripts/record_amp_dataset_kinematic.py \\
    --dataset-dir dataset/rollerskating_amp --output-dir artifacts/videos/amp_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from moviepy import ImageSequenceClip

from mjlab_roller.assets.robots.roller.g1 import (
  STANDING_SKATE_INIT_KEYFRAME,
  get_g1_spec,
)
from mjlab_roller.core.control_spec import CONTROLLED_JOINT_NAMES


def _apply_floating_base(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  mujoco.mj_resetData(model, data)
  j_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
  if j_free < 0:
    raise RuntimeError("floating_base_joint not found")
  adr = model.jnt_qposadr[j_free]
  px, py, pz = STANDING_SKATE_INIT_KEYFRAME.pos
  data.qpos[adr : adr + 3] = px, py, pz
  qw, qx, qy, qz = STANDING_SKATE_INIT_KEYFRAME.rot
  data.qpos[adr + 3 : adr + 7] = qw, qx, qy, qz


def _set_controlled_from_row(
  model: mujoco.MjModel, data: mujoco.MjData, row: np.ndarray
) -> None:
  for i, name in enumerate(CONTROLLED_JOINT_NAMES):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
      raise RuntimeError(f"Joint not in model: {name}")
    qadr = model.jnt_qposadr[j]
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      raise RuntimeError(f"Expected hinge for {name}")
    data.qpos[qadr] = float(row[i])


def _load_clip(npy_path: Path) -> tuple[np.ndarray, float]:
  arr = np.load(npy_path, allow_pickle=False)
  if arr.ndim != 2 or arr.shape[1] != len(CONTROLLED_JOINT_NAMES):
    raise ValueError(
      f"{npy_path}: expected (N, {len(CONTROLLED_JOINT_NAMES)}), got {arr.shape}"
    )
  manifest_dir = npy_path.parent / "manifest.json"
  fps = 50.0
  if manifest_dir.is_file():
    for entry in json.loads(manifest_dir.read_text(encoding="utf-8")):
      if entry.get("file_name") == npy_path.name:
        fps = float(entry.get("fps", fps))
        break
  return arr.astype(np.float64), fps


def _render_clip(
  model: mujoco.MjModel,
  frames: np.ndarray,
  fps: float,
  width: int,
  height: int,
) -> list[np.ndarray]:
  data = mujoco.MjData(model)

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  cam.distance = 2.8
  cam.elevation = -18.0
  cam.azimuth = 135.0

  renderer = mujoco.Renderer(model, height=height, width=width)
  rgb_list: list[np.ndarray] = []
  for t in range(frames.shape[0]):
    _apply_floating_base(model, data)
    _set_controlled_from_row(model, data, frames[t])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    rgb_list.append(np.asarray(rgb))
  renderer.close()
  return rgb_list


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=Path("dataset/rollerskating_amp"),
    help="Directory with *_amp.npy and optional manifest.json",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/videos/amp_dataset"),
  )
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  out_dir = args.output_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  npy_files = sorted(dataset_dir.glob("*_amp.npy"))
  if not npy_files:
    print(f"No *_amp.npy under {dataset_dir}", file=sys.stderr)
    sys.exit(1)

  print("[INFO] Compiling roller G1 model...")
  model = get_g1_spec().compile()
  model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
  model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

  for npy_path in npy_files:
    frames, fps = _load_clip(npy_path)
    stem = npy_path.stem.replace("_amp", "")
    mp4_path = out_dir / f"{stem}_amp_sim.mp4"
    print(f"[INFO] {npy_path.name}: {frames.shape[0]} frames @ {fps} fps -> {mp4_path}")
    rgb_frames = _render_clip(model, frames, fps, args.width, args.height)
    uint8 = []
    for fr in rgb_frames:
      if fr.dtype != np.uint8:
        fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
      uint8.append(fr)
    clip = ImageSequenceClip(uint8, fps=fps)
    clip.write_videofile(str(mp4_path), logger=None)
    print(f"[INFO] Wrote {mp4_path}")

  print("[INFO] Done.")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
