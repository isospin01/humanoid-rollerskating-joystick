"""Build AMP datasets from retargeted poses or legacy bootstrap clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mjlab_roller.core.project_paths import project_root, resolve_project_path
from mjlab_roller.data.amp_dataset import (
  DEFAULT_DATASET_FPS,
  make_manifest_entry,
  normalize_amp_clip,
  resample_clip,
  validate_amp_dataset_dir,
  write_manifest,
)


def _write_dataset(clips: list[tuple[str, np.ndarray, str, Path | None]], output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_entries = []
  root = project_root()
  for clip_id, clip, source_kind, source_path in clips:
    output_path = output_dir / f"{clip_id}.npy"
    np.save(output_path, clip.astype(np.float32))
    manifest_source = source_path
    if source_path is not None:
      try:
        manifest_source = source_path.resolve().relative_to(root)
      except ValueError:
        manifest_source = source_path
    manifest_entries.append(
      make_manifest_entry(
        clip_id=clip_id,
        file_name=output_path.name,
        num_frames=int(clip.shape[0]),
        fps=DEFAULT_DATASET_FPS,
        source_kind=source_kind,
        source_path=manifest_source,
      )
    )
  write_manifest(output_dir, manifest_entries)
  validate_amp_dataset_dir(output_dir, num_frames=5)


def build_from_legacy(input_dir: Path, output_dir: Path) -> None:
  clips = []
  for source_path in sorted(input_dir.glob("*.npy")):
    clip = normalize_amp_clip(np.load(source_path, allow_pickle=False))
    clips.append((source_path.stem, clip, "legacy_bootstrap_projection", source_path))
  if not clips:
    raise ValueError(f"No legacy AMP clips found in {input_dir}")
  _write_dataset(clips, output_dir)


def build_from_retargeted_manifest(manifest_path: Path, output_dir: Path) -> None:
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if not isinstance(manifest, list):
    raise ValueError("Retarget manifest must be a JSON list")
  clips = []
  for entry in manifest:
    clip_id = entry["clip_id"]
    pose_path = resolve_project_path(entry["joint_positions_path"])
    pose_data = np.load(pose_path, allow_pickle=False)
    frame_start = int(entry.get("frame_start", 0))
    frame_end = int(entry.get("frame_end", pose_data.shape[0]))
    source_fps = float(entry.get("fps", DEFAULT_DATASET_FPS))
    clip = resample_clip(pose_data[frame_start:frame_end], source_fps=source_fps, target_fps=DEFAULT_DATASET_FPS)
    clips.append((clip_id, clip, "retargeted_video_pose", pose_path))
  if not clips:
    raise ValueError("Retarget manifest did not produce any AMP clips")
  _write_dataset(clips, output_dir)


def main() -> None:
  parser = argparse.ArgumentParser(description="Build the roller AMP dataset.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  legacy_parser = subparsers.add_parser("from-legacy", help="Project legacy 36-column bootstrap clips to 23-D AMP clips.")
  legacy_parser.add_argument("--input-dir", required=True)
  legacy_parser.add_argument("--output-dir", default="dataset/roller_push")

  retarget_parser = subparsers.add_parser("from-retargeted-poses", help="Build AMP clips from retargeted joint-position arrays.")
  retarget_parser.add_argument("--manifest", required=True)
  retarget_parser.add_argument("--output-dir", default="dataset/roller_push")

  args = parser.parse_args()
  if args.command == "from-legacy":
    build_from_legacy(resolve_project_path(args.input_dir), resolve_project_path(args.output_dir))
  else:
    build_from_retargeted_manifest(resolve_project_path(args.manifest), resolve_project_path(args.output_dir))


if __name__ == "__main__":
  main()
