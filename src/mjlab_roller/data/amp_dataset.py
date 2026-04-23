"""Dataset contract helpers for AMP motion clips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from mjlab_roller.core.control_spec import CONTROLLED_JOINT_COUNT, CONTROLLED_JOINT_NAMES

LEGACY_MOTION_DIM = 36
LEGACY_CONTROLLED_COLUMN_GROUPS: tuple[tuple[int, int], ...] = ((7, 26), (29, 33))
DEFAULT_DATASET_FPS = 50.0
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class AmpClipManifestEntry:
  clip_id: str
  file_name: str
  num_frames: int
  fps: float
  source_kind: str
  export_version: int
  joint_order: tuple[str, ...] = CONTROLLED_JOINT_NAMES
  source_path: str | None = None
  source_sha256: str | None = None


def _file_sha256(path: Path) -> str:
  digest = sha256()
  digest.update(path.read_bytes())
  return digest.hexdigest()


def project_legacy_motion_clip(clip: np.ndarray) -> np.ndarray:
  return np.concatenate([clip[:, start:end] for start, end in LEGACY_CONTROLLED_COLUMN_GROUPS], axis=1)


def normalize_amp_clip(clip: np.ndarray) -> np.ndarray:
  array = np.asarray(clip, dtype=np.float32)
  if array.ndim != 2:
    raise ValueError(f"AMP clip must be rank-2, got shape {array.shape}")
  if array.shape[1] == CONTROLLED_JOINT_COUNT:
    normalized = array
  elif array.shape[1] == LEGACY_MOTION_DIM:
    normalized = project_legacy_motion_clip(array)
  else:
    raise ValueError(
      f"AMP clip must have {CONTROLLED_JOINT_COUNT} or {LEGACY_MOTION_DIM} columns, got {array.shape[1]}"
    )
  if normalized.shape[0] == 0:
    raise ValueError("AMP clip must contain at least one frame")
  if not np.isfinite(normalized).all():
    raise ValueError("AMP clip contains NaN or Inf values")
  return normalized.astype(np.float32, copy=False)


def resample_clip(clip: np.ndarray, source_fps: float, target_fps: float = DEFAULT_DATASET_FPS) -> np.ndarray:
  if source_fps <= 0 or target_fps <= 0:
    raise ValueError("FPS values must be positive")
  normalized = normalize_amp_clip(clip)
  if normalized.shape[0] == 1 or abs(source_fps - target_fps) < 1e-6:
    return normalized
  duration = (normalized.shape[0] - 1) / source_fps
  target_count = max(2, int(round(duration * target_fps)) + 1)
  source_times = np.linspace(0.0, duration, normalized.shape[0], dtype=np.float32)
  target_times = np.linspace(0.0, duration, target_count, dtype=np.float32)
  resampled = np.empty((target_count, normalized.shape[1]), dtype=np.float32)
  for column in range(normalized.shape[1]):
    resampled[:, column] = np.interp(target_times, source_times, normalized[:, column])
  return resampled


def load_manifest(dataset_dir: str | Path) -> list[dict]:
  manifest_path = Path(dataset_dir) / MANIFEST_NAME
  if not manifest_path.exists():
    return []
  data = json.loads(manifest_path.read_text(encoding="utf-8"))
  if not isinstance(data, list):
    raise ValueError("AMP dataset manifest must be a JSON list")
  return data


def validate_amp_dataset_dir(dataset_dir: str | Path, num_frames: int) -> dict[str, object]:
  root = Path(dataset_dir)
  if not root.exists():
    raise FileNotFoundError(f"AMP dataset directory not found: {root}")
  clip_paths = sorted(path for path in root.glob("*.npy") if path.is_file())
  if not clip_paths:
    raise ValueError(f"No AMP clips found in {root}")
  clip_lengths: dict[str, int] = {}
  for clip_path in clip_paths:
    clip = normalize_amp_clip(np.load(clip_path, allow_pickle=False))
    if clip.shape[0] < num_frames:
      raise ValueError(
        f"AMP clip {clip_path.name} has {clip.shape[0]} frames, fewer than required num_frames={num_frames}"
      )
    clip_lengths[clip_path.name] = int(clip.shape[0])
  manifest_entries = load_manifest(root)
  manifest_files = {entry["file_name"] for entry in manifest_entries if "file_name" in entry}
  if manifest_entries and manifest_files != {path.name for path in clip_paths}:
    raise ValueError("AMP dataset manifest does not match the clip files present on disk")
  return {
    "dataset_dir": str(root),
    "num_clips": len(clip_paths),
    "num_frames": num_frames,
    "clip_lengths": clip_lengths,
  }


def write_manifest(
  dataset_dir: str | Path,
  entries: list[AmpClipManifestEntry],
) -> Path:
  manifest_path = Path(dataset_dir) / MANIFEST_NAME
  payload = [asdict(entry) for entry in entries]
  manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  return manifest_path


def make_manifest_entry(
  *,
  clip_id: str,
  file_name: str,
  num_frames: int,
  fps: float = DEFAULT_DATASET_FPS,
  source_kind: str,
  source_path: str | Path | None = None,
  export_version: int = 1,
) -> AmpClipManifestEntry:
  path_obj = None if source_path is None else Path(source_path)
  return AmpClipManifestEntry(
    clip_id=clip_id,
    file_name=file_name,
    num_frames=num_frames,
    fps=fps,
    source_kind=source_kind,
    export_version=export_version,
    source_path=None if path_obj is None else str(path_obj),
    source_sha256=None if path_obj is None or not path_obj.exists() else _file_sha256(path_obj),
  )
