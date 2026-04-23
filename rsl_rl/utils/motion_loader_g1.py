from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab_roller.core.project_paths import resolve_project_path
from mjlab_roller.data.amp_dataset import DEFAULT_DATASET_FPS, load_manifest, normalize_amp_clip


class G1_AMPLoader:
    """Loads fixed-width controlled-joint AMP clips."""

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files,
        preload_transitions=False,
        num_preload_transitions=1000000,
        num_frames=5,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames
        self.motion_dir = resolve_project_path(motion_files)

        manifest_entries = {entry["file_name"]: entry for entry in load_manifest(self.motion_dir)}
        clip_paths = sorted(path for path in Path(self.motion_dir).glob("*.npy") if path.is_file())
        if not clip_paths:
            raise ValueError(f"No AMP clips found in {self.motion_dir}")

        self.trajectories: list[torch.Tensor] = []
        self.trajectory_names: list[str] = []
        self.trajectory_frame_durations: list[float] = []
        self.trajectory_lens: list[float] = []
        self.trajectory_num_frames: list[int] = []

        for clip_path in clip_paths:
            raw_clip = np.load(clip_path, allow_pickle=False)
            clip = normalize_amp_clip(raw_clip)
            metadata = manifest_entries.get(clip_path.name, {})
            fps = float(metadata.get("fps", DEFAULT_DATASET_FPS))
            frame_duration = 1.0 / fps
            self.trajectories.append(torch.tensor(clip, dtype=torch.float32, device=self.device))
            self.trajectory_names.append(clip_path.name)
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_lens.append(max(0.0, (clip.shape[0] - 1) * frame_duration))
            self.trajectory_num_frames.append(int(clip.shape[0]))

        self.trajectory_idxs = np.arange(len(self.trajectories))
        self.trajectory_weights = np.ones(len(self.trajectories), dtype=np.float32)
        self.trajectory_weights /= self.trajectory_weights.sum()
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float32)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float32)
        self.trajectory_num_frames = np.asarray(self.trajectory_num_frames, dtype=np.int32)
        self._observation_dim = int(self.trajectories[0].shape[1])

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_sequences = self._sample_sequence_batch(traj_idxs, times)
        else:
            self.preloaded_sequences = None

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample_batch(self, traj_idxs):
        max_time = np.maximum(0.0, self.trajectory_lens[traj_idxs] - self.time_between_frames)
        return np.random.uniform(0.0, 1.0, size=len(traj_idxs)).astype(np.float32) * max_time

    def _sample_frame_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        frames = torch.zeros((len(traj_idxs), self._observation_dim), dtype=torch.float32, device=self.device)
        for traj_idx in np.unique(traj_idxs):
            mask = traj_idxs == traj_idx
            trajectory = self.trajectories[int(traj_idx)]
            frame_duration = float(self.trajectory_frame_durations[int(traj_idx)])
            positions = np.clip(times[mask] / frame_duration, 0.0, trajectory.shape[0] - 1)
            idx_low = np.floor(positions).astype(np.int32)
            idx_high = np.clip(idx_low + 1, 0, trajectory.shape[0] - 1)
            blend = torch.tensor((positions - idx_low)[:, None], dtype=torch.float32, device=self.device)
            start = trajectory[idx_low]
            end = trajectory[idx_high]
            frames[mask] = (1.0 - blend) * start + blend * end
        return frames

    def _sample_sequence_batch(self, traj_idxs: np.ndarray, times: np.ndarray) -> torch.Tensor:
        sequences = []
        offsets = np.arange(-(self.num_frames - 1), 1, dtype=np.float32) * self.time_between_frames
        for offset in offsets:
            frame_times = np.clip(times + offset, 0.0, self.trajectory_lens[traj_idxs])
            sequences.append(self._sample_frame_batch(traj_idxs, frame_times))
        return torch.stack(sequences, dim=1)

    def feed_forward_generator_23dof_multi(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if self.preloaded_sequences is not None:
                idxs = np.random.choice(self.preloaded_sequences.shape[0], size=mini_batch_size, replace=True)
                yield self.preloaded_sequences[idxs]
                continue
            traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
            times = self.traj_time_sample_batch(traj_idxs)
            yield self._sample_sequence_batch(traj_idxs, times)

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def num_motions(self):
        return len(self.trajectory_names)
