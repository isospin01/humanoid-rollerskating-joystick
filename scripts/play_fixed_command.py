"""Play with a pinned (vx, vy, wz) command to verify policy I/O matching.

Usage:
    uv run python scripts/play_fixed_command.py --vx 1.0 --vy 0 --wz 0
    uv run python scripts/play_fixed_command.py --vx 0 --vy 0.3 --wz 0
    uv run python scripts/play_fixed_command.py --vx 0 --vy 0 --wz 1.0

Writes an mp4 per command under videos/manual/<tag>/.
Set MUJOCO_GL=egl for headless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab_roller.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab_roller.tasks.bootstrap import bootstrap_task_registry
from mjlab_roller.tasks.registry import (
    load_env_cfg,
    load_env_cls,
    load_rl_cfg,
    load_runner_cls,
)


@dataclass(frozen=True)
class Config:
    vx: float = 0.5
    vy: float = 0.0
    wz: float = 0.0
    checkpoint_file: str = "model_58999.pt"
    steps: int = 500
    task: str = "Mjlab-Roller-Joystick-Flat-Unitree-G1"
    video_dir: str = "videos/manual"
    device: str | None = None


def main(cfg: Config):
    bootstrap_task_registry()
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(cfg.task, play=True)
    env_cfg.scene.num_envs = 1
    rl_cfg = load_rl_cfg(cfg.task)

    env_cls = load_env_cls(cfg.task)
    env = env_cls(cfg=env_cfg, device=device, render_mode="rgb_array")

    tag = f"vx{cfg.vx:+.2f}_vy{cfg.vy:+.2f}_wz{cfg.wz:+.2f}".replace(".", "p")
    out_dir = Path(cfg.video_dir) / tag
    env = VideoRecorder(
        env,
        video_folder=out_dir,
        step_trigger=lambda step: step == 0,
        video_length=cfg.steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    from rsl_rl.runners import OnPolicyRunner
    runner_cls = load_runner_cls(cfg.task) or OnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), device=device)
    runner.load(cfg.checkpoint_file, load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd_term = env.unwrapped.command_manager.get_term("skate")
    pinned = torch.tensor([[cfg.vx, cfg.vy, cfg.wz]], device=device, dtype=torch.float32)

    obs, _ = env.reset()
    cmd_term.command_b[:] = pinned
    for step in range(cfg.steps):
        # Pin BEFORE step so policy sees the fixed command in its observation.
        cmd_term.command_b[:] = pinned
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        if bool(dones.any()):
            print(f"[step {step}] termination, resetting")
            obs, _ = env.reset()
            cmd_term.command_b[:] = pinned

    env.close()
    print(f"[done] video -> {out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(Config))
