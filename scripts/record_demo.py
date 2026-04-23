"""Record a demo video with scripted command sequences."""

import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

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
from rsl_rl.runners import OnPolicyRunner


def build_command_schedule(step_dt: float) -> list[tuple[int, float, float, float, str]]:
    """Return list of (start_step, vx, vy, wz, label) segments."""

    def s2step(seconds: float) -> int:
        return int(seconds / step_dt)

    schedule = [
        # Phase 1: straight forward (0-7s)
        (s2step(0.0), 1.0, 0.0, 0.0, "straight forward"),
        # Phase 2: turn right (7-12s)
        (s2step(7.0), 0.6, 0.0, -0.7, "turn right"),
        # Phase 3: new direction with slight lateral (12-19s)
        (s2step(12.0), 0.9, 0.15, 0.0, "forward-left"),
        # Phase 4: turn left (19-22s)
        (s2step(19.0), 0.5, 0.0, 0.8, "turn left"),
        # Phase 5: circle — sinusoidal wz (22-30s)
        (s2step(22.0), 0.8, 0.0, 0.0, "circle"),
    ]
    return schedule


def get_circle_wz(step: int, circle_start_step: int, step_dt: float) -> float:
    """Produce a smooth sinusoidal yaw-rate for the circle phase."""
    t = (step - circle_start_step) * step_dt
    period = 4.0
    return 0.9 * math.sin(2 * math.pi * t / period)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    configure_torch_backends()
    bootstrap_task_registry()
    task_id = "Mjlab-Roller-Joystick-Flat-Unitree-G1"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 5.0
    env_cfg.viewer.height = args.height
    env_cfg.viewer.width = args.width

    env_cls = load_env_cls(task_id)
    env = env_cls(cfg=env_cfg, device=device, render_mode="rgb_array")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "videos" / "demo"
    total_steps = int(args.duration / env.step_dt)

    env = VideoRecorder(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy(device=device)

    schedule = build_command_schedule(env.unwrapped.step_dt)
    circle_start_step = schedule[-1][0]

    skate_cmd = env.unwrapped.command_manager.get_term("skate")

    obs = env.get_observations()
    print(f"Recording {args.duration}s demo ({total_steps} steps) ...")

    for step in range(total_steps + 10):
        current_phase = schedule[0]
        for s_start, vx, vy, wz, label in schedule:
            if step >= s_start:
                current_phase = (s_start, vx, vy, wz, label)

        _, vx, vy, wz, label = current_phase
        if label == "circle":
            wz = get_circle_wz(step, circle_start_step, env.unwrapped.step_dt)

        skate_cmd.command_b[0, 0] = vx
        skate_cmd.command_b[0, 1] = vy
        skate_cmd.command_b[0, 2] = wz

        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        if step % 250 == 0:
            t = step * env.unwrapped.step_dt
            print(f"  step {step:5d} / {total_steps}  t={t:5.1f}s  phase={label}  cmd=[{vx:.2f}, {vy:.2f}, {wz:.2f}]")

    print(f"\nVideo saved to: {output_dir}/")
    env.close()


if __name__ == "__main__":
    main()
