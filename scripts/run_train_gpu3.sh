#!/usr/bin/env bash
set -euo pipefail

source /home/muchenxu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONPATH=/home/muchenxu/humanoid-rollerskatingv1/src
export MUJOCO_GL=egl
export WANDB_SILENT=true

exec python -u -m mjlab_roller.cli.train \
  Mjlab-Roller-Flat-Unitree-G1 \
  --env.scene.num_envs 32 \
  --agent.logger wandb \
  --agent.wandb_project humanoid-rollerskating \
  --agent.run_name train_gpu3 \
  --gpu-ids 3 \
  --enable-nan-guard True
