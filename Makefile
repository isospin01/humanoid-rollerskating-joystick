# mjlab_roller — unified entrypoints.
# Run `make help` for the list.

TASK          ?= Mjlab-Roller-Joystick-Flat-Unitree-G1
TASK_AMP      ?= Mjlab-Roller-Joystick-AMP-Flat-Unitree-G1
TASK_LEGACY   ?= Mjlab-Roller-Flat-Unitree-G1
NUM_ENVS      ?= 1
PY            ?= uv run python
TB_LOGDIR     ?= logs/rsl_rl

.DEFAULT_GOAL := help
.PHONY: help install sync test smoke \
        train train-amp train-legacy train-smoke \
        play-joystick play-sampled play-zero \
        tensorboard registry compile clean-logs

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nmjlab_roller targets:\n\n"} \
	     /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } \
	     /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup
install: sync ## alias for `sync`
sync: ## Install dependencies via uv
	uv sync

##@ Verification
test: ## Run unit tests
	$(PY) -m unittest discover -s tests -v

compile: ## Byte-compile all sources (fast syntax check)
	$(PY) -m compileall src tests

registry: ## Print registered task IDs
	$(PY) -c "from mjlab_roller.tasks.bootstrap import bootstrap_task_registry; \
	bootstrap_task_registry(); \
	from mjlab_roller.tasks.registry import list_tasks; \
	import pprint; pprint.pprint(list_tasks())"

smoke: ## Headless env smoke: 1 reset + 1 step (uses scripts/smoke_env.py)
	MUJOCO_GL=egl $(PY) scripts/smoke_env.py

##@ Training
train: ## Full PPO training run (default joystick task)
	$(PY) -m mjlab_roller.cli.train $(TASK)

train-amp: ## AMP-augmented training run
	$(PY) -m mjlab_roller.cli.train $(TASK_AMP)

train-legacy: ## Legacy cycle-based baseline
	$(PY) -m mjlab_roller.cli.train $(TASK_LEGACY)

train-smoke: ## 2-iteration PPO smoke on CPU (sanity check)
	CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl $(PY) -m mjlab_roller.cli.train $(TASK) \
	    --env.scene.num_envs 8 --agent.max_iterations 2

##@ Play / visualize
play-joystick: ## Trained policy + USB gamepad
	$(PY) -m mjlab_roller.cli.play $(TASK) \
	    --agent trained --command-source joystick --num-envs $(NUM_ENVS)

play-sampled: ## Trained policy + scripted command sweep
	$(PY) -m mjlab_roller.cli.play $(TASK) \
	    --agent trained --command-source sampled --num-envs $(NUM_ENVS)

play-zero: ## Random (untrained) policy — visual sanity check
	$(PY) -m mjlab_roller.cli.play $(TASK) \
	    --agent zero --command-source sampled --num-envs $(NUM_ENVS)

##@ Monitoring
tensorboard: ## Launch TensorBoard on $(TB_LOGDIR)
	tensorboard --logdir $(TB_LOGDIR)

##@ Housekeeping
clean-logs: ## WARNING: deletes everything under logs/ (confirm first)
	@read -p "Delete all logs/ contents? [y/N] " ans && [ "$$ans" = "y" ] && rm -rf logs/* || echo "aborted"
