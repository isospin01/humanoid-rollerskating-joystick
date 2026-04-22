# `tests/`

Lightweight sanity tests — no GPU, no long rollouts. Run with `make test`.

| Test | What it checks |
|------|----------------|
| `test_project_paths.py` | `project_root()` / `data_path()` resolve correctly from any cwd. |
| `test_task_bootstrap.py` | `bootstrap_task_registry()` is idempotent and populates mjlab tasks. |
| `test_task_registry_entries.py` | Every expected task ID is registered with env_cls / env_cfg / rl_cfg. |
| `test_joystick_teleop.py` | Pygame-free deadzone & smoothing math is correct. |
| `test_exporter_metadata.py` | ONNX exporter attaches obs-norm stats and joint metadata. |
| `test_amp_dataset.py` | AMP manifest parsing + clip shape/FPS validation. |
| `test_skater_domain_randomization.py` | DR ranges clamp correctly; no NaNs in reset tensors. |
| `test_transition_reference_alignment.py` | Reference-pose / init-keyframe joint ordering matches `CONTROLLED_JOINT_NAMES`. |

## Running

```bash
make test                                   # all tests
uv run python -m unittest tests.test_joystick_teleop   # one test module
```

## Adding a new test

- Keep it fast (< 2 s). Long integration tests belong in `scripts/` as standalone jobs.
- Avoid GPU requirements — these run in CI / on laptops.
- Follow the naming pattern `test_<area>.py`.
