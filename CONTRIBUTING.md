# Contributing

## Set up the environment

- Python 3.12 or 3.13 (see `pyproject.toml`).
- Install [uv](https://docs.astral.sh/uv/) and run from the repository root:

  ```bash
  uv sync
  ```

- The project depends on `mjlab` from a pinned Git revision (`pyproject.toml` → `[tool.uv.sources]`). First sync may take a while.

## Run checks before you open a pull request

```bash
python -m compileall src tests
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Headless environment smoke and short training smokes are documented in `README.md` under **Local validation**.

## What belongs in this repository

- **Include:** `src/`, `rsl_rl/`, `tests/`, `scripts/`, `docs/`, small `dataset/` reference and AMP arrays used by tests and training.
- **Do not commit:** training logs, W&B runs, checkpoints, large videos, or the `AMPdatasetgen/` tree (gigabytes of third-party and generated data). Paths are listed in `.gitignore`.

## Style

- Match existing module layout and naming in `src/mjlab_roller/`.
- Prefer small, reviewable changes with a short description of behavior and any reward or sim assumptions you touched.

## License

By contributing, you agree that your contributions are licensed under the same terms as the project (Apache-2.0, see `LICENSE`).
