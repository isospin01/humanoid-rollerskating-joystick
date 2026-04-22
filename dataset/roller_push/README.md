This folder now uses the runtime AMP contract for the roller project:

- each `.npy` file is a `[T, 23]` `float32` array
- columns match the controlled joint order recorded in `manifest.json`
- the checked-in clips are bootstrap projections from the legacy dataset

Use `validate-amp-dataset` to check the folder, and replace the bootstrap clips with retargeted roller-skating demonstrations for task-valid AMP training.
