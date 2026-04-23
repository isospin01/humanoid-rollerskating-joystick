"""Helpers for making sure the mjlab task registry is populated."""

from __future__ import annotations

import importlib


def bootstrap_task_registry() -> None:
  importlib.import_module("mjlab.tasks")
  importlib.import_module("mjlab_roller.tasks")
