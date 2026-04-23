"""Helpers for locating project-owned files independent of cwd."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
  return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
  return project_root() / "dataset"


def data_path(*parts: str) -> Path:
  return data_dir().joinpath(*parts)


def resolve_project_path(path: str | Path) -> Path:
  candidate = Path(path)
  if candidate.is_absolute():
    return candidate
  if candidate.exists():
    return candidate.resolve()
  return (project_root() / candidate).resolve()
