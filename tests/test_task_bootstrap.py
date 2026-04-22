from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

bootstrap_spec = importlib.util.spec_from_file_location(
  "task_bootstrap",
  ROOT / "src" / "mjlab_roller" / "tasks" / "bootstrap.py",
)
assert bootstrap_spec is not None and bootstrap_spec.loader is not None
bootstrap_module = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(bootstrap_module)
bootstrap_task_registry = bootstrap_module.bootstrap_task_registry


class TaskBootstrapTests(unittest.TestCase):
  def test_bootstrap_imports_mjlab_tasks(self) -> None:
    with patch("importlib.import_module") as import_module:
      bootstrap_task_registry()
      self.assertEqual(import_module.call_count, 2)
      import_module.assert_any_call("mjlab.tasks")
      import_module.assert_any_call("mjlab_roller.tasks")


if __name__ == "__main__":
  unittest.main()
