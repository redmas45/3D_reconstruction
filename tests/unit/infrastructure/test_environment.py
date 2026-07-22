import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.environment import load_environment_file


class EnvironmentLoaderTests(unittest.TestCase):
    def test_loads_values_without_overriding_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment_path = Path(temporary_directory) / ".env"
            environment_path.write_text("EXISTING=from-file\nNEW_VALUE='loaded'\n", encoding="utf-8")
            with patch.dict(os.environ, {"EXISTING": "process-value"}, clear=True):
                loaded_names = load_environment_file(environment_path)

                self.assertEqual("process-value", os.environ["EXISTING"])
                self.assertEqual("loaded", os.environ["NEW_VALUE"])
                self.assertEqual({"NEW_VALUE"}, loaded_names)


if __name__ == "__main__":
    unittest.main()
