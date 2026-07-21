import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.reconstruction_pipeline import DEFAULT_CONFIG_PATH
from domain.configuration import ConfigurationValidationError, load_validated_configuration, validate_configuration


class ConfigurationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = load_validated_configuration(PROJECT_ROOT / "config" / "reconstruction_config.json")

    def test_project_configuration_is_valid(self) -> None:
        validate_configuration(self.configuration)
        self.assertEqual(PROJECT_ROOT / "config" / "reconstruction_config.json", DEFAULT_CONFIG_PATH)

    def test_rejects_invalid_gap_duration_bounds(self) -> None:
        invalid_configuration = copy.deepcopy(self.configuration)
        invalid_configuration["gap"]["min_seconds"] = 3.0
        invalid_configuration["gap"]["max_seconds"] = 1.0

        with self.assertRaisesRegex(ConfigurationValidationError, "Gap duration bounds"):
            validate_configuration(invalid_configuration)

    def test_rejects_missing_configuration_section(self) -> None:
        invalid_configuration = copy.deepcopy(self.configuration)
        del invalid_configuration["scene"]

        with self.assertRaisesRegex(ConfigurationValidationError, "scene"):
            validate_configuration(invalid_configuration)

    def test_rejects_unknown_default_renderer(self) -> None:
        invalid_configuration = copy.deepcopy(self.configuration)
        invalid_configuration["renderer"]["default_mode"] = "unknown"

        with self.assertRaisesRegex(ConfigurationValidationError, "renderer.default_mode"):
            validate_configuration(invalid_configuration)

    def test_rejects_excessive_parallel_gap_renderers(self) -> None:
        invalid_configuration = copy.deepcopy(self.configuration)
        invalid_configuration["renderer"]["max_parallel_gap_renders"] = 5

        with self.assertRaisesRegex(ConfigurationValidationError, "max_parallel_gap_renders"):
            validate_configuration(invalid_configuration)


if __name__ == "__main__":
    unittest.main()
