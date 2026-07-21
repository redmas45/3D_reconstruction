import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import _center_error


class EvaluationMetricTests(unittest.TestCase):
    def test_center_error_penalizes_reconstruction_only_classes(self) -> None:
        error = _center_error({}, {"car": [(10.0, 10.0)]}, diagonal=100.0)

        self.assertEqual(1.0, error)


if __name__ == "__main__":
    unittest.main()
