import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.camera_calibration import calibration_confidence, robust_height_prior


class CameraCalibrationTests(unittest.TestCase):
    def test_height_prior_rejects_zero_mad_outlier(self) -> None:
        detections = [
            {"bbox": [20, 20, 60, 120], "confidence": 0.9}
            for _ in range(6)
        ] + [{"bbox": [20, 20, 60, 320], "confidence": 0.9}]
        report = robust_height_prior(
            [{"id": "person_1", "class_name": "person", "detections": detections}], 640, 480
        )

        self.assertTrue(report["stable"])
        self.assertEqual(100.0, report["median_height_pixels"])
        self.assertEqual(6, report["tracks"][0]["accepted_observations"])

    def test_incomplete_calibration_is_capped_below_review_threshold(self) -> None:
        report = calibration_confidence({"camera_motion_fit_score": 1.0})

        self.assertLess(report["score"], 0.50)
        self.assertEqual("unreliable", report["tier"])


if __name__ == "__main__":
    unittest.main()
