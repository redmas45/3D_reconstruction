import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.camera_motion_estimator import summarize_camera_motion


class CameraMotionEstimatorTests(unittest.TestCase):
    def test_classifies_stable_background_as_static(self) -> None:
        report = summarize_camera_motion([_pair_report(0.1, 0.01, 0.0001)], 1)

        self.assertEqual("static_camera", report["classification"])

    def test_classifies_translation_above_threshold_as_dynamic(self) -> None:
        report = summarize_camera_motion([_pair_report(1.2, 0.01, 0.0001)], 1)

        self.assertEqual("dynamic_camera", report["classification"])
        self.assertFalse(report["render_transform_available"])

    def test_no_measurements_remain_unclassified(self) -> None:
        report = summarize_camera_motion([], 4)

        self.assertEqual("unclassified", report["classification"])


def _pair_report(translation: float, rotation: float, scale_change: float) -> dict:
    return {
        "translation_pixels_per_frame": translation,
        "rotation_degrees_per_frame": rotation,
        "scale_change_per_frame": scale_change,
        "inlier_ratio": 0.9,
        "fit_residual_pixels": 0.5,
    }


if __name__ == "__main__":
    unittest.main()
