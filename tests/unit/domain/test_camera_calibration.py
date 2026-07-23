import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.camera_calibration import (
    build_camera_contract,
    calibration_confidence,
    estimate_ground_geometry,
    robust_height_prior,
)


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

    def test_camera_contract_does_not_claim_unmeasured_geometry(self) -> None:
        contract = build_camera_contract({
            "video": {"width": 640, "height": 480},
            "tracks": [],
            "camera_motion_report": {
                "classification": "dynamic_camera",
                "static_feature_inlier_score": 0.9,
                "camera_motion_fit_score": 0.9,
            },
        })

        self.assertEqual("generic_ground_prior", contract["mode"])
        self.assertEqual("experimental", contract["compatibility"]["status"])
        self.assertFalse(contract["motion_applied_to_render"])
        self.assertNotIn("ground_reprojection_score", contract["calibration_report"]["components"])
        self.assertLess(contract["calibration_confidence"], 0.50)
        self.assertEqual("stabilized_forensic_view", contract["presentation_mode"])

    def test_visible_person_contacts_fit_a_per_video_horizon(self) -> None:
        detections = []
        for height in range(50, 151, 10):
            bottom = 200 + round(1.5 * height)
            detections.append({
                "bbox": [200, bottom - height, 240, bottom],
                "confidence": 0.9,
            })

        report = estimate_ground_geometry(
            [{"class_name": "person", "detections": detections}],
            frame_width=640,
            frame_height=480,
        )

        self.assertTrue(report["supported"])
        self.assertAlmostEqual(200 / 480, report["horizon_normalized_y"], places=3)
        self.assertGreater(report["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
