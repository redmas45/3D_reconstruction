import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.camera_calibration import (
    build_camera_contract,
    calibration_confidence,
    estimate_camera_height,
    estimate_ground_geometry,
    image_point_to_world,
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

    def test_camera_height_uses_visible_vertical_geometry(self) -> None:
        horizon_pixels = 200
        detections = []
        for bottom in range(260, 361, 20):
            top = horizon_pixels + round((bottom - horizon_pixels) * 0.5)
            detections.append({
                "bbox": [200, top, 240, bottom],
                "confidence": 0.9,
            })

        report = estimate_camera_height(
            [{"class_name": "person", "detections": detections}],
            horizon_pixels / 480,
            640,
            480,
        )

        self.assertTrue(report["supported"])
        self.assertAlmostEqual(3.44, report["height_meters"], places=2)

    def test_pinhole_mapping_places_higher_ground_points_farther_away(self) -> None:
        contract = build_camera_contract({
            "video": {"width": 1280, "height": 720},
            "tracks": [],
            "camera_motion_report": {
                "classification": "static_camera",
                "static_feature_inlier_score": 0.9,
                "camera_motion_fit_score": 0.9,
            },
        })

        near_point = image_point_to_world(640, 650, 1280, 720, contract)
        far_point = image_point_to_world(640, 360, 1280, 720, contract)

        self.assertEqual("pinhole_ground_plane_v2", contract["projection_model"])
        self.assertGreater(far_point[1], near_point[1])
        self.assertAlmostEqual(0.0, near_point[0], places=3)


if __name__ == "__main__":
    unittest.main()
