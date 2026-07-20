import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.camera_calibration import build_camera_contract, image_point_to_world
from domain.path_prediction import (
    build_entity_prediction,
    heading_confidence_multiplier,
    position_residual_confidence_multiplier,
)


class PathPredictionTests(unittest.TestCase):
    def test_post_gap_position_remains_soft_observation(self) -> None:
        camera = build_camera_contract(_scene_report())
        track = {
            "id": "person_1",
            "class_name": "person",
            "continuity_confidence": 0.9,
            "detections": [
                {"frame": 8, "bbox": [90, 100, 110, 200], "confidence": 0.9},
                {"frame": 9, "bbox": [100, 100, 120, 200], "confidence": 0.9},
                {"frame": 21, "bbox": [300, 100, 320, 200], "confidence": 0.9},
                {"frame": 22, "bbox": [310, 100, 330, 200], "confidence": 0.9},
            ],
        }

        prediction = build_entity_prediction(track, (10, 20), 10.0, (640, 480), camera)

        self.assertIsNotNone(prediction)
        predicted_end = prediction["path_prediction"]["waypoints"][-1]["world"]
        observed_end = image_point_to_world(310.0, 200.0, 640, 480, camera)
        self.assertNotEqual(observed_end, predicted_end)
        self.assertGreater(prediction["boundary_evidence"]["post_gap_position_residual_meters"], 0.0)
        self.assertEqual("soft_consistency_check", prediction["path_prediction"]["post_gap_observation_role"])

    def test_large_heading_disagreement_receives_strong_penalty(self) -> None:
        self.assertEqual(0.35, heading_confidence_multiplier(120.0))

    def test_large_position_residual_reduces_confidence_without_forcing_arrival(self) -> None:
        self.assertEqual(0.30, position_residual_confidence_multiplier(12.0))

    def test_opposite_boundary_headings_reject_continuous_identity_claim(self) -> None:
        camera = build_camera_contract(_scene_report())
        track = {
            "id": "person_2",
            "class_name": "person",
            "continuity_confidence": 0.9,
            "detections": [
                {"frame": 8, "bbox": [80, 100, 100, 200], "confidence": 0.9},
                {"frame": 9, "bbox": [100, 100, 120, 200], "confidence": 0.9},
                {"frame": 21, "bbox": [300, 100, 320, 200], "confidence": 0.9},
                {"frame": 22, "bbox": [280, 100, 300, 200], "confidence": 0.9},
            ],
        }

        prediction = build_entity_prediction(track, (10, 20), 10.0, (640, 480), camera)

        self.assertIsNotNone(prediction)
        self.assertEqual("uncertain", prediction["lifecycle"])


def _scene_report() -> dict:
    return {"video": {"width": 640, "height": 480}, "tracks": []}


if __name__ == "__main__":
    unittest.main()
