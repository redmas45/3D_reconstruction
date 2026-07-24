import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.motion_profile import (
    build_motion_profile,
    cadence_scale,
    motion_clip,
    synchronize_motion_profile,
)


class MotionProfileTests(unittest.TestCase):
    def test_visible_boundary_pose_sets_phase_and_evidence_source(self) -> None:
        track = {
            "detections": [
                _detection(9, left_ankle=(0.2, 0.9), right_ankle=(0.8, 0.7)),
                _detection(21, left_ankle=(0.7, 0.8), right_ankle=(0.3, 0.9)),
            ],
        }

        profile = build_motion_profile(track, (10, 20), 1.2, 0.95)

        self.assertEqual("yolo_pose_visible_boundaries", profile["source"])
        self.assertEqual([9, 21], [item["frame"] for item in profile["evidence"]])
        self.assertNotEqual(0.95, profile["phase_offset"])
        self.assertEqual("walk", profile["clip"])

    def test_missing_pose_uses_identity_phase_without_inventing_evidence(self) -> None:
        profile = build_motion_profile(
            {"detections": [{"frame": 9, "bbox": [0, 0, 10, 20]}]},
            (10, 20),
            0.0,
            0.72,
        )

        self.assertEqual("kinematic_fallback", profile["source"])
        self.assertEqual([], profile["evidence"])
        self.assertEqual(0.72, profile["phase_offset"])
        self.assertEqual("idle", profile["clip"])

    def test_clip_selection_and_cadence_are_speed_bounded(self) -> None:
        self.assertEqual("brisk_walk", motion_clip("walk", 1.8))
        self.assertEqual("run", motion_clip("walk", 3.0))
        self.assertLessEqual(cadence_scale("walk", 20.0), 1.45)

    def test_reasoning_speed_change_resynchronizes_motion_clip(self) -> None:
        entity = {
            "animation": {"state": "walk", "speed_meters_per_second": 2.8},
            "motion_profile": {
                "clip": "walk",
                "cadence_scale": 1.0,
            },
        }

        synchronize_motion_profile(entity)

        self.assertEqual("run", entity["motion_profile"]["clip"])
        self.assertGreater(entity["motion_profile"]["cadence_scale"], 0.0)

    def test_malformed_cached_pose_falls_back_without_crashing(self) -> None:
        track = {
            "detections": [
                {"frame": 9, "pose_evidence": {"keypoints": {"invalid": True}}},
            ],
        }

        profile = build_motion_profile(track, (10, 20), 1.0, 0.4)

        self.assertEqual(0.4, profile["phase_offset"])
        self.assertEqual(0.0, profile["pose_confidence"])
        self.assertEqual("kinematic_fallback", profile["source"])
        self.assertEqual([], profile["evidence"])


def _detection(
    frame_index: int,
    left_ankle: tuple[float, float],
    right_ankle: tuple[float, float],
) -> dict:
    keypoints = [[0.5, 0.5, 0.9] for _ in range(17)]
    keypoints[15] = [*left_ankle, 0.9]
    keypoints[16] = [*right_ankle, 0.9]
    return {
        "frame": frame_index,
        "bbox": [0, 0, 100, 200],
        "pose_evidence": {"keypoints": keypoints},
    }


if __name__ == "__main__":
    unittest.main()
