import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_detection import attach_pose_evidence
from detect import _pose_sample_frames


class PoseDetectionTests(unittest.TestCase):
    def test_pose_is_attached_only_to_best_overlapping_person(self) -> None:
        detections = [
            {"class_name": "person", "bbox": [10, 10, 60, 100]},
            {"class_name": "person", "bbox": [100, 10, 150, 100]},
            {"class_name": "car", "bbox": [10, 120, 100, 180]},
        ]
        candidates = [
            {"bbox": [11, 11, 61, 101], "keypoints": [[0.5, 0.5, 0.9]] * 17},
        ]

        enriched = attach_pose_evidence(detections, candidates)

        self.assertIn("pose_evidence", enriched[0])
        self.assertNotIn("pose_evidence", enriched[1])
        self.assertNotIn("pose_evidence", enriched[2])
        self.assertNotIn("pose_evidence", detections[0])

    def test_low_overlap_pose_is_not_attached(self) -> None:
        detections = [{"class_name": "person", "bbox": [0, 0, 20, 20]}]
        candidates = [{"bbox": [15, 15, 35, 35], "keypoints": []}]

        enriched = attach_pose_evidence(detections, candidates)

        self.assertNotIn("pose_evidence", enriched[0])

    def test_pose_sampling_is_limited_to_visible_segment_boundaries(self) -> None:
        frames = _pose_sample_frames(100, 200, frame_stride=8, samples_per_boundary=2)

        self.assertEqual({100, 108, 188, 196}, frames)


if __name__ == "__main__":
    unittest.main()
