import sys
import unittest
from pathlib import Path


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from scene_intelligence import build_tracks


def detection(frame: int, segment: int, source_id: int, x: int, appearance_index: int) -> dict:
    appearance = [0.0] * 32
    appearance[appearance_index] = 1.0
    return {
        "frame": frame,
        "segment_index": segment,
        "source_track_id": source_id,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "bbox": [x, 20, x + 30, 100],
        "appearance": appearance,
    }


def blended_detection(
    frame: int, segment: int, source_id: int, x: int, appearance: list[float],
) -> dict:
    item = detection(frame, segment, source_id, x, 0)
    return {**item, "appearance": appearance}


class SceneIntelligenceTests(unittest.TestCase):
    def test_appearance_keeps_people_distinct_across_multiple_gaps(self) -> None:
        detections = []
        for segment, base_frame in enumerate((0, 20, 40)):
            detections.extend([
                detection(base_frame, segment, 1, 10 + segment * 4, 0),
                detection(base_frame + 2, segment, 1, 12 + segment * 4, 0),
                detection(base_frame, segment, 2, 200 - segment * 4, 1),
                detection(base_frame + 2, segment, 2, 198 - segment * 4, 1),
            ])

        tracks = build_tracks(detections, fps=30.0)

        self.assertEqual(2, len(tracks))
        self.assertEqual([6, 6], sorted(track["frames_seen"] for track in tracks))
        self.assertTrue(all(track["continuity_confidence"] > 0.48 for track in tracks))

    def test_crossing_people_with_conflicting_appearance_never_merge(self) -> None:
        detections = [
            detection(0, 0, 1, 20, 0),
            detection(2, 0, 1, 80, 0),
            detection(0, 0, 2, 180, 1),
            detection(2, 0, 2, 120, 1),
            detection(20, 1, 3, 75, 1),
            detection(22, 1, 3, 20, 1),
            detection(20, 1, 4, 125, 0),
            detection(22, 1, 4, 180, 0),
        ]

        tracks = build_tracks(detections, fps=30.0)

        self.assertEqual(2, len(tracks))
        left_origin_track = min(tracks, key=lambda track: track["first_bbox"][0])
        right_origin_track = max(tracks, key=lambda track: track["first_bbox"][0])
        self.assertEqual("right", left_origin_track["direction"])
        self.assertEqual("left", right_origin_track["direction"])
        self.assertGreater(left_origin_track["last_bbox"][0], left_origin_track["first_bbox"][0])
        self.assertLess(right_origin_track["last_bbox"][0], right_origin_track["first_bbox"][0])

    def test_gradual_pairwise_matches_cannot_bridge_conflicting_appearances(self) -> None:
        first_appearance = [1.0, 0.0] + [0.0] * 30
        middle_appearance = [0.7, 0.7] + [0.0] * 30
        last_appearance = [0.0, 1.0] + [0.0] * 30
        detections = []
        for segment, base_frame, appearance in (
            (0, 0, first_appearance),
            (1, 20, middle_appearance),
            (2, 40, last_appearance),
        ):
            detections.extend([
                blended_detection(base_frame, segment, 1, 20 + segment * 4, appearance),
                blended_detection(base_frame + 2, segment, 1, 22 + segment * 4, appearance),
            ])

        tracks = build_tracks(detections, fps=30.0)

        self.assertEqual(2, len(tracks))
        self.assertEqual([2, 4], sorted(track["frames_seen"] for track in tracks))


if __name__ == "__main__":
    unittest.main()
