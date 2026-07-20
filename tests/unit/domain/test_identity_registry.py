import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.identity_registry import build_identity_registry


class IdentityRegistryTests(unittest.TestCase):
    def test_same_video_and_track_produce_identical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "identity_fixture.avi"
            self._write_fixture_video(video_path)
            scene_report = {"tracks": [_track_fixture()]}

            first_registry = build_identity_registry(scene_report, video_path)
            second_registry = build_identity_registry(scene_report, video_path)

        self.assertEqual(first_registry, second_registry)
        identity = first_registry["identities"]["person_7"]
        self.assertEqual("visible_evidence", identity["appearance"]["source"])

    def _write_fixture_video(self, video_path: Path) -> None:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
        )
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(5):
                frame = np.full((64, 64, 3), (30, 80, 180), dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()


def _track_fixture() -> dict:
    return {
        "id": "person_7",
        "class_name": "person",
        "avg_confidence": 0.9,
        "detections": [
            {"frame": frame, "bbox": [12, 8, 52, 60], "confidence": 0.9}
            for frame in (0, 2, 4)
        ],
    }


if __name__ == "__main__":
    unittest.main()
