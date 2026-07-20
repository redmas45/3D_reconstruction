import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.reconstruction_pipeline import write_timeline_segments


class EvidenceTimelineTests(unittest.TestCase):
    def test_hidden_truth_is_not_materialized_during_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_path = temporary_root / "source.mp4"
            _write_video(video_path)
            timeline = [
                {"kind": "visible", "index": 0, "start": 0, "end": 3},
                {"kind": "hidden", "index": 0, "start": 4, "end": 7},
                {"kind": "visible", "index": 1, "start": 8, "end": 11},
            ]

            paths = write_timeline_segments(
                video_path, timeline, temporary_root / "segments", False, None
            )

            self.assertTrue(paths[("visible", 0)].is_file())
            self.assertTrue(paths[("visible", 1)].is_file())
            self.assertFalse(paths[("hidden", 0)].exists())


def _write_video(video_path: Path) -> None:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    for frame_index in range(12):
        writer.write(np.full((48, 64, 3), frame_index * 10, dtype=np.uint8))
    writer.release()


if __name__ == "__main__":
    unittest.main()
