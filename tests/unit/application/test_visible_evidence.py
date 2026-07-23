import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.visible_evidence import (
    export_visible_evidence,
    validate_visual_evidence_manifest,
)


class VisibleEvidenceTests(unittest.TestCase):
    def test_exported_manifest_never_contains_hidden_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            video_path = directory / "source.avi"
            _write_video(video_path)
            scene = _scene_fixture(video_path)
            manifest = export_visible_evidence(
                video_path, scene, [_plan_fixture()], directory / "evidence",
                {"max_global_keyframes": 4, "boundary_frames_per_side": 2, "crops_per_track": 2},
            )
            validate_visual_evidence_manifest(manifest, scene)
            frames = {item["frame"] for item in manifest["images"]}
            self.assertTrue(frames)
            self.assertFalse(any(10 <= frame <= 15 for frame in frames))
            self.assertTrue(any(item["kind"] == "entity_crop" for item in manifest["images"]))


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("Test video writer could not open")
    try:
        for frame_index in range(30):
            frame = np.full((48, 64, 3), frame_index * 5, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _scene_fixture(video_path: Path) -> dict:
    return {
        "video": {
            "path": str(video_path), "width": 64, "height": 48,
            "fps": 10.0, "frames": 30, "sha256": "test",
        },
        "visible_ranges": [{"start": 0, "end": 9}, {"start": 16, "end": 29}],
        "hidden_ranges": [{"start": 10, "end": 15}],
        "tracks": [{
            "id": "person_1",
            "detections": [
                {"frame": 8, "bbox": [10, 8, 30, 42]},
                {"frame": 9, "bbox": [12, 8, 32, 42]},
                {"frame": 16, "bbox": [20, 8, 40, 42]},
                {"frame": 17, "bbox": [22, 8, 42, 42]},
            ],
        }],
    }


def _plan_fixture() -> dict:
    return {
        "gap_index": 0,
        "hidden_range": {"start": 10, "end": 15},
        "entities": [{"id": "person_1"}],
    }


if __name__ == "__main__":
    unittest.main()
