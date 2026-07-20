import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from evaluate import _ssim
from evidence_compositor import render_evidence_reconstruction


class EvidenceCompositorTests(unittest.TestCase):
    def test_renderer_preserves_duration_and_real_background(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.mp4"
            output_path = Path(temporary_directory) / "reconstruction.mp4"
            self._write_source(source_path)
            render_evidence_reconstruction(
                str(output_path), str(source_path), self._plan(), {"tracks": []}, 160, 120, 10.0,
                {"plate_window_seconds": 0.3, "transition_seconds": 0.1, "show_uncertainty_paths": False},
            )
            capture = cv2.VideoCapture(str(output_path))
            self.assertEqual(10, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            success, frame = capture.read()
            capture.release()
            self.assertTrue(success)
            self.assertGreater(float(frame.mean()), 30.0)

    def test_identical_frames_have_perfect_ssim(self) -> None:
        frame = np.full((40, 60, 3), 127, dtype=np.uint8)
        self.assertAlmostEqual(1.0, _ssim(frame, frame), places=6)

    @staticmethod
    def _write_source(path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
        for frame_index in range(30):
            frame = np.full((120, 160, 3), (70, 90, 110), dtype=np.uint8)
            cv2.rectangle(frame, (20 + frame_index, 35), (45 + frame_index, 100), (20, 30, 220), -1)
            writer.write(frame)
        writer.release()

    @staticmethod
    def _plan() -> dict:
        path = [
            {"frame": frame, "bbox": [30 + frame, 35, 55 + frame, 100], "opacity": 1.0, "uncertainty_px": 5}
            for frame in range(10, 20)
        ]
        return {
            "hidden_range": {"start": 10, "end": 19},
            "overall_confidence": 0.8,
            "entities": [{
                "id": "person_1", "class_name": "person", "confidence": 0.8,
                "reference_before": {"frame": 9, "bbox": [29, 35, 54, 100]},
                "reference_after": {"frame": 20, "bbox": [40, 35, 65, 100]},
                "alternative_path_offset_px": 5, "path": path,
            }],
        }


if __name__ == "__main__":
    unittest.main()
