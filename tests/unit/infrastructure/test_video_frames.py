import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.video_frames import (
    _distributed_visible_context_indexes,
    _foreground_mask,
    _masked_frame_median,
)


class ForensicContextFrameTests(unittest.TestCase):
    def test_foreground_mask_removes_nearby_visible_tracks(self) -> None:
        scene_report = {
            "tracks": [{
                "detections": [{
                    "frame": 100,
                    "bbox": [20, 25, 50, 80],
                    "confidence": 0.9,
                }],
            }],
        }

        mask = _foreground_mask((100, 120, 3), scene_report, frame_index=102)

        self.assertEqual(np.uint8(255), mask[50, 35])
        self.assertEqual(np.uint8(0), mask[5, 5])

    def test_foreground_mask_ignores_distant_track_observations(self) -> None:
        scene_report = {
            "tracks": [{
                "detections": [{
                    "frame": 10,
                    "bbox": [20, 25, 50, 80],
                    "confidence": 0.9,
                }],
            }],
        }

        mask = _foreground_mask((100, 120, 3), scene_report, frame_index=100)

        self.assertEqual(0, int(mask.max()))

    def test_masked_median_ignores_foreground_pixels(self) -> None:
        background = np.full((2, 2, 3), 20, dtype=np.uint8)
        foreground = background.copy()
        foreground[0, 0] = [240, 240, 240]
        clear_mask = np.zeros((2, 2), dtype=np.uint8)
        foreground_mask = clear_mask.copy()
        foreground_mask[0, 0] = 255

        median = _masked_frame_median(
            [background, foreground, background],
            [clear_mask, foreground_mask, clear_mask],
        )

        self.assertEqual([20, 20, 20], median[0, 0].tolist())

    def test_distributed_context_samples_only_visible_frames(self) -> None:
        indexes = _distributed_visible_context_indexes(
            maximum_frame_index=200,
            hidden_ranges=[[40, 160]],
        )

        self.assertGreaterEqual(len(indexes), 3)
        self.assertTrue(all(not 40 <= index <= 160 for index in indexes))


if __name__ == "__main__":
    unittest.main()
