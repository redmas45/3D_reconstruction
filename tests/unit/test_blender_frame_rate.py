import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "blender"))

from frame_rate import blender_frame_rate


class BlenderFrameRateTests(unittest.TestCase):
    def test_represents_5994_fps_without_exceeding_blender_limit(self) -> None:
        nominal_fps, fps_base = blender_frame_rate(60_000 / 1_001)

        self.assertEqual(60, nominal_fps)
        self.assertAlmostEqual(60_000 / 1_001, nominal_fps / fps_base, places=9)

    def test_preserves_2997_fps(self) -> None:
        nominal_fps, fps_base = blender_frame_rate(30_000 / 1_001)

        self.assertEqual(30, nominal_fps)
        self.assertAlmostEqual(30_000 / 1_001, nominal_fps / fps_base, places=9)

    def test_rejects_invalid_frame_rates(self) -> None:
        for invalid_fps in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(invalid_fps=invalid_fps):
                with self.assertRaises(ValueError):
                    blender_frame_rate(invalid_fps)
