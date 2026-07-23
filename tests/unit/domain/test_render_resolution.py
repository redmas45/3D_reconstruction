import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.render_resolution import (
    adaptive_render_scale_percent,
    scaled_render_dimensions,
)


class RenderResolutionTests(unittest.TestCase):
    def test_small_source_is_never_upscaled(self) -> None:
        self.assertEqual(100, adaptive_render_scale_percent(640, 480, 45))

    def test_standard_source_reaches_sharp_minimum(self) -> None:
        scale = adaptive_render_scale_percent(1280, 720, 45)

        self.assertEqual(75, scale)
        self.assertEqual((960, 540), scaled_render_dimensions(1280, 720, scale))

    def test_large_source_is_bounded(self) -> None:
        scale = adaptive_render_scale_percent(3840, 2160, 75)
        width, height = scaled_render_dimensions(3840, 2160, scale)

        self.assertLessEqual(width, 1300)
        self.assertGreaterEqual(width, 1200)
        self.assertEqual(0, width % 2)
        self.assertEqual(0, height % 2)


if __name__ == "__main__":
    unittest.main()
