import tempfile
import unittest
from pathlib import Path

import cv2
import numpy

from application.browser_playback import write_browser_playback_video


class BrowserPlaybackTests(unittest.TestCase):
    def test_visible_frames_are_kept_and_gaps_use_their_plate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._write_source(source)
            plates = {
                0: numpy.full((24, 32, 3), (10, 40, 180), dtype=numpy.uint8),
                1: numpy.full((24, 32, 3), (20, 160, 40), dtype=numpy.uint8),
            }
            output = root / "playback.mp4"

            write_browser_playback_video(
                source,
                output,
                [(2, 3), (6, 6)],
                lambda index: plates[index],
            )

            frames = self._read_frames(output)
            self.assertEqual(8, len(frames))
            self.assertGreater(float(frames[0].mean()), 0.0)
            self.assertGreater(float(frames[1].mean()), 0.0)
            self.assertGreater(float(frames[4].mean()), 0.0)
            self.assertGreater(float(frames[5].mean()), 0.0)
            self.assertLess(abs(float(frames[2][:, :, 2].mean()) - 180), 15)
            self.assertLess(abs(float(frames[6][:, :, 1].mean()) - 160), 15)

    def test_missing_plate_transitions_between_sharp_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self._write_source(source)
            output = root / "boundary-fallback.mp4"

            write_browser_playback_video(source, output, [(2, 3)], lambda _index: None)

            frames = self._read_frames(output)
            before_mean = float(frames[1].mean())
            after_mean = float(frames[4].mean())
            self.assertGreater(float(frames[2].mean()), before_mean + 5)
            self.assertGreater(float(frames[3].mean()), float(frames[2].mean()) + 5)
            self.assertLess(float(frames[3].mean()), after_mean - 5)

    @staticmethod
    def _write_source(path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (32, 24))
        for index in range(8):
            writer.write(numpy.full((24, 32, 3), index * 20 + 20, dtype=numpy.uint8))
        writer.release()

    @staticmethod
    def _read_frames(path: Path) -> list[numpy.ndarray]:
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    return frames
                frames.append(frame)
        finally:
            capture.release()
