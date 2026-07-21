import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.reconstruction_pipeline import (
    _parallel_gap_renderer_count,
    _render_blender_gaps,
    reserve_timeline_segment_paths,
)


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

            paths = reserve_timeline_segment_paths(timeline, temporary_root / "segments", None)

            self.assertFalse(paths[("visible", 0)].exists())
            self.assertFalse(paths[("visible", 1)].exists())
            self.assertFalse(paths[("hidden", 0)].exists())

    def test_parallel_gap_workers_are_bounded_by_gap_count(self) -> None:
        configuration = {"renderer": {"max_parallel_gap_renders": 2}}

        self.assertEqual(2, _parallel_gap_renderer_count(configuration, gap_count=8))
        self.assertEqual(1, _parallel_gap_renderer_count(configuration, gap_count=1))
        self.assertEqual(3, _parallel_gap_renderer_count({"renderer": {}}, gap_count=8))

    def test_blender_gaps_use_two_worker_threads(self) -> None:
        worker_names: set[str] = set()
        progress_details: list[str] = []
        start_barrier = threading.Barrier(2)
        context = SimpleNamespace(
            renderer_mode="blender",
            configuration={"renderer": {"max_parallel_gap_renders": 2}},
            prepared=SimpleNamespace(gap_selection={"hidden_ranges": [[0, 1], [2, 3]]}),
            cancellation_check=None,
        )

        def render_gap(
            unused_context: object,
            gap_index: int,
            progress_callback: Callable[[int, int], None],
        ) -> Path:
            worker_names.add(threading.current_thread().name)
            progress_callback(1, 2)
            start_barrier.wait(timeout=2.0)
            return Path(f"gap_{gap_index}.mp4")

        def report_progress(stage: str, progress: float, detail: str) -> None:
            del stage, progress
            progress_details.append(detail)

        with patch("application.reconstruction_pipeline._render_blender_hidden_segment", side_effect=render_gap):
            rendered_paths = _render_blender_gaps(context, report_progress)

        self.assertEqual({0, 1}, set(rendered_paths))
        self.assertEqual(2, len(worker_names))
        self.assertTrue(any("frame 1 of 2" in detail for detail in progress_details))


def _write_video(video_path: Path) -> None:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    for frame_index in range(12):
        writer.write(np.full((48, 64, 3), frame_index * 10, dtype=np.uint8))
    writer.release()


if __name__ == "__main__":
    unittest.main()
