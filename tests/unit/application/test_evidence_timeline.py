import json
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import call, patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.reconstruction_pipeline import (
    TimelineRenderContext,
    _load_detections,
    _load_selection,
    _new_selection,
    _parallel_gap_renderer_count,
    _render_blender_gaps,
    _validate_source_resource_limits,
    _scaled_render_dimension,
    _selection_cache_is_compatible,
    _validate_runtime_dependencies,
    reserve_timeline_segment_paths,
)


class EvidenceTimelineTests(unittest.TestCase):
    def test_gap_cache_requires_matching_video_and_policy(self) -> None:
        info = {"width": 640, "height": 480, "fps": 30.0, "frames": 600, "sha256": "first"}
        gap_config = {
            "missing_fraction": 0.25,
            "min_seconds": 1.0,
            "max_seconds": 3.0,
            "context_seconds": 2.0,
        }
        selection = _new_selection(info, gap_config, random.Random(4))

        self.assertTrue(_selection_cache_is_compatible(selection, info, gap_config))
        self.assertFalse(_selection_cache_is_compatible(
            selection, {**info, "frames": 601}, gap_config,
        ))
        self.assertFalse(_selection_cache_is_compatible(
            selection, info, {**gap_config, "missing_fraction": 0.20},
        ))
        self.assertFalse(_selection_cache_is_compatible(
            selection, {**info, "sha256": "second"}, gap_config,
        ))

    def test_corrupt_selection_cache_is_recomputed_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            selection_path = work_directory / "gap_selection.json"
            selection_path.write_text("{", encoding="utf-8")
            info = {"width": 640, "height": 480, "fps": 30.0, "frames": 600, "sha256": "source"}
            configuration = {"gap": {"missing_fraction": 0.25, "min_seconds": 1.0, "max_seconds": 3.0}}

            selection = _load_selection(
                work_directory, info, configuration, random.Random(4), reuse_work=True,
            )

            self.assertTrue(_selection_cache_is_compatible(selection, info, configuration["gap"]))
            self.assertEqual([], list(work_directory.glob("*.tmp")))

    def test_corrupt_detection_cache_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            info = {"width": 640, "height": 480, "fps": 30.0, "frames": 600, "sha256": "source"}
            selection = _new_selection(info, {}, random.Random(4))
            configuration = {"yolo": {"frame_stride": 8, "downscale_width": 640}}
            with patch("application.reconstruction_pipeline.detect_scene_objects", return_value=[]):
                _load_detections(
                    Path("source.mp4"), work_directory, selection, configuration,
                    reuse_work=False, progress_callback=None, cancellation_check=None,
                )
            (work_directory / "detections.json").write_text("{", encoding="utf-8")

            with patch("application.reconstruction_pipeline.detect_scene_objects", return_value=[]) as detector:
                detections = _load_detections(
                    Path("source.mp4"), work_directory, selection, configuration,
                    reuse_work=True, progress_callback=None, cancellation_check=None,
                )

            self.assertEqual([], detections)
            detector.assert_called_once()

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

    def test_runtime_dependencies_are_checked_before_expensive_processing(self) -> None:
        with patch("application.reconstruction_pipeline.find_media_tool") as media_tool:
            with patch("application.reconstruction_pipeline.find_blender_executable") as blender_tool:
                _validate_runtime_dependencies("blender")
                self.assertEqual([call("ffmpeg"), call("ffprobe")], media_tool.call_args_list)
                blender_tool.assert_called_once_with()

        with patch("application.reconstruction_pipeline.find_media_tool") as media_tool:
            with patch("application.reconstruction_pipeline.find_blender_executable") as blender_tool:
                _validate_runtime_dependencies("2d")
                self.assertEqual([call("ffmpeg"), call("ffprobe")], media_tool.call_args_list)
                blender_tool.assert_not_called()

    def test_scaled_render_dimensions_are_even_and_never_smaller_than_two(self) -> None:
        configuration = {"renderer": {"production_scale_percent": 99}}

        self.assertEqual(634, _scaled_render_dimension(640, configuration))
        self.assertEqual(476, _scaled_render_dimension(480, configuration))
        self.assertEqual(
            2,
            _scaled_render_dimension(64, {"renderer": {"production_scale_percent": 1}}),
        )

    def test_extreme_source_contracts_are_rejected_before_decoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "4K pixel budget"):
            _validate_source_resource_limits(
                {"width": 7680, "height": 4320, "fps": 30.0, "frames": 300},
            )
        with self.assertRaisesRegex(ValueError, "120 fps"):
            _validate_source_resource_limits(
                {"width": 1920, "height": 1080, "fps": 240.0, "frames": 300},
            )
        with self.assertRaisesRegex(ValueError, "10-minute"):
            _validate_source_resource_limits(
                {"width": 1920, "height": 1080, "fps": 30.0, "frames": 18_001},
            )
        with self.assertRaisesRegex(ValueError, "10-minute"):
            _validate_source_resource_limits(
                {"width": 1920, "height": 1080, "fps": 30.0, "frames": 10 ** 1_000},
            )

    def test_blender_gaps_use_two_worker_threads(self) -> None:
        worker_names: set[str] = set()
        progress_details: list[str] = []
        start_barrier = threading.Barrier(2)
        context = TimelineRenderContext(
            video_path=Path("source.mp4"),
            renderer_mode="blender",
            configuration={"renderer": {"max_parallel_gap_renders": 2}},
            prepared=SimpleNamespace(gap_selection={"hidden_ranges": [[0, 1], [2, 3]]}),
            reuse_work=False,
            blender_rendered_paths={},
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

    def test_first_failed_gap_stops_running_sibling(self) -> None:
        sibling_started = threading.Event()
        sibling_stopped = threading.Event()
        context = TimelineRenderContext(
            video_path=Path("source.mp4"),
            renderer_mode="blender",
            configuration={"renderer": {"max_parallel_gap_renders": 2}},
            prepared=SimpleNamespace(gap_selection={"hidden_ranges": [[0, 1], [2, 3]]}),
            reuse_work=False,
            blender_rendered_paths={},
            cancellation_check=None,
        )

        def render_gap(
            worker_context: TimelineRenderContext,
            gap_index: int,
            progress_callback: Callable[[int, int], None],
        ) -> Path:
            del progress_callback
            if gap_index == 0:
                sibling_started.wait(timeout=1.0)
                raise RuntimeError("primary render failed")
            sibling_started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if worker_context.cancellation_check and worker_context.cancellation_check():
                    sibling_stopped.set()
                    raise RuntimeError("sibling stopped")
                time.sleep(0.01)
            raise AssertionError("Sibling render was not asked to stop")

        with patch("application.reconstruction_pipeline._render_blender_hidden_segment", side_effect=render_gap):
            with self.assertRaisesRegex(RuntimeError, "primary render failed"):
                _render_blender_gaps(context, None)

        self.assertTrue(sibling_stopped.is_set())

    def test_runtime_budget_stops_after_representative_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            plan_paths = []
            for gap_index, duration_seconds in enumerate((1.0, 3.0)):
                plan_path = work_directory / f"plan_{gap_index}.json"
                plan_path.write_text(json.dumps({
                    "gap_index": gap_index,
                    "fps": 30.0,
                    "duration_seconds": duration_seconds,
                    "render": {"target_fps": 10},
                    "entities": [{"fidelity_tier": "supported"}],
                }), encoding="utf-8")
                plan_paths.append(plan_path)
            context = TimelineRenderContext(
                video_path=Path("source.mp4"),
                renderer_mode="blender",
                configuration={"renderer": {
                    "max_parallel_gap_renders": 2,
                    "runtime_budget_enabled": True,
                    "maximum_predicted_render_seconds": 60,
                    "allow_runtime_budget_override": False,
                    "interactive_preview_approval": False,
                }},
                prepared=SimpleNamespace(
                    gap_selection={"hidden_ranges": [[0, 29], [30, 119]]},
                    blender_plan_paths=plan_paths,
                    work_dir=work_directory,
                ),
                reuse_work=False,
                blender_rendered_paths={},
                cancellation_check=None,
            )

            with (
                patch(
                    "application.reconstruction_pipeline._render_blender_hidden_segment",
                    return_value=work_directory / "representative.mp4",
                ) as render_mock,
                patch(
                    "application.reconstruction_pipeline._representative_elapsed_seconds",
                    return_value=120.0,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime budget"):
                    _render_blender_gaps(context, None)

        render_mock.assert_called_once()

    def test_keyboard_interrupt_stops_running_gap_workers(self) -> None:
        sibling_started = threading.Event()
        sibling_stopped = threading.Event()
        context = TimelineRenderContext(
            video_path=Path("source.mp4"),
            renderer_mode="blender",
            configuration={"renderer": {"max_parallel_gap_renders": 2}},
            prepared=SimpleNamespace(gap_selection={"hidden_ranges": [[0, 1], [2, 3]]}),
            reuse_work=False,
            blender_rendered_paths={},
            cancellation_check=None,
        )

        def render_gap(
            worker_context: TimelineRenderContext,
            gap_index: int,
            progress_callback: Callable[[int, int], None],
        ) -> Path:
            del progress_callback
            if gap_index == 0:
                sibling_started.wait(timeout=1.0)
                raise KeyboardInterrupt()
            sibling_started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if worker_context.cancellation_check and worker_context.cancellation_check():
                    sibling_stopped.set()
                    raise RuntimeError("sibling stopped")
                time.sleep(0.01)
            raise AssertionError("Sibling worker did not receive abort state")

        with patch("application.reconstruction_pipeline._render_blender_hidden_segment", side_effect=render_gap):
            with self.assertRaises(KeyboardInterrupt):
                _render_blender_gaps(context, None)

        self.assertTrue(sibling_stopped.is_set())


def _write_video(video_path: Path) -> None:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    for frame_index in range(12):
        writer.write(np.full((48, 64, 3), frame_index * 10, dtype=np.uint8))
    writer.release()


if __name__ == "__main__":
    unittest.main()
