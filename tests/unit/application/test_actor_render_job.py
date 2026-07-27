"""Drives `ActorRenderJob` against a fake Blender that writes real PNG layers.

The fake honours the protocol contract that matters: it refuses to render before a gap
is prepared, it writes a file per requested frame at exactly the size the requested
region implies, and it can be told to die mid-gap. That is enough to test resume,
recycling, and crash recovery without a GPU.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.actor_render_job import (
    GAP_RECYCLE_INTERVAL,
    ActorJobError,
    ActorRenderJob,
    actor_path_is_supported,
    gap_specification_digest,
    render_actor_gaps,
)
from domain.render_region import RenderRegion
from infrastructure.blender_protocol import frame_filename
from infrastructure.blender_service import BlenderServiceCrashed

FRAME_WIDTH = 320
FRAME_HEIGHT = 180


def _plan(gap_index=0, frame_count=60, hidden_start=100, target_fps=6.0, fps=30.0):
    return {
        "gap_index": gap_index,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "hidden_range": {"start": hidden_start, "end": hidden_start + frame_count - 1},
        "camera": {
            "projection_model": "pinhole_ground_plane_v2",
            "field_of_view_degrees": 54.0,
            "horizon_normalized_y": 0.42,
            "position": [0.0, 0.0, 3.0],
            "ground_mapping": {"near_y": 0.98, "far_y": 0.44},
            "calibration_confidence": 0.8,
        },
        "render": {"target_fps": target_fps, "engine": "BLENDER_EEVEE_NEXT", "cycles_samples": 4},
        "entities": [
            {
                "id": "person_1",
                "kind": "person",
                "appearance": {"upper_color": [0.3, 0.4, 0.6]},
                "path_prediction": {
                    "waypoints": [
                        {"role": "start", "frame": hidden_start, "world": [-1.5, 16.0, 0.0]},
                        {"role": "end", "frame": hidden_start + frame_count - 1,
                         "world": [1.5, 16.0, 0.0]},
                    ],
                },
            },
        ],
    }


class FakeBlenderService:
    """Enough of `BlenderService` to exercise the job loop."""

    def __init__(self, crash_after_frames=None) -> None:
        self.started = 0
        self.opened_manifests = []
        self.prepared_gaps = []
        self.rendered_indexes = []
        self.reset_calls = 0
        self.closed_jobs = 0
        self.shutdowns = 0
        self.is_alive = True
        self._prepared = False
        self._crash_after_frames = crash_after_frames
        self._frames_rendered = 0

    def start(self) -> dict:
        self.started += 1
        self.is_alive = True
        self._prepared = False
        return {"blender_version": "4.5.0", "engines": ["BLENDER_EEVEE_NEXT"]}

    def open_job(self, manifest_path: Path, timeout_seconds: float = 600.0) -> dict:
        self.opened_manifests.append(Path(manifest_path))
        return {"shell_built": True, "engine": "BLENDER_EEVEE_NEXT", "shadow_catcher": False}

    def prepare_gap(self, gap_index: int, storyboard_path: Path) -> dict:
        self.prepared_gaps.append(gap_index)
        self._prepared = True
        return {"gap_index": gap_index, "actor_ids": ["person_1"]}

    def render_frames(
        self, gap_index, frame_indexes, output_directory,
        timeout_seconds=None, on_progress=None, regions=None,
    ) -> dict:
        if not self._prepared:
            raise AssertionError("render_frames was called before prepare_gap")
        if regions is None or len(regions) != len(frame_indexes):
            raise AssertionError("every rendered frame requires exactly one region")
        for index, region in zip(frame_indexes, regions):
            if self._crash_after_frames is not None:
                if self._frames_rendered >= self._crash_after_frames:
                    self.is_alive = False
                    raise BlenderServiceCrashed("fake Blender died")
            self._write_layer(Path(output_directory), index, region)
            self._frames_rendered += 1
            self.rendered_indexes.append(index)
        return {"gap_index": gap_index, "rendered_frame_indexes": list(frame_indexes)}

    @staticmethod
    def _write_layer(output_directory: Path, index: int, region) -> None:
        """Frame-sized and transparent outside the border, exactly as Blender emits it."""
        left, top, right, bottom = RenderRegion(*region).pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        layer = numpy.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=numpy.uint8)
        layer[top:bottom, left:right, :3] = 200
        layer[top:bottom, left:right, 3] = 255
        output_directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_directory / frame_filename(index)), layer)

    def reset_gap(self) -> dict:
        self.reset_calls += 1
        self._prepared = False
        return {"cleared": True}

    def close_job(self) -> dict:
        self.closed_jobs += 1
        return {"closed": True}

    def shutdown(self, timeout_seconds: float = 10.0) -> None:
        self.shutdowns += 1
        self.is_alive = False


def _job(tmp_path, services, plate=None):
    """Hands out the next fake service each time the job starts a process."""
    queue = list(services)
    if plate is None:
        plate = numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 60, dtype=numpy.uint8)
    return ActorRenderJob(
        blender_executable=Path("blender"),
        project_root=Path("."),
        job_root=tmp_path / "job",
        plate_for_gap=lambda _: plate,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        source_fps=30.0,
        stall_timeout_seconds=30.0,
        service_factory=lambda: queue.pop(0),
    )


class TestSupportCheck:
    def test_a_calibrated_plan_with_entities_is_supported(self):
        supported, _ = actor_path_is_supported(_plan())
        assert supported is True

    def test_a_plan_without_entities_is_refused_with_a_reason(self):
        plan = _plan()
        plan["entities"] = []
        supported, reason = actor_path_is_supported(plan)
        assert supported is False
        assert "no entities" in reason

    def test_an_unprojectable_camera_is_refused_with_a_reason(self):
        plan = _plan()
        plan["camera"] = {"ground_mapping": {"near_y": 0.9, "far_y": 0.4}}
        supported, reason = actor_path_is_supported(plan)
        assert supported is False
        assert "forward projection" in reason


class TestDigest:
    def test_the_same_inputs_give_the_same_digest(self):
        assert gap_specification_digest({"a": 1}, {"b": 2}) == gap_specification_digest(
            {"a": 1}, {"b": 2},
        )

    def test_a_changed_gap_changes_the_digest(self):
        assert gap_specification_digest({"a": 1}, {"b": 2}) != gap_specification_digest(
            {"a": 2}, {"b": 2},
        )

    def test_a_changed_camera_changes_the_digest(self):
        """Same actors through a different lens are different pixels."""
        assert gap_specification_digest({"a": 1}, {"b": 2}) != gap_specification_digest(
            {"a": 1}, {"b": 3},
        )


class TestRenderingOneGap:
    def test_a_gap_produces_a_video_of_the_full_hidden_length(self, tmp_path):
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00")
        assert outcome.source_frame_count == 60
        capture = cv2.VideoCapture(str(outcome.video_path))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
        finally:
            capture.release()

    def test_only_the_sparse_frames_are_rendered(self, tmp_path):
        """60 source frames at 30fps is 2s; at 6fps that is 12 renders, not 60."""
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00")
        assert outcome.sparse_frame_count == 12
        assert len(service.rendered_indexes) == 12

    def test_the_job_is_opened_once_and_the_gap_prepared_once(self, tmp_path):
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        assert service.started == 1
        assert len(service.opened_manifests) == 1
        assert service.prepared_gaps == [0]

    def test_actors_are_cleared_after_the_gap(self, tmp_path):
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        assert service.reset_calls == 1

    def test_the_job_is_closed_and_the_process_stopped(self, tmp_path):
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        assert service.closed_jobs == 1
        assert service.shutdowns == 1

    def test_a_report_records_what_was_rendered(self, tmp_path):
        import json

        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        report = json.loads((tmp_path / "gap_00" / "actor_render_report.json").read_text())
        assert report["mode"] == "actor_composite"
        assert report["rendered_frames"] == 12
        assert report["resolution"] == [FRAME_WIDTH, FRAME_HEIGHT]

    def test_a_report_records_where_the_actors_came_from(self, tmp_path):
        """A sparse-looking gap has two very different causes — entities the planner
        never selected, and entities it selected that had no usable footage. Only a
        persisted record separates them after the run."""
        import json

        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        report = json.loads((tmp_path / "gap_00" / "actor_render_report.json").read_text())
        assert report["actor_source"]["mode"] == "rendered_geometry"

    def test_the_composite_keeps_the_plate_outside_the_actor_region(self, tmp_path):
        plate = numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 60, dtype=numpy.uint8)
        with _job(tmp_path, [FakeBlenderService()], plate) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00")
        capture = cv2.VideoCapture(str(outcome.video_path))
        try:
            _, frame = capture.read()
        finally:
            capture.release()
        # The top-left corner is well outside any actor's projected box.
        assert abs(int(frame[2, 2].mean()) - 60) <= 4

    def test_an_unsupported_plan_is_refused_rather_than_rendered_blank(self, tmp_path):
        plan = _plan()
        plan["entities"] = []
        with _job(tmp_path, [FakeBlenderService()]) as job:
            with pytest.raises(ActorJobError, match="cannot use the actor path"):
                job.render_gap(plan, tmp_path / "gap_00")


class TestResume:
    def test_a_second_run_re_renders_nothing(self, tmp_path):
        first = FakeBlenderService()
        with _job(tmp_path, [first]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        second = FakeBlenderService()
        with _job(tmp_path, [second]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00", reuse_work=True)
        assert outcome.rendered_frame_count == 0
        assert outcome.reused_frame_count == 12
        assert second.started == 0  # Blender was never even launched

    def test_only_the_missing_frames_are_re_rendered(self, tmp_path):
        with _job(tmp_path, [FakeBlenderService()]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        layers = sorted((tmp_path / "gap_00" / "layers").glob("*/frame_*.png"))
        for path in layers[:4]:
            path.unlink()
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00", reuse_work=True)
        assert outcome.rendered_frame_count == 4
        assert len(service.rendered_indexes) == 4

    def test_reuse_can_be_declined(self, tmp_path):
        with _job(tmp_path, [FakeBlenderService()]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00", reuse_work=False)
        assert outcome.rendered_frame_count == 12

    def test_a_changed_plan_does_not_reuse_the_old_layers(self, tmp_path):
        """Layers rendered for a different path would put the actor in the wrong place."""
        with _job(tmp_path, [FakeBlenderService()]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        moved = _plan()
        moved["entities"][0]["path_prediction"]["waypoints"][1]["world"] = [9.0, 16.0, 0.0]
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(moved, tmp_path / "gap_00", reuse_work=True)
        assert outcome.rendered_frame_count == 12

    def test_a_truncated_layer_is_treated_as_missing(self, tmp_path):
        with _job(tmp_path, [FakeBlenderService()]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        layer = sorted((tmp_path / "gap_00" / "layers").glob("*/frame_*.png"))[0]
        layer.write_bytes(b"not a png")
        service = FakeBlenderService()
        with _job(tmp_path, [service]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00", reuse_work=True)
        assert outcome.rendered_frame_count == 1


class TestCrashRecovery:
    def test_a_crash_is_retried_with_a_fresh_process(self, tmp_path):
        crashing = FakeBlenderService(crash_after_frames=5)
        healthy = FakeBlenderService()
        with _job(tmp_path, [crashing, healthy]) as job:
            outcome = job.render_gap(_plan(), tmp_path / "gap_00")
        assert outcome.sparse_frame_count == 12
        assert healthy.started == 1

    def test_the_retry_only_renders_what_was_lost(self, tmp_path):
        crashing = FakeBlenderService(crash_after_frames=5)
        healthy = FakeBlenderService()
        with _job(tmp_path, [crashing, healthy]) as job:
            job.render_gap(_plan(), tmp_path / "gap_00")
        assert len(healthy.rendered_indexes) == 7

    def test_a_persistent_crash_is_reported_rather_than_looping(self, tmp_path):
        services = [FakeBlenderService(crash_after_frames=0) for _ in range(4)]
        with _job(tmp_path, services) as job:
            with pytest.raises(ActorJobError, match="frames outstanding"):
                job.render_gap(_plan(), tmp_path / "gap_00")


class TestRenderingEveryGap:
    def test_every_gap_is_rendered_through_one_process(self, tmp_path):
        service = FakeBlenderService()
        plans = [_plan(gap_index=index, hidden_start=100 + index * 200) for index in range(3)]
        directories = [tmp_path / f"gap_{index:02d}" for index in range(3)]
        rendered = render_actor_gaps(plans, directories, _job(tmp_path, [service]))
        assert sorted(rendered) == [0, 1, 2]
        assert service.started == 1
        assert service.prepared_gaps == [0, 1, 2]

    def test_the_process_is_recycled_after_enough_gaps(self, tmp_path):
        services = [FakeBlenderService() for _ in range(3)]
        count = GAP_RECYCLE_INTERVAL + 1
        plans = [_plan(gap_index=index, hidden_start=100 + index * 200) for index in range(count)]
        directories = [tmp_path / f"gap_{index:02d}" for index in range(count)]
        render_actor_gaps(plans, directories, _job(tmp_path, services))
        assert services[0].prepared_gaps == list(range(GAP_RECYCLE_INTERVAL))
        assert services[1].prepared_gaps == [GAP_RECYCLE_INTERVAL]

    def test_progress_is_reported_per_gap(self, tmp_path):
        reported = []
        plans = [_plan(gap_index=index, hidden_start=100 + index * 200) for index in range(2)]
        directories = [tmp_path / f"gap_{index:02d}" for index in range(2)]
        render_actor_gaps(
            plans, directories, _job(tmp_path, [FakeBlenderService()]),
            progress_callback=lambda position, completed, total: reported.append(
                (position, completed, total),
            ),
        )
        assert {position for position, _, _ in reported} == {0, 1}
        assert all(total == 12 for _, _, total in reported)

    def test_mismatched_plans_and_directories_are_refused(self, tmp_path):
        with pytest.raises(ActorJobError, match="exactly one gap directory"):
            render_actor_gaps([_plan()], [], _job(tmp_path, [FakeBlenderService()]))
