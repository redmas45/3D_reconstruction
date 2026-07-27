import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.actor_gap_renderer import (
    ActorRenderError,
    build_gap_specification,
    build_job_manifest,
    composite_sparse_frames,
    expand_to_source_frames,
    load_actor_layer,
    plan_sparse_frames,
    reconstruction_fps,
    sparse_frame_count,
    write_gap_video,
)
from domain.camera_projection import blender_camera_parameters
from domain.render_region import FULL_FRAME_REGION
from infrastructure.blender_protocol import frame_filename

FRAME_WIDTH = 320
FRAME_HEIGHT = 180


def _plan(target_fps=12.0, source_fps=30.0, frame_count=150, hidden_start=600):
    return {
        "gap_index": 2,
        "fps": source_fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / source_fps,
        "hidden_range": {"start": hidden_start, "end": hidden_start + frame_count - 1},
        "camera": {
            "projection_model": "pinhole_ground_plane_v2",
            "field_of_view_degrees": 54.0,
            "horizon_normalized_y": 0.42,
            "position": [0.0, 0.0, 3.0],
            "rotation_degrees": [68.0, 0.0, 0.0],
            "focal_length_mm": 35.0,
            "ground_mapping": {"near_y": 0.98, "far_y": 0.44},
        },
        "render": {"target_fps": target_fps, "engine": "BLENDER_EEVEE_NEXT", "cycles_samples": 4},
        "entities": [
            {
                "id": "person_7",
                "kind": "person",
                "appearance": {"upper_color": [0.2, 0.3, 0.7]},
                "animation": {"heading_degrees": 88.0},
                "path_prediction": {
                    "waypoints": [
                        {"role": "start", "frame": hidden_start, "world": [-2.0, 18.0, 0.0]},
                        {"role": "mid", "frame": hidden_start + 75, "world": [0.0, 18.0, 0.0]},
                        {"role": "end", "frame": hidden_start + 149, "world": [2.0, 18.0, 0.0]},
                    ],
                },
            },
        ],
    }


class TestSparseRate:
    def test_render_rate_is_capped_at_the_source_rate(self):
        assert reconstruction_fps(_plan(target_fps=60.0, source_fps=30.0)) == 30.0

    def test_configured_rate_is_used_when_below_the_source_rate(self):
        assert reconstruction_fps(_plan(target_fps=12.0, source_fps=30.0)) == 12.0

    def test_frame_count_follows_duration_and_rate(self):
        # 150 source frames at 30fps is 5s; at 12fps that is 60 samples.
        assert sparse_frame_count(_plan()) == 60

    def test_a_very_short_gap_still_renders_two_samples(self):
        assert sparse_frame_count(_plan(frame_count=2)) >= 2


class TestSparseFramePlanning:
    def test_one_entry_per_sparse_sample(self):
        frames = plan_sparse_frames(_plan(), FRAME_WIDTH, FRAME_HEIGHT)
        assert len(frames) == sparse_frame_count(_plan())

    def test_render_indexes_are_one_based_and_contiguous(self):
        frames = plan_sparse_frames(_plan(), FRAME_WIDTH, FRAME_HEIGHT)
        assert [frame.render_index for frame in frames] == list(range(1, len(frames) + 1))

    def test_source_frames_span_the_hidden_range(self):
        plan = _plan()
        frames = plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT)
        assert frames[0].source_frame == plan["hidden_range"]["start"]
        assert frames[-1].source_frame == plan["hidden_range"]["end"]

    def test_source_frames_never_leave_the_hidden_range(self):
        plan = _plan()
        frames = plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT)
        start, end = plan["hidden_range"]["start"], plan["hidden_range"]["end"]
        assert all(start <= frame.source_frame <= end for frame in frames)

    def test_every_sample_carries_a_region(self):
        frames = plan_sparse_frames(_plan(), FRAME_WIDTH, FRAME_HEIGHT)
        assert all(frame.region.coverage > 0.0 for frame in frames)

    def test_plan_without_entities_uses_the_full_frame(self):
        plan = _plan()
        plan["entities"] = []
        frames = plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT)
        assert all(frame.region == FULL_FRAME_REGION for frame in frames)




class TestJobManifest:
    def test_resolution_comes_from_the_source(self):
        manifest = build_job_manifest(_plan(), FRAME_WIDTH, FRAME_HEIGHT)
        assert manifest["resolution"] == [FRAME_WIDTH, FRAME_HEIGHT]

    def test_camera_is_derived_from_the_projection_not_copied(self):
        """The plan's stated lens is ignored in favour of the one its projection implies.

        A copied `focal_length_mm` would let Blender frame the scene differently from
        the projection the crop rectangle was computed with, and the actor would render
        outside its own crop.
        """
        plan = _plan()
        plan["camera"]["focal_length_mm"] = 200.0
        camera = build_job_manifest(plan, FRAME_WIDTH, FRAME_HEIGHT)["camera"]
        expected = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, plan["camera"])
        assert camera == expected
        assert camera["focal_length_mm"] != 200.0

    def test_camera_position_comes_from_the_contract(self):
        camera = build_job_manifest(_plan(), FRAME_WIDTH, FRAME_HEIGHT)["camera"]
        assert camera["position"] == [0.0, 0.0, 3.0]

    def test_camera_sensor_width_is_declared_for_blender(self):
        camera = build_job_manifest(_plan(), FRAME_WIDTH, FRAME_HEIGHT)["camera"]
        assert camera["sensor_width_mm"] == 36.0


class TestActorLayerLoading:
    def test_rgba_layer_is_read_with_alpha(self, tmp_path):
        path = tmp_path / "layer.png"
        layer = numpy.zeros((10, 10, 4), dtype=numpy.uint8)
        layer[..., 3] = 128
        cv2.imwrite(str(path), layer)
        assert load_actor_layer(path).shape[2] == 4

    def test_rgb_layer_gains_an_opaque_alpha(self, tmp_path):
        path = tmp_path / "layer.png"
        cv2.imwrite(str(path), numpy.zeros((10, 10, 3), dtype=numpy.uint8))
        loaded = load_actor_layer(path)
        assert loaded.shape[2] == 4
        assert (loaded[..., 3] == 255).all()

    def test_missing_layer_fails_cleanly(self, tmp_path):
        with pytest.raises(ActorRenderError, match="Could not read"):
            load_actor_layer(tmp_path / "absent.png")


class TestCompositingSparseFrames:
    def test_every_sparse_frame_is_composited(self, tmp_path):
        plate = numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 60, dtype=numpy.uint8)
        frames = plan_sparse_frames(_plan(frame_count=30), FRAME_WIDTH, FRAME_HEIGHT)[:3]
        for sparse_frame in frames:
            self._write_layer(tmp_path, plate, sparse_frame)
        composed = composite_sparse_frames(plate, frames, tmp_path)
        assert len(composed) == len(frames)
        assert all(frame.shape == plate.shape for frame in composed)

    def test_a_missing_layer_is_reported_not_skipped(self, tmp_path):
        plate = numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 60, dtype=numpy.uint8)
        frames = plan_sparse_frames(_plan(frame_count=30), FRAME_WIDTH, FRAME_HEIGHT)[:2]
        self._write_layer(tmp_path, plate, frames[0])
        with pytest.raises(ActorRenderError, match="missing"):
            composite_sparse_frames(plate, frames, tmp_path)

    @staticmethod
    def _write_layer(directory: Path, plate, sparse_frame) -> None:
        """Frame-sized with alpha only inside the region, as the service now emits."""
        left, top, right, bottom = sparse_frame.region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        layer = numpy.zeros((FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=numpy.uint8)
        layer[top:bottom, left:right, 3] = 255
        cv2.imwrite(str(directory / frame_filename(sparse_frame.render_index)), layer)


class TestTemporalExpansion:
    def test_expansion_restores_the_exact_source_frame_count(self):
        composed = [numpy.full((4, 4, 3), value, dtype=numpy.uint8) for value in (10, 20, 30)]
        assert len(expand_to_source_frames(composed, 150)) == 150

    def test_first_and_last_samples_anchor_the_expansion(self):
        composed = [numpy.full((4, 4, 3), value, dtype=numpy.uint8) for value in (10, 20, 30)]
        expanded = expand_to_source_frames(composed, 9)
        assert expanded[0].mean() == 10
        assert expanded[-1].mean() == 30

    def test_expansion_is_monotonic_through_the_samples(self):
        composed = [numpy.full((4, 4, 3), value, dtype=numpy.uint8) for value in (10, 20, 30)]
        means = [frame.mean() for frame in expand_to_source_frames(composed, 12)]
        assert means == sorted(means)

    def test_single_sample_expands_to_every_frame(self):
        composed = [numpy.full((4, 4, 3), 10, dtype=numpy.uint8)]
        assert len(expand_to_source_frames(composed, 40)) == 40

    def test_empty_input_is_rejected(self):
        with pytest.raises(ActorRenderError, match="No composited frames"):
            expand_to_source_frames([], 10)

    def test_non_positive_frame_count_is_rejected(self):
        composed = [numpy.zeros((4, 4, 3), dtype=numpy.uint8)]
        with pytest.raises(ActorRenderError, match="must be positive"):
            expand_to_source_frames(composed, 0)


class TestGapVideoWriting:
    def test_video_is_written_with_the_expected_frame_count(self, tmp_path):
        frames = [numpy.full((32, 48, 3), value, dtype=numpy.uint8) for value in (10, 20, 30)]
        output = write_gap_video(frames, 30.0, tmp_path / "gap.mp4")
        assert output.is_file()
        capture = cv2.VideoCapture(str(output))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
        finally:
            capture.release()

    def test_no_partial_file_survives(self, tmp_path):
        frames = [numpy.zeros((32, 48, 3), dtype=numpy.uint8)]
        output = write_gap_video(frames, 30.0, tmp_path / "gap.mp4")
        assert [path.name for path in output.parent.iterdir()] == ["gap.mp4"]

    def test_empty_frame_list_is_rejected(self, tmp_path):
        with pytest.raises(ActorRenderError, match="no frames"):
            write_gap_video([], 30.0, tmp_path / "gap.mp4")


class TestGapSpecification:
    @staticmethod
    def _specification(plan=None):
        plan = plan or _plan()
        return build_gap_specification(
            plan, plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT),
        )

    def test_the_class_name_is_carried_through_verbatim(self):
        assert self._specification()["actors"][0]["class_name"] == "person"

    def test_a_person_is_drawn_with_the_articulated_proxy(self):
        assert self._specification()["actors"][0]["proxy"] == "humanoid"

    def test_a_vehicle_class_picks_up_its_own_dimensions(self):
        plan = _plan()
        plan["entities"][0]["kind"] = "truck"
        actor = self._specification(plan)["actors"][0]
        assert actor["proxy"] == "vehicle"
        assert actor["dimensions"] == [7.0, 2.4, 2.9]

    def test_an_unknown_class_falls_back_rather_than_failing(self):
        plan = _plan()
        plan["entities"][0]["kind"] = "unicycle"
        actor = self._specification(plan)["actors"][0]
        assert actor["proxy"] == "box"
        assert actor["class_name"] == "unicycle"

    def test_a_carried_object_is_lifted_off_the_ground(self):
        plan = _plan()
        plan["entities"][0]["kind"] = "handbag"
        assert self._specification(plan)["actors"][0]["ground_offset"] > 0.5

    def test_one_keyframe_per_render_sample(self):
        """The regression that used to bite: keyframes and renders share a timeline.

        Emitting a keyframe per render index makes it structurally impossible for the
        actor's animation and its crop rectangles to describe different moments.
        """
        specification = self._specification()
        keyframes = specification["actors"][0]["keyframes"]
        assert len(keyframes) == specification["frame_count"]
        assert [key["frame"] for key in keyframes] == list(range(1, len(keyframes) + 1))

    def test_keyframes_preserve_world_coordinates_at_the_ends(self):
        keyframes = self._specification()["actors"][0]["keyframes"]
        assert keyframes[0]["location"] == [-2.0, 18.0, 0.0]
        assert keyframes[-1]["location"] == [2.0, 18.0, 0.0]

    def test_appearance_colour_is_converted_to_linear_light(self):
        colour = self._specification()["actors"][0]["color"]
        assert colour == pytest.approx([0.033105, 0.073239, 0.447988], abs=1e-5)

    def test_articulated_actors_carry_a_pose_per_sample(self):
        keyframes = self._specification()["actors"][0]["keyframes"]
        assert all("pose" in key for key in keyframes)
        assert "thigh.L" in keyframes[0]["pose"]["rotations"]

    def test_rigid_actors_carry_no_pose(self):
        plan = _plan()
        plan["entities"][0]["kind"] = "car"
        keyframes = self._specification(plan)["actors"][0]["keyframes"]
        assert all("pose" not in key for key in keyframes)

    def test_the_walk_cycle_actually_advances_across_the_gap(self):
        """A frozen pose would mean the figure slides instead of walking."""
        keyframes = self._specification()["actors"][0]["keyframes"]
        thighs = [key["pose"]["rotations"]["thigh.L"][0] for key in keyframes]
        assert len(set(thighs)) > len(thighs) // 2

    def test_a_stationary_actor_does_not_walk_on_the_spot(self):
        plan = _plan()
        for waypoint in plan["entities"][0]["path_prediction"]["waypoints"]:
            waypoint["world"] = [1.0, 18.0, 0.0]
        keyframes = self._specification(plan)["actors"][0]["keyframes"]
        thighs = {key["pose"]["rotations"]["thigh.L"][0] for key in keyframes}
        assert thighs == {0.0}

    def test_speed_is_reported_alongside_the_pose(self):
        keyframes = self._specification()["actors"][0]["keyframes"]
        assert all(key["speed"] >= 0.0 for key in keyframes)

    def test_the_declared_heading_wins_when_the_plan_states_one(self):
        keyframes = self._specification()["actors"][0]["keyframes"]
        assert all(key["heading_degrees"] == 88.0 for key in keyframes)

    def test_heading_is_derived_from_travel_when_the_plan_states_none(self):
        plan = _plan()
        plan["entities"][0]["animation"] = {}
        keyframes = self._specification(plan)["actors"][0]["keyframes"]
        # Travel is along +X at constant Y, which is 90 degrees from the +Y facing axis.
        assert keyframes[len(keyframes) // 2]["heading_degrees"] == pytest.approx(90.0)
