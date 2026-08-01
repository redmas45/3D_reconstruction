"""Narrowing the evidence to one take before the camera is fitted to it.

Ground calibration pools the apparent height of every tracked person to infer the
horizon and the camera's height. Pooled across a montage, the median describes no camera
in the file — on the project's own test footage the per-shot median ranges from 77 to 272
pixels, so a single figure of 136 is wrong for every take. These tests fix the scoping
that prevents it.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.shot_scene import combine_shot_motion, scene_report_for_shot


def _track(track_id: str, shot_index, first: int, last: int) -> dict:
    return {
        "id": track_id,
        "class_name": "person",
        "shot_index": shot_index,
        "first_frame": first,
        "last_frame": last,
        "detections": [{"frame": first, "bbox": [0, 0, 10, 20]}],
    }


@pytest.fixture
def report() -> dict:
    return {
        "video": {"width": 1280, "height": 720, "frames": 1000, "fps": 30.0},
        "tracks": [
            _track("person_1", 0, 10, 200),
            _track("person_2", 0, 50, 300),
            _track("person_3", 1, 600, 800),
            _track("person_4", None, 440, 460),
        ],
        "people": [
            {"id": "person_1"}, {"id": "person_2"}, {"id": "person_3"}, {"id": "person_4"},
        ],
        "vehicles": [{"id": "car_1"}],
        "visible_ranges": [{"start": 0, "end": 399}, {"start": 500, "end": 999}],
        "hidden_ranges": [{"start": 100, "end": 200}, {"start": 700, "end": 800}],
    }


class TestScopingToAShot:
    def test_only_that_shots_tracks_survive(self, report):
        scoped = scene_report_for_shot(report, 0, (0, 430))
        assert [track["id"] for track in scoped["tracks"]] == ["person_1", "person_2"]

    def test_a_track_seen_only_during_a_transition_is_dropped(self, report):
        """During a dissolve the picture is a blend of two clips, so a box measured on
        it describes a position in neither."""
        for shot_index, bounds in ((0, (0, 430)), (1, (470, 999))):
            scoped = scene_report_for_shot(report, shot_index, bounds)
            assert "person_4" not in {track["id"] for track in scoped["tracks"]}

    def test_visible_ranges_are_clipped_to_the_shot(self, report):
        scoped = scene_report_for_shot(report, 1, (470, 999))
        assert scoped["visible_ranges"] == [{"start": 500, "end": 999}]

    def test_a_range_straddling_the_shot_edge_is_trimmed(self, report):
        scoped = scene_report_for_shot(report, 0, (0, 200))
        assert scoped["visible_ranges"] == [{"start": 0, "end": 200}]

    def test_hidden_ranges_are_clipped_too(self, report):
        scoped = scene_report_for_shot(report, 1, (470, 999))
        assert scoped["hidden_ranges"] == [{"start": 700, "end": 800}]

    def test_person_summaries_match_the_surviving_tracks(self, report):
        scoped = scene_report_for_shot(report, 1, (470, 999))
        assert [person["id"] for person in scoped["people"]] == ["person_3"]

    def test_the_shot_is_recorded_on_the_scoped_report(self, report):
        scoped = scene_report_for_shot(report, 1, (470, 999))
        assert scoped["shot"] == {"index": 1, "start": 470, "end": 999}

    def test_video_metadata_is_carried_through_unchanged(self, report):
        assert scene_report_for_shot(report, 0, (0, 430))["video"] == report["video"]

    def test_the_original_report_is_not_modified(self, report):
        scene_report_for_shot(report, 0, (0, 430))
        assert len(report["tracks"]) == 4
        assert len(report["visible_ranges"]) == 2


def _motion(classification: str, translation: float = 0.0, fit: float = 1.0) -> dict:
    return {
        "classification": classification,
        "sample_count": 2,
        "median_translation_pixels_per_frame": translation,
        "median_rotation_degrees_per_frame": 0.0,
        "median_scale_change_per_frame": 0.0,
        "static_feature_inlier_score": 1.0,
        "camera_motion_fit_score": fit,
        "pair_reports": [{"first_frame": 1, "second_frame": 6}],
    }


class TestCombiningMotionAcrossShots:
    def test_all_static_takes_combine_to_a_static_camera(self):
        combined = combine_shot_motion({0: _motion("static_camera"), 1: _motion("static_camera")})
        assert combined["classification"] == "static_camera"

    def test_one_moving_take_makes_the_whole_video_dynamic(self):
        """A single handheld shot must not hide behind eight locked-off ones."""
        combined = combine_shot_motion({
            0: _motion("static_camera"), 1: _motion("dynamic_camera", translation=6.0),
        })
        assert combined["classification"] == "dynamic_camera"

    def test_the_reported_motion_is_the_largest_seen_in_any_take(self):
        combined = combine_shot_motion({
            0: _motion("static_camera", translation=0.1),
            1: _motion("dynamic_camera", translation=6.0),
        })
        assert combined["median_translation_pixels_per_frame"] == 6.0

    def test_the_reported_fit_score_is_the_worst_seen_in_any_take(self):
        combined = combine_shot_motion({
            0: _motion("static_camera", fit=1.0), 1: _motion("static_camera", fit=0.3),
        })
        assert combined["camera_motion_fit_score"] == 0.3

    def test_each_takes_verdict_is_preserved_for_inspection(self):
        combined = combine_shot_motion({
            0: _motion("static_camera"), 1: _motion("dynamic_camera"),
        })
        assert combined["per_shot_classification"] == {"0": "static_camera", "1": "dynamic_camera"}

    def test_an_unclassifiable_take_prevents_a_static_verdict(self):
        combined = combine_shot_motion({
            0: _motion("static_camera"), 1: _motion("unclassified"),
        })
        assert combined["classification"] == "unclassified"

    def test_pair_reports_from_every_take_are_kept(self):
        combined = combine_shot_motion({0: _motion("static_camera"), 1: _motion("static_camera")})
        assert len(combined["pair_reports"]) == 2

    def test_no_measurements_at_all_reports_unclassified(self):
        assert combine_shot_motion({})["classification"] == "unclassified"

    def test_sample_counts_are_summed_across_takes(self):
        combined = combine_shot_motion({0: _motion("static_camera"), 1: _motion("static_camera")})
        assert combined["sample_count"] == 4
