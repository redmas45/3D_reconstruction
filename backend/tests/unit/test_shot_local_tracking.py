"""Tracks must not carry one identity across a scene cut.

Detection runs over whole visible segments and a visible segment can contain a cut. The
tracker has no notion of scenes, so left alone it will hand the same identity to a person
in one city and a person in another, and the reconstruction will draw a trajectory
between them. On the project's own montage footage this affected 119 of 220 tracks.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scene_intelligence import build_tracks, shot_index_lookup

FPS = 30.0
SHOTS = [(0, 199), (220, 499)]


def _detection(frame: int, source_track_id: int, left: float) -> dict:
    return {
        "frame": frame,
        "segment_index": 0,
        "class_name": "person",
        "class_id": 0,
        "source_track_id": source_track_id,
        "bbox": [left, 300.0, left + 40.0, 420.0],
        "confidence": 0.9,
        "appearance": [0.5] * 8,
    }


def _walk(frames, source_track_id: int, start_left: float) -> list[dict]:
    return [
        _detection(frame, source_track_id, start_left + index * 2.0)
        for index, frame in enumerate(frames)
    ]


class TestShotIndexLookup:
    def test_a_video_with_no_shots_maps_everything_to_one_scene(self):
        lookup = shot_index_lookup(None)
        assert lookup(0) == 0 and lookup(10_000) == 0

    def test_frames_map_to_the_shot_containing_them(self):
        lookup = shot_index_lookup(SHOTS)
        assert lookup(100) == 0
        assert lookup(300) == 1

    def test_a_frame_in_a_transition_maps_to_no_shot(self):
        assert shot_index_lookup(SHOTS)(210) is None

    def test_shot_bounds_are_inclusive(self):
        lookup = shot_index_lookup(SHOTS)
        assert lookup(199) == 0 and lookup(220) == 1


class TestTracksDoNotSpanCuts:
    def test_one_tracker_identity_across_a_cut_becomes_two_tracks(self):
        """The tracker holds id 7 straight through the cut. That is one identity for two
        different people, and it must not survive into the scene report."""
        detections = _walk(range(100, 190, 6), 7, 100.0) + _walk(range(240, 330, 6), 7, 600.0)
        tracks = build_tracks(detections, fps=FPS, shots=SHOTS)
        assert len(tracks) == 2
        assert {track["shot_index"] for track in tracks} == {0, 1}

    def test_without_segmentation_the_same_input_stays_one_track(self):
        """Confirms the split is caused by the shot structure and not by the data."""
        detections = _walk(range(100, 190, 6), 7, 100.0) + _walk(range(240, 330, 6), 7, 600.0)
        assert len(build_tracks(detections, fps=FPS, shots=None)) == 1

    def test_no_track_extends_beyond_its_own_shot(self):
        detections = _walk(range(100, 190, 6), 7, 100.0) + _walk(range(240, 330, 6), 7, 600.0)
        for track in build_tracks(detections, fps=FPS, shots=SHOTS):
            start, end = SHOTS[track["shot_index"]]
            assert start <= track["first_frame"] and track["last_frame"] <= end

    def test_a_track_within_one_shot_is_left_whole(self):
        tracks = build_tracks(_walk(range(100, 190, 6), 7, 100.0), fps=FPS, shots=SHOTS)
        assert len(tracks) == 1
        assert tracks[0]["shot_index"] == 0

    def test_every_track_carries_the_shot_it_was_seen_in(self):
        detections = _walk(range(100, 190, 6), 7, 100.0) + _walk(range(240, 330, 6), 9, 600.0)
        tracks = build_tracks(detections, fps=FPS, shots=SHOTS)
        assert all(track["shot_index"] is not None for track in tracks)

    def test_detections_inside_a_transition_form_their_own_track(self):
        """They belong to no scene, so they must not be attached to either neighbour."""
        detections = (
            _walk(range(100, 190, 6), 7, 100.0)
            + _walk(range(202, 216, 3), 7, 400.0)
            + _walk(range(240, 330, 6), 7, 600.0)
        )
        tracks = build_tracks(detections, fps=FPS, shots=SHOTS)
        assert {track["shot_index"] for track in tracks} == {0, None, 1}
