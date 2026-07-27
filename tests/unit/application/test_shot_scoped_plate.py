"""Choosing which recovered background a gap composites onto.

Each shot has its own background, so each gap needs the one belonging to the take it
sits in. Gap selection guarantees a gap lies wholly inside one shot, but a selection
restored from a cache written before segmentation existed does not, and a run that
inherits one must still finish rather than crash on a lookup.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.reconstruction_pipeline import (
    PreparedReconstruction,
    _pipeline_shots,
    _plate_directory,
    _shot_for_plan,
)
from domain.shot_timeline import Shot

SHOTS = (
    Shot(index=0, start_frame=0, end_frame=463),
    Shot(index=1, start_frame=493, end_frame=935),
    Shot(index=2, start_frame=957, end_frame=1279),
)


def _plan(gap_index: int, start: int, end: int) -> dict:
    return {"gap_index": gap_index, "hidden_range": {"start": start, "end": end}}


class TestGapToShotAttribution:
    def test_a_gap_inside_a_shot_uses_that_shots_background(self):
        assert _shot_for_plan(SHOTS, _plan(0, 147, 325)) == 0
        assert _shot_for_plan(SHOTS, _plan(1, 660, 836)) == 1
        assert _shot_for_plan(SHOTS, _plan(2, 1000, 1200)) == 2

    def test_a_gap_touching_the_exact_bounds_of_a_shot_still_resolves(self):
        assert _shot_for_plan(SHOTS, _plan(0, 493, 935)) == 1

    def test_a_gap_straddling_a_cut_falls_back_to_the_take_it_starts_in(self):
        """Only reachable from a stale cache. Finishing with the closest real background
        beats failing, but it must not silently pick an unrelated take."""
        assert _shot_for_plan(SHOTS, _plan(3, 900, 1000)) == 1

    def test_a_gap_starting_in_a_transition_falls_back_to_the_first_take(self):
        assert _shot_for_plan(SHOTS, _plan(4, 470, 480)) == 0

    def test_a_gap_beyond_every_shot_does_not_raise(self):
        assert _shot_for_plan(SHOTS, _plan(5, 5_000, 5_100)) == 0

    def test_a_plan_without_a_hidden_range_does_not_raise(self):
        assert _shot_for_plan(SHOTS, {"gap_index": 6}) == 0


class TestPlateLayout:
    def test_a_single_take_video_keeps_the_flat_plate_directory(self):
        """One shot is the ordinary case, and its output layout should not change."""
        assert _plate_directory(Path("work"), 0, 1) == Path("work/plate")

    def test_a_montage_gets_one_directory_per_take(self):
        assert _plate_directory(Path("work"), 2, 9) == Path("work/plate/shot_02")

    def test_each_take_gets_a_distinct_directory(self):
        directories = {_plate_directory(Path("work"), index, 4) for index in range(4)}
        assert len(directories) == 4


class TestPipelineShots:
    def _prepared(self, shots) -> PreparedReconstruction:
        return PreparedReconstruction(
            video_info={"frames": 1_000, "width": 640, "height": 360},
            gap_selection={},
            segment_paths={},
            scene_report={},
            work_dir=Path("work"),
            blender_plan_paths=[],
            shots=shots,
        )

    def test_a_run_with_no_segmentation_reports_one_shot_over_the_whole_video(self):
        shots = _pipeline_shots(self._prepared(()))
        assert len(shots) == 1
        assert (shots[0].start_frame, shots[0].end_frame) == (0, 999)

    def test_reported_shots_are_rebuilt_in_order(self):
        shots = _pipeline_shots(self._prepared(((0, 399), (420, 999))))
        assert [(shot.index, shot.start_frame, shot.end_frame) for shot in shots] == [
            (0, 0, 399), (1, 420, 999),
        ]

    def test_rebuilt_shots_resolve_gaps(self):
        shots = _pipeline_shots(self._prepared(((0, 399), (420, 999))))
        assert _shot_for_plan(shots, _plan(0, 500, 600)) == 1
