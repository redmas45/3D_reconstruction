"""The shot/transition model that keeps every later stage inside one scene.

The property that matters is that shots and transitions describe the same footage
without disagreeing — a frame the timeline calls "in shot 2" must not also be a frame
the transition list calls a dissolve. Deriving shots from transitions makes that
structural, and these tests pin the derivation.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.shot_timeline import (
    MINIMUM_SHOT_FRAMES,
    Shot,
    ShotTimelineError,
    Transition,
    clip_ranges_to_shot,
    coverage_report,
    shot_containing,
    shot_spanning,
    shots_from_transitions,
    single_shot_timeline,
    split_range_at_shots,
    validate_shots,
)

FRAME_COUNT = 1000


class TestConstruction:
    def test_a_video_with_no_cuts_is_one_shot_covering_everything(self):
        shots = shots_from_transitions((), FRAME_COUNT)
        assert len(shots) == 1
        assert (shots[0].start_frame, shots[0].end_frame) == (0, FRAME_COUNT - 1)

    def test_one_transition_produces_two_shots_that_exclude_it(self):
        shots = shots_from_transitions((Transition(400, 430),), FRAME_COUNT)
        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [
            (0, 399), (431, 999),
        ]

    def test_transition_frames_belong_to_no_shot(self):
        """A dissolve is a blend of two clips, so it is evidence for neither."""
        shots = shots_from_transitions((Transition(400, 430),), FRAME_COUNT)
        assert shot_containing(shots, 399) is not None
        assert shot_containing(shots, 415) is None
        assert shot_containing(shots, 431) is not None

    def test_shots_are_indexed_in_order(self):
        shots = shots_from_transitions(
            (Transition(200, 230), Transition(600, 630)), FRAME_COUNT,
        )
        assert [shot.index for shot in shots] == [0, 1, 2]

    def test_transitions_are_accepted_out_of_order(self):
        shots = shots_from_transitions(
            (Transition(600, 630), Transition(200, 230)), FRAME_COUNT,
        )
        assert [shot.start_frame for shot in shots] == [0, 231, 631]

    def test_a_transition_at_the_start_is_skipped_rather_than_producing_an_empty_shot(self):
        shots = shots_from_transitions((Transition(0, 40),), FRAME_COUNT)
        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(41, 999)]

    def test_a_transition_running_to_the_end_leaves_no_trailing_shot(self):
        shots = shots_from_transitions((Transition(900, FRAME_COUNT - 1),), FRAME_COUNT)
        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(0, 899)]

    def test_a_run_too_short_to_be_a_scene_is_discarded(self):
        """A sliver between two cuts cannot support a plate, so it is not offered as one."""
        sliver = MINIMUM_SHOT_FRAMES - 2
        shots = shots_from_transitions(
            (Transition(200, 230), Transition(231 + sliver, 300)), FRAME_COUNT,
        )
        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(0, 199), (301, 999)]

    def test_a_video_shorter_than_the_sliver_minimum_is_still_one_shot(self):
        """The minimum exists to discard fragments left between cuts. A short clip with
        no cuts is an ordinary video, and filtering it would leave nothing to render."""
        brief = MINIMUM_SHOT_FRAMES // 2
        shots = shots_from_transitions((), brief)
        assert [(shot.start_frame, shot.end_frame) for shot in shots] == [(0, brief - 1)]

    def test_overlapping_transitions_are_refused(self):
        with pytest.raises(ShotTimelineError, match="overlap"):
            shots_from_transitions((Transition(200, 300), Transition(250, 400)), FRAME_COUNT)

    def test_a_video_of_no_frames_is_refused(self):
        with pytest.raises(ShotTimelineError, match="cannot have"):
            shots_from_transitions((), 0)

    def test_a_video_that_is_entirely_transition_is_refused(self):
        with pytest.raises(ShotTimelineError, match="no scene"):
            shots_from_transitions((Transition(0, FRAME_COUNT - 1),), FRAME_COUNT)

    def test_single_shot_timeline_matches_the_no_transition_case(self):
        assert single_shot_timeline(FRAME_COUNT) == shots_from_transitions((), FRAME_COUNT)


class TestLookup:
    @pytest.fixture
    def shots(self):
        return shots_from_transitions(
            (Transition(400, 430), Transition(700, 730)), FRAME_COUNT,
        )

    def test_a_range_inside_one_shot_reports_that_shot(self, shots):
        assert shot_spanning(shots, 100, 200).index == 0

    def test_a_range_straddling_a_cut_reports_nothing(self, shots):
        """A gap across a cut has no single background to reconstruct against."""
        assert shot_spanning(shots, 380, 500) is None

    def test_a_range_ending_inside_a_transition_reports_nothing(self, shots):
        assert shot_spanning(shots, 380, 415) is None

    def test_frame_count_counts_both_endpoints(self):
        assert Shot(index=0, start_frame=10, end_frame=19).frame_count == 10

    def test_seconds_uses_the_frame_rate(self):
        assert Shot(index=0, start_frame=0, end_frame=59).seconds(30.0) == 2.0

    def test_a_zero_frame_rate_reports_no_duration_rather_than_dividing_by_zero(self):
        assert Shot(index=0, start_frame=0, end_frame=59).seconds(0.0) == 0.0


class TestRangeClipping:
    @pytest.fixture
    def shot(self):
        return Shot(index=1, start_frame=500, end_frame=799)

    def test_ranges_are_clipped_to_the_shot(self, shot):
        assert clip_ranges_to_shot([(400, 600), (700, 900)], shot) == [(500, 600), (700, 799)]

    def test_ranges_outside_the_shot_are_dropped(self, shot):
        assert clip_ranges_to_shot([(0, 100), (900, 950)], shot) == []

    def test_a_range_containing_the_whole_shot_yields_the_shot(self, shot):
        assert clip_ranges_to_shot([(0, 999)], shot) == [(500, 799)]

    def test_a_single_frame_range_survives(self, shot):
        assert clip_ranges_to_shot([(600, 600)], shot) == [(600, 600)]


class TestRangeSplitting:
    def test_a_range_crossing_a_cut_is_split_and_the_transition_dropped(self):
        shots = shots_from_transitions((Transition(400, 430),), FRAME_COUNT)
        pieces = split_range_at_shots(380, 460, shots)
        assert [(start, end, shot.index) for start, end, shot in pieces] == [
            (380, 399, 0), (431, 460, 1),
        ]

    def test_a_range_inside_one_shot_is_not_split(self):
        shots = shots_from_transitions((Transition(400, 430),), FRAME_COUNT)
        pieces = split_range_at_shots(100, 200, shots)
        assert len(pieces) == 1 and pieces[0][2].index == 0

    def test_a_range_entirely_inside_a_transition_yields_nothing(self):
        shots = shots_from_transitions((Transition(400, 430),), FRAME_COUNT)
        assert split_range_at_shots(410, 420, shots) == []


class TestCoverageReport:
    def test_transition_frames_are_reported_as_lost(self):
        report = coverage_report(
            shots_from_transitions((Transition(400, 430),), FRAME_COUNT), FRAME_COUNT,
        )
        assert report["frames_in_transitions"] == 31
        assert report["frames_in_shots"] == FRAME_COUNT - 31
        assert report["shot_count"] == 2

    def test_a_video_with_no_cuts_reports_full_coverage(self):
        report = coverage_report(single_shot_timeline(FRAME_COUNT), FRAME_COUNT)
        assert report["shot_coverage"] == 1.0
        assert report["frames_in_transitions"] == 0

    def test_the_longest_shot_is_reported(self):
        report = coverage_report(
            shots_from_transitions((Transition(100, 130),), FRAME_COUNT), FRAME_COUNT,
        )
        assert report["longest_shot_frames"] == FRAME_COUNT - 131


class TestValidation:
    def test_a_valid_timeline_passes(self):
        validate_shots(shots_from_transitions((Transition(400, 430),), FRAME_COUNT), FRAME_COUNT)

    def test_an_empty_timeline_is_refused(self):
        with pytest.raises(ShotTimelineError, match="at least one shot"):
            validate_shots([], FRAME_COUNT)

    def test_a_shot_past_the_end_of_the_video_is_refused(self):
        with pytest.raises(ShotTimelineError, match="outside a video"):
            validate_shots([Shot(index=0, start_frame=0, end_frame=FRAME_COUNT)], FRAME_COUNT)

    def test_overlapping_shots_are_refused(self):
        with pytest.raises(ShotTimelineError, match="overlap"):
            validate_shots(
                [Shot(index=0, start_frame=0, end_frame=500),
                 Shot(index=1, start_frame=400, end_frame=900)],
                FRAME_COUNT,
            )

    def test_misindexed_shots_are_refused(self):
        with pytest.raises(ShotTimelineError, match="indexed"):
            validate_shots([Shot(index=3, start_frame=0, end_frame=500)], FRAME_COUNT)

    def test_a_shot_ending_before_it_starts_is_refused(self):
        with pytest.raises(ShotTimelineError, match="ends before"):
            validate_shots([Shot(index=0, start_frame=500, end_frame=100)], FRAME_COUNT)
