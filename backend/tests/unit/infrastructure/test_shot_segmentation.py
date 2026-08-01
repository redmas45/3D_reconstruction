"""Shot detection, tested against a video whose cuts are known exactly.

The hard case is not finding a cut — it is *not* finding one when somebody walks past
the lens. A person filling the frame removes almost all the background there is to match,
which looks identical to a scene change by every measure taken at that instant. On real
footage this is the only candidate a single-take video produces, so a detector that
cannot reject it reports cuts in videos that have none.

So the fixture contains both: a genuine cut, and an occlusion that blanks the frame just
as thoroughly without the camera ever moving. A detector that finds two shots and
attributes the occlusion correctly has made the distinction that matters.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.shot_segmentation import (
    ShotDetectionError,
    detect_shots,
    shots_from_report,
)

WIDTH = 480
HEIGHT = 270
SHOT_LENGTH = 220
CUT_FRAME = SHOT_LENGTH
TOTAL_FRAMES = SHOT_LENGTH * 2
OCCLUSION_RANGE = (96, 116)
OCCLUSION_BORDER = 8


def _textured_background(seed: int) -> numpy.ndarray:
    """A busy but static background. Random blocks survive video compression, where
    per-pixel noise would be smoothed into a featureless wash and match nothing."""
    generator = numpy.random.default_rng(seed)
    frame = numpy.zeros((HEIGHT, WIDTH, 3), dtype=numpy.uint8)
    for _ in range(260):
        x = int(generator.integers(0, WIDTH - 30))
        y = int(generator.integers(0, HEIGHT - 30))
        size = int(generator.integers(8, 30))
        colour = tuple(int(value) for value in generator.integers(30, 235, size=3))
        cv2.rectangle(frame, (x, y), (x + size, y + size), colour, -1)
    return frame


def _write(path: Path, frames) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _walker(frame: numpy.ndarray, frame_index: int) -> numpy.ndarray:
    """A figure crossing the frame: motion that must not be mistaken for a cut."""
    moving = frame.copy()
    left = int((frame_index / SHOT_LENGTH) * (WIDTH - 40))
    cv2.rectangle(moving, (left, HEIGHT - 90), (left + 40, HEIGHT - 10), (250, 250, 250), -1)
    return moving


@pytest.fixture
def two_shot_video(tmp_path):
    """Two takes with one cut, and one near-total occlusion inside the first take."""
    first = _textured_background(seed=1)
    second = _textured_background(seed=2)
    frames = []
    for frame_index in range(TOTAL_FRAMES):
        if frame_index < CUT_FRAME:
            frame = _walker(first, frame_index)
            if OCCLUSION_RANGE[0] <= frame_index <= OCCLUSION_RANGE[1]:
                # Someone passing right in front of the lens: everything but a sliver of
                # the border is gone, so there is almost nothing left to match against.
                frame = frame.copy()
                frame[OCCLUSION_BORDER:HEIGHT - OCCLUSION_BORDER,
                      OCCLUSION_BORDER:WIDTH - OCCLUSION_BORDER] = (24, 22, 20)
        else:
            frame = _walker(second, frame_index - CUT_FRAME)
        frames.append(frame)
    path = tmp_path / "two_shots.mp4"
    _write(path, frames)
    return path


@pytest.fixture
def single_shot_video(tmp_path):
    background = _textured_background(seed=7)
    path = tmp_path / "one_shot.mp4"
    _write(path, [_walker(background, index) for index in range(TOTAL_FRAMES)])
    return path


class TestCutDetection:
    def test_a_cut_is_found(self, two_shot_video):
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        assert report["shot_count"] == 2, report

    def test_the_transition_brackets_the_real_cut(self, two_shot_video):
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        transition = report["transitions"][0]
        assert transition["start"] <= CUT_FRAME <= transition["end"] + 1

    def test_the_two_shots_land_on_either_side_of_the_cut(self, two_shot_video):
        shots = shots_from_report(detect_shots(two_shot_video, TOTAL_FRAMES), TOTAL_FRAMES)
        assert shots[0].end_frame < CUT_FRAME
        assert shots[1].start_frame >= CUT_FRAME

    def test_the_reason_is_recorded_as_a_successful_analysis(self, two_shot_video):
        assert detect_shots(two_shot_video, TOTAL_FRAMES)["reason"] == "ok"


class TestOcclusionIsNotACut:
    def test_a_person_filling_the_lens_does_not_split_the_shot(self, two_shot_video):
        """The occlusion sits inside the first take; finding three shots means it was
        mistaken for a scene change."""
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        assert report["shot_count"] == 2, report

    def test_no_transition_covers_the_occlusion(self, two_shot_video):
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        occluded = range(*OCCLUSION_RANGE)
        for transition in report["transitions"]:
            assert not set(range(transition["start"], transition["end"] + 1)) & set(occluded)

    def test_the_occlusion_is_raised_and_then_rejected(self, two_shot_video):
        """It should be caught as a candidate — that is the detector working — and then
        thrown out by the verification pass rather than never noticed at all."""
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        assert report["candidates_found"] >= 2
        assert report["candidates_rejected_as_occlusion"] >= 1

    def test_a_video_with_no_cuts_stays_one_shot(self, single_shot_video):
        report = detect_shots(single_shot_video, TOTAL_FRAMES)
        assert report["shot_count"] == 1
        assert report["shot_coverage"] == 1.0

    def test_a_crossing_figure_alone_raises_no_candidate(self, single_shot_video):
        assert detect_shots(single_shot_video, TOTAL_FRAMES)["candidates_found"] == 0


class TestDegenerateFootage:
    def test_footage_with_no_static_texture_reports_one_shot_rather_than_inventing_cuts(
        self, tmp_path,
    ):
        """A blank wall gives nothing to match. The honest answer is one shot with the
        reason recorded, not a timeline assembled from noise."""
        path = tmp_path / "flat.mp4"
        _write(path, [numpy.full((HEIGHT, WIDTH, 3), 128, numpy.uint8)] * TOTAL_FRAMES)
        report = detect_shots(path, TOTAL_FRAMES)
        assert report["shot_count"] == 1
        assert report["reason"] == "insufficient_static_texture"

    def test_a_video_too_short_to_sample_reports_one_shot(self, tmp_path):
        path = tmp_path / "tiny.mp4"
        _write(path, [_textured_background(seed=3)] * 8)
        report = detect_shots(path, 8)
        assert report["shot_count"] == 1
        assert report["reason"] == "too_short_to_analyse"

    def test_an_unreadable_video_raises(self, tmp_path):
        missing = tmp_path / "absent.mp4"
        with pytest.raises(ShotDetectionError, match="Cannot read"):
            detect_shots(missing, 100)


class TestReportRoundTrip:
    def test_shots_rebuild_from_a_persisted_report(self, two_shot_video):
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        rebuilt = shots_from_report(report, TOTAL_FRAMES)
        assert [shot.as_dict() for shot in rebuilt] == report["shots"]

    def test_coverage_accounts_for_every_frame(self, two_shot_video):
        report = detect_shots(two_shot_video, TOTAL_FRAMES)
        assert report["frames_in_shots"] + report["frames_in_transitions"] == TOTAL_FRAMES
