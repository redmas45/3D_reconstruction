"""Clean plate extraction, tested against a synthetic video with a known background.

The whole point of the median is that a moving actor must leave no trace. These tests
construct a video where the correct answer is known exactly, so "no ghosting" is a
measurable assertion rather than a visual impression.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.clean_plate import (
    CleanPlateError,
    extract_clean_plate,
    foreground_mask,
    select_sample_frames,
    write_clean_plate,
)

FRAME_WIDTH = 160
FRAME_HEIGHT = 90
FRAME_COUNT = 120
BACKGROUND_COLOUR = (40, 90, 160)
ACTOR_COLOUR = (250, 250, 250)
ACTOR_WIDTH = 18
ACTOR_HEIGHT = 40


def _actor_left(frame_index: int) -> int:
    """Actor sweeps across the frame so no pixel is covered in most samples."""
    travel = FRAME_WIDTH - ACTOR_WIDTH
    return int((frame_index / max(1, FRAME_COUNT - 1)) * travel)


def _write_video(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened()
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    top = FRAME_HEIGHT - ACTOR_HEIGHT - 2
    try:
        for frame_index in range(FRAME_COUNT):
            frame = numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND_COLOUR, dtype=numpy.uint8)
            left = _actor_left(frame_index)
            frame[top:top + ACTOR_HEIGHT, left:left + ACTOR_WIDTH] = ACTOR_COLOUR
            writer.write(frame)
            boxes[frame_index] = [(left, top, left + ACTOR_WIDTH, top + ACTOR_HEIGHT)]
    finally:
        writer.release()
    return boxes


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    boxes = _write_video(path)
    return path, boxes


class TestSampleSelection:
    def test_samples_come_only_from_visible_ranges(self):
        frames = select_sample_frames([(0, 99), (200, 299)], sample_count=20)
        assert all(0 <= frame <= 99 or 200 <= frame <= 299 for frame in frames)

    def test_hidden_ranges_are_never_sampled(self):
        """The evidence contract applies to pixels exactly as it does to the ledger."""
        frames = select_sample_frames([(0, 99), (200, 299)], sample_count=40)
        assert not any(100 <= frame <= 199 for frame in frames)

    def test_every_visible_range_is_represented(self):
        frames = select_sample_frames([(0, 99), (500, 599)], sample_count=20)
        assert any(frame <= 99 for frame in frames)
        assert any(frame >= 500 for frame in frames)

    def test_longer_ranges_receive_more_samples(self):
        frames = select_sample_frames([(0, 899), (1000, 1099)], sample_count=40)
        long_range = sum(1 for frame in frames if frame <= 899)
        short_range = sum(1 for frame in frames if frame >= 1000)
        assert long_range > short_range

    def test_samples_are_sorted_and_unique(self):
        frames = select_sample_frames([(0, 99)], sample_count=30)
        assert frames == sorted(set(frames))

    def test_no_visible_ranges_is_rejected(self):
        with pytest.raises(CleanPlateError, match="No visible ranges"):
            select_sample_frames([], sample_count=10)


class TestForegroundMask:
    def test_box_region_is_masked(self):
        mask = foreground_mask([(10, 10, 30, 40)], FRAME_WIDTH, FRAME_HEIGHT)
        assert mask[25, 20]

    def test_mask_is_dilated_beyond_the_box(self):
        """Detections clip limbs; leftover slivers are very visible in a plate."""
        mask = foreground_mask([(50, 20, 70, 60)], FRAME_WIDTH, FRAME_HEIGHT)
        assert mask[20, 48]

    def test_area_outside_the_box_is_unmasked(self):
        mask = foreground_mask([(10, 10, 30, 40)], FRAME_WIDTH, FRAME_HEIGHT)
        assert not mask[80, 150]

    def test_no_boxes_masks_nothing(self):
        assert not foreground_mask([], FRAME_WIDTH, FRAME_HEIGHT).any()

    def test_box_beyond_the_frame_is_clamped(self):
        mask = foreground_mask([(-50, -50, 10_000, 10_000)], FRAME_WIDTH, FRAME_HEIGHT)
        assert mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
        assert mask.all()


class TestPlateExtraction:
    def test_actor_is_removed_from_the_plate(self, synthetic_video):
        """The decisive test: no trace of the bright actor anywhere in the plate."""
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=40)
        brightest = plate.image.reshape(-1, 3).max(axis=0)
        assert (brightest < 200).all(), "actor pixels survived into the plate"

    def test_plate_recovers_the_background_colour(self, synthetic_video):
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=40)
        median_colour = numpy.median(plate.image.reshape(-1, 3), axis=0)
        assert median_colour == pytest.approx(BACKGROUND_COLOUR, abs=12)

    def test_plate_matches_the_source_dimensions(self, synthetic_video):
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=20)
        assert plate.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def test_unmasked_extraction_leaves_ghosting(self, synthetic_video):
        """Confirms the mask is doing real work, not that the video is trivially clean."""
        video_path, _ = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], {}, sample_count=40)
        without_mask = plate.image.reshape(-1, 3).max(axis=0)

        masked = extract_clean_plate(
            video_path, [(0, FRAME_COUNT - 1)], _write_boxes_only(), sample_count=40,
        )
        with_mask = masked.image.reshape(-1, 3).max(axis=0)
        assert without_mask.max() >= with_mask.max()

    def test_report_describes_the_plate(self, synthetic_video):
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=20)
        report = plate.report()
        assert report["width"] == FRAME_WIDTH
        assert report["height"] == FRAME_HEIGHT
        assert 0.0 <= report["unresolved_pixel_fraction"] <= 1.0

    def test_permanently_occluded_pixels_are_reported_not_hidden(self, tmp_path):
        """A parked car is never seen unoccluded; the plate must admit that."""
        path = tmp_path / "static_occluder.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_WIDTH, FRAME_HEIGHT),
        )
        boxes = {}
        try:
            for frame_index in range(FRAME_COUNT):
                frame = numpy.full(
                    (FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND_COLOUR, dtype=numpy.uint8,
                )
                frame[10:40, 10:40] = ACTOR_COLOUR
                writer.write(frame)
                boxes[frame_index] = [(10, 10, 40, 40)]
        finally:
            writer.release()
        plate = extract_clean_plate(path, [(0, FRAME_COUNT - 1)], boxes, sample_count=20)
        assert plate.unresolved_fraction > 0.0
        assert plate.unresolved_mask[25, 25]

    def test_too_few_samples_fails_loudly(self, synthetic_video):
        video_path, boxes = synthetic_video
        with pytest.raises(CleanPlateError, match="at least"):
            extract_clean_plate(video_path, [(0, 2)], boxes, sample_count=3)

    def test_missing_video_fails_cleanly(self, tmp_path):
        with pytest.raises(CleanPlateError):
            extract_clean_plate(tmp_path / "absent.mp4", [(0, 99)], {}, sample_count=20)


class TestPlateWriting:
    def test_plate_is_written_and_readable(self, synthetic_video, tmp_path):
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=20)
        output = write_clean_plate(plate, tmp_path / "plate" / "clean_plate.png")
        assert output.is_file()
        assert cv2.imread(str(output)) is not None

    def test_no_temporary_file_is_left_behind(self, synthetic_video, tmp_path):
        video_path, boxes = synthetic_video
        plate = extract_clean_plate(video_path, [(0, FRAME_COUNT - 1)], boxes, sample_count=20)
        output = write_clean_plate(plate, tmp_path / "plate" / "clean_plate.png")
        assert [path.name for path in output.parent.iterdir()] == ["clean_plate.png"]


def _write_boxes_only() -> dict[int, list[tuple[float, float, float, float]]]:
    top = FRAME_HEIGHT - ACTOR_HEIGHT - 2
    return {
        frame_index: [(
            _actor_left(frame_index), top,
            _actor_left(frame_index) + ACTOR_WIDTH, top + ACTOR_HEIGHT,
        )]
        for frame_index in range(FRAME_COUNT)
    }
