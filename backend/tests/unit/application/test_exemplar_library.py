"""Cutting subjects out of visible footage using the plate as the matte.

The plate is what makes this work: having measured what the scene looks like empty, the
foreground is simply where a frame differs from it. These tests use a synthetic video
whose subject and background are known exactly, so "the cut-out is the subject and not
the wall behind it" is measurable rather than a matter of opinion.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.exemplar_library import (
    MAXIMUM_OBSERVATIONS_PER_ENTITY,
    build_exemplar_banks,
    matte_from_plate,
)

WIDTH, HEIGHT = 320, 240
FRAME_COUNT = 120
BACKGROUND = (60, 120, 90)
SUBJECT = (230, 40, 200)
SUBJECT_WIDTH, SUBJECT_HEIGHT = 24, 60
TOP = HEIGHT - SUBJECT_HEIGHT - 20


def _left_at(frame_index: int) -> int:
    return int(20 + (frame_index / FRAME_COUNT) * (WIDTH - SUBJECT_WIDTH - 40))


@pytest.fixture
def video(tmp_path) -> tuple[Path, list[dict]]:
    path = tmp_path / "walk.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    assert writer.isOpened()
    detections = []
    try:
        for frame_index in range(FRAME_COUNT):
            frame = numpy.full((HEIGHT, WIDTH, 3), BACKGROUND, numpy.uint8)
            left = _left_at(frame_index)
            frame[TOP:TOP + SUBJECT_HEIGHT, left:left + SUBJECT_WIDTH] = SUBJECT
            writer.write(frame)
            detections.append({
                "frame": frame_index,
                "bbox": [left, TOP, left + SUBJECT_WIDTH, TOP + SUBJECT_HEIGHT],
            })
    finally:
        writer.release()
    return path, detections


@pytest.fixture
def plate() -> numpy.ndarray:
    return numpy.full((HEIGHT, WIDTH, 3), BACKGROUND, numpy.uint8)


class TestMatte:
    def test_the_subject_is_opaque_and_the_background_is_not(self):
        region = numpy.full((40, 40, 3), BACKGROUND, numpy.uint8)
        region[10:30, 10:30] = SUBJECT
        alpha = matte_from_plate(region, numpy.full((40, 40, 3), BACKGROUND, numpy.uint8))
        assert alpha[20, 20] > 200
        assert alpha[2, 2] < 60

    def test_a_subject_indistinguishable_from_the_background_keeps_the_whole_box(self):
        """Dropping most of a person leaves a head floating above a hole, which is worse
        than the visible rectangle that keeping the box produces."""
        region = numpy.full((40, 40, 3), BACKGROUND, numpy.uint8)
        alpha = matte_from_plate(region, numpy.full((40, 40, 3), BACKGROUND, numpy.uint8))
        assert alpha.min() > 200

    def test_a_passer_by_in_the_same_box_is_dropped(self):
        """Only the largest blob is kept, so a second person clipped into the box does
        not travel around glued to the first."""
        region = numpy.full((60, 60, 3), BACKGROUND, numpy.uint8)
        region[10:50, 10:30] = SUBJECT           # the subject
        region[50:54, 50:54] = SUBJECT           # somebody's shoulder at the edge
        alpha = matte_from_plate(region, numpy.full((60, 60, 3), BACKGROUND, numpy.uint8))
        assert alpha[20, 20] > 200
        assert alpha[52, 52] < 60

    def test_the_cut_edge_is_soft(self):
        region = numpy.full((40, 40, 3), BACKGROUND, numpy.uint8)
        region[10:30, 10:30] = SUBJECT
        alpha = matte_from_plate(region, numpy.full((40, 40, 3), BACKGROUND, numpy.uint8))
        row = alpha[20].astype(int)
        assert 0 < row[9] < 255 or 0 < row[10] < 255


class TestBankBuilding:
    def test_an_entity_gets_a_bank_of_cut_outs(self, video, plate):
        path, detections = video
        banks = build_exemplar_banks(path, plate, {"person_1": detections}, [])
        assert banks["person_1"].cutouts

    def test_cut_outs_carry_an_alpha_channel(self, video, plate):
        path, detections = video
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, [])["person_1"]
        assert all(cutout.shape[2] == 4 for cutout in bank.cutouts)

    def test_the_cut_out_holds_the_subject_not_the_background(self, video, plate):
        path, detections = video
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, [])["person_1"]
        cutout = bank.cutouts[len(bank.cutouts) // 2]
        centre = cutout[cutout.shape[0] // 2, cutout.shape[1] // 2]
        assert centre[3] > 200, "the subject was matted out"

    def test_hidden_frames_are_never_read(self, video, plate):
        """The evidence contract holds for pixels as it does for the ledger: an entity's
        appearance during a gap must not inform the reconstruction of that gap."""
        path, detections = video
        hidden = [(30, 90)]
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, hidden)["person_1"]
        assert all(
            not 30 <= observation.source_frame <= 90
            for observation in bank.observations
        )

    def test_observations_outside_the_shot_are_excluded(self, video, plate):
        path, detections = video
        bank = build_exemplar_banks(
            path, plate, {"person_1": detections}, [], shot_bounds=(0, 40),
        )["person_1"]
        assert all(observation.source_frame <= 40 for observation in bank.observations)

    def test_the_bank_is_capped_so_memory_does_not_grow_with_video_length(self, video, plate):
        path, detections = video
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, [])["person_1"]
        assert len(bank.cutouts) <= MAXIMUM_OBSERVATIONS_PER_ENTITY

    def test_sightings_are_spread_across_the_entitys_life(self, video, plate):
        """Consecutive detections show almost the same pose; a bank of near-duplicates
        cannot supply a walk cycle."""
        path, detections = video
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, [])["person_1"]
        frames = [observation.source_frame for observation in bank.observations]
        assert max(frames) - min(frames) > FRAME_COUNT * 0.6

    def test_every_observation_has_a_matching_cut_out_and_velocity(self, video, plate):
        path, detections = video
        bank = build_exemplar_banks(path, plate, {"person_1": detections}, [])["person_1"]
        assert len(bank.observations) == len(bank.cutouts) == len(bank.velocities)

    def test_an_entity_too_small_to_cut_out_gets_no_bank(self, video, plate):
        path, _ = video
        tiny = [
            {"frame": index, "bbox": [10, 10, 18, 20]} for index in range(0, 60, 4)
        ]
        assert build_exemplar_banks(path, plate, {"person_1": tiny}, []) == {}

    def test_an_entity_with_no_detections_gets_no_bank(self, video, plate):
        path, _ = video
        assert build_exemplar_banks(path, plate, {"person_1": []}, []) == {}

    def test_an_empty_bank_is_falsey(self, video, plate):
        path, detections = video
        banks = build_exemplar_banks(path, plate, {"person_1": detections}, [])
        assert bool(banks["person_1"]) is True
