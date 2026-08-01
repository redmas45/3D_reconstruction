import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.clean_plate import CleanPlate
from application.plate_evidence import (
    PLATE_IMAGE_NAME,
    PLATE_MASK_NAME,
    PLATE_REPORT_NAME,
    PlateEvidenceError,
    box_window_frames,
    build_clean_plate,
    detection_boxes_by_frame,
    foreground_boxes_for_samples,
    load_cached_plate,
    plate_cache_contract,
    resolve_clean_plate,
    store_plate,
)

FRAME_WIDTH = 160
FRAME_HEIGHT = 90
BACKGROUND_VALUE = 70
ACTOR_VALUE = 245


def _detection(frame, box, track=1):
    return {
        "frame": frame,
        "bbox": list(box),
        "source_track_id": track,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
    }


def _write_video(path: Path, frame_count: int, actor_width: int = 24) -> list[list[int]]:
    """A static background with one bright actor sweeping left to right.

    Returns the actor's box on each frame, so a test can mask exactly what it drew.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened()
    boxes = []
    try:
        for index in range(frame_count):
            frame = numpy.full(
                (FRAME_HEIGHT, FRAME_WIDTH, 3), BACKGROUND_VALUE, dtype=numpy.uint8,
            )
            left = int(index * (FRAME_WIDTH - actor_width) / max(1, frame_count - 1))
            frame[20:70, left:left + actor_width] = ACTOR_VALUE
            writer.write(frame)
            boxes.append([left, 20, left + actor_width, 70])
    finally:
        writer.release()
    return boxes


class TestDetectionGrouping:
    def test_records_are_grouped_by_frame(self):
        grouped = detection_boxes_by_frame([
            _detection(0, [1, 2, 3, 4]), _detection(0, [5, 6, 7, 8]), _detection(8, [1, 1, 2, 2]),
        ])
        assert len(grouped[0]) == 2
        assert len(grouped[8]) == 1

    def test_malformed_records_are_skipped_rather_than_crashing(self):
        grouped = detection_boxes_by_frame([
            {"frame": 0, "bbox": None}, {"frame": 0, "bbox": [1, 2]}, _detection(0, [1, 2, 3, 4]),
        ])
        assert len(grouped[0]) == 1

    def test_no_detections_gives_no_groups(self):
        assert detection_boxes_by_frame([]) == {}


class TestSampleWindowing:
    def test_boxes_from_nearby_frames_reach_a_sample(self):
        """The whole point: detection runs on a stride, so samples borrow neighbours."""
        boxes = foreground_boxes_for_samples([_detection(8, [0, 0, 10, 10])], [10], 4)
        assert boxes[10] == [(0.0, 0.0, 10.0, 10.0)]

    def test_boxes_beyond_the_window_do_not_reach_a_sample(self):
        boxes = foreground_boxes_for_samples([_detection(100, [0, 0, 10, 10])], [10], 4)
        assert boxes[10] == []

    def test_a_box_can_serve_several_samples(self):
        boxes = foreground_boxes_for_samples([_detection(10, [0, 0, 10, 10])], [8, 10, 12], 4)
        assert all(len(boxes[sample]) == 1 for sample in (8, 10, 12))

    def test_every_sample_is_present_even_with_nothing_to_mask(self):
        boxes = foreground_boxes_for_samples([], [1, 2, 3], 4)
        assert sorted(boxes) == [1, 2, 3]

    def test_window_edges_are_inclusive(self):
        assert foreground_boxes_for_samples([_detection(14, [0, 0, 1, 1])], [10], 4)[10]
        assert foreground_boxes_for_samples([_detection(6, [0, 0, 1, 1])], [10], 4)[10]

    def test_window_scales_with_the_detection_stride(self):
        assert box_window_frames(8) == 16
        assert box_window_frames(1) == 4  # never narrower than the floor


class TestPlateExtraction:
    def test_the_actor_does_not_survive_into_the_plate(self, tmp_path):
        """The assertion that matters: no ghost of a swept actor in the background."""
        video = tmp_path / "input.mp4"
        boxes = _write_video(video, 60)
        detections = [_detection(index, box) for index, box in enumerate(boxes)]
        plate = build_clean_plate(
            video, [(0, 59)], [], detections, detection_stride=1, sample_count=40,
        )
        assert plate.image.max() < 200

    def test_the_background_itself_is_recovered(self, tmp_path):
        video = tmp_path / "input.mp4"
        boxes = _write_video(video, 60)
        detections = [_detection(index, box) for index, box in enumerate(boxes)]
        plate = build_clean_plate(
            video, [(0, 59)], [], detections, detection_stride=1, sample_count=40,
        )
        assert abs(float(plate.image.mean()) - BACKGROUND_VALUE) < 12

    def test_strided_detections_still_mask_the_actor(self, tmp_path):
        """Only every eighth frame is detected, as the real pipeline does."""
        video = tmp_path / "input.mp4"
        boxes = _write_video(video, 96)
        detections = [
            _detection(index, box)
            for index, box in enumerate(boxes) if index % 8 == 0
        ]
        plate = build_clean_plate(
            video, [(0, 95)], [], detections, detection_stride=8, sample_count=40,
        )
        assert plate.image.max() < 200

    def test_hidden_frames_are_never_sampled(self, tmp_path):
        video = tmp_path / "input.mp4"
        _write_video(video, 60)
        with pytest.raises(PlateEvidenceError, match="No visible frames remain"):
            build_clean_plate(video, [(0, 59)], [(0, 59)], [], 8, sample_count=40)

    def test_too_few_readable_samples_is_reported(self, tmp_path):
        video = tmp_path / "input.mp4"
        _write_video(video, 60)
        with pytest.raises(PlateEvidenceError):
            build_clean_plate(video, [(0, 3)], [], [], 8, sample_count=4)


class TestPlateCache:
    @staticmethod
    def _plate():
        return CleanPlate(
            image=numpy.full((8, 12, 3), 40, dtype=numpy.uint8),
            unresolved_mask=numpy.zeros((8, 12), dtype=bool),
            sample_count=17,
        )

    def test_a_stored_plate_round_trips(self, tmp_path):
        contract = plate_cache_contract("abc", [(0, 10)], 8, 48)
        store_plate(self._plate(), tmp_path, contract)
        restored = load_cached_plate(tmp_path, contract)
        assert restored is not None
        assert restored.sample_count == 17
        assert (restored.image == 40).all()

    def test_unresolved_pixels_survive_the_round_trip(self, tmp_path):
        plate = self._plate()
        plate.unresolved_mask[2:4, 3:5] = True
        contract = plate_cache_contract("abc", [(0, 10)], 8, 48)
        store_plate(plate, tmp_path, contract)
        restored = load_cached_plate(tmp_path, contract)
        assert restored.unresolved_mask.sum() == 4

    def test_a_different_contract_is_not_reused(self, tmp_path):
        store_plate(self._plate(), tmp_path, plate_cache_contract("abc", [(0, 10)], 8, 48))
        other = plate_cache_contract("different", [(0, 10)], 8, 48)
        assert load_cached_plate(tmp_path, other) is None

    def test_changing_the_visible_ranges_invalidates_the_cache(self, tmp_path):
        store_plate(self._plate(), tmp_path, plate_cache_contract("abc", [(0, 10)], 8, 48))
        assert load_cached_plate(tmp_path, plate_cache_contract("abc", [(0, 20)], 8, 48)) is None

    def test_an_empty_directory_has_nothing_to_reuse(self, tmp_path):
        assert load_cached_plate(tmp_path, plate_cache_contract("abc", [(0, 10)], 8, 48)) is None

    def test_a_missing_image_is_not_reused_despite_a_matching_report(self, tmp_path):
        contract = plate_cache_contract("abc", [(0, 10)], 8, 48)
        store_plate(self._plate(), tmp_path, contract)
        (tmp_path / PLATE_IMAGE_NAME).unlink()
        assert load_cached_plate(tmp_path, contract) is None

    def test_every_cache_artifact_is_written(self, tmp_path):
        store_plate(self._plate(), tmp_path, plate_cache_contract("abc", [(0, 10)], 8, 48))
        written = {path.name for path in tmp_path.iterdir()}
        assert {PLATE_IMAGE_NAME, PLATE_MASK_NAME, PLATE_REPORT_NAME} <= written

    def test_no_partial_files_are_left_behind(self, tmp_path):
        store_plate(self._plate(), tmp_path, plate_cache_contract("abc", [(0, 10)], 8, 48))
        assert not any(".writing" in path.name for path in tmp_path.iterdir())


class TestResolveCleanPlate:
    def _extract(self, tmp_path, reuse_work):
        return resolve_clean_plate(
            video_path=tmp_path / "input.mp4",
            plate_directory=tmp_path / "plate",
            visible_ranges=[(0, 59)],
            hidden_ranges=[],
            detections=[],
            detection_stride=8,
            video_sha256="deadbeef",
            sample_count=20,
            reuse_work=reuse_work,
        )

    def test_a_plate_is_extracted_and_persisted(self, tmp_path):
        _write_video(tmp_path / "input.mp4", 60)
        plate = self._extract(tmp_path, reuse_work=False)
        assert plate.image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        assert (tmp_path / "plate" / PLATE_REPORT_NAME).is_file()

    def test_a_second_call_reuses_the_cached_plate(self, tmp_path):
        _write_video(tmp_path / "input.mp4", 60)
        first = self._extract(tmp_path, reuse_work=False)
        (tmp_path / "input.mp4").unlink()  # only a cache hit can succeed now
        second = self._extract(tmp_path, reuse_work=True)
        assert (first.image == second.image).all()

    def test_reuse_is_not_attempted_when_it_was_not_asked_for(self, tmp_path):
        _write_video(tmp_path / "input.mp4", 60)
        self._extract(tmp_path, reuse_work=False)
        (tmp_path / "input.mp4").unlink()
        with pytest.raises(PlateEvidenceError):
            self._extract(tmp_path, reuse_work=False)


def _write_panning_video(path: Path, frame_count: int) -> None:
    """A camera that pans across a textured wall: no actor, but every sample differs."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened()
    columns = numpy.arange(FRAME_WIDTH * 3)
    try:
        for index in range(frame_count):
            stripes = (((columns + index * 6) // 7) % 2) * 210 + 20
            frame = numpy.repeat(
                stripes[index * 2:index * 2 + FRAME_WIDTH][None, :], FRAME_HEIGHT, axis=0,
            )
            writer.write(numpy.repeat(frame[..., None], 3, axis=2).astype(numpy.uint8))
    finally:
        writer.release()


class TestPlateStability:
    def test_a_static_camera_produces_a_stable_plate(self, tmp_path):
        video = tmp_path / "input.mp4"
        boxes = _write_video(video, 60)
        detections = [_detection(index, box) for index, box in enumerate(boxes)]
        plate = build_clean_plate(
            video, [(0, 59)], [], detections, detection_stride=1, sample_count=40,
        )
        assert plate.is_stable
        assert plate.disagreement < 12.0

    def test_a_moving_camera_is_detected_as_unstable(self, tmp_path):
        """The plate would be a blend of viewpoints, and the run must say so."""
        video = tmp_path / "panning.mp4"
        _write_panning_video(video, 60)
        plate = build_clean_plate(video, [(0, 59)], [], [], 1, sample_count=40)
        assert not plate.is_stable

    def test_stability_is_reported_alongside_the_measurement(self, tmp_path):
        video = tmp_path / "input.mp4"
        boxes = _write_video(video, 60)
        detections = [_detection(index, box) for index, box in enumerate(boxes)]
        report = build_clean_plate(
            video, [(0, 59)], [], detections, 1, sample_count=40,
        ).report()
        assert report["stable"] is True
        assert "sample_disagreement" in report

    def test_unmasked_motion_also_registers_as_instability(self, tmp_path):
        """The metric measures sample disagreement; it does not claim to name the cause.

        Here the camera is static but the actor was never detected, so the plate is a
        blend for a different reason. Reporting it is still the right outcome.
        """
        video = tmp_path / "input.mp4"
        _write_video(video, 60)
        plate = build_clean_plate(video, [(0, 59)], [], [], 1, sample_count=40)
        assert not plate.is_stable

    def test_stability_survives_the_cache_round_trip(self, tmp_path):
        unstable = CleanPlate(
            image=numpy.zeros((8, 12, 3), dtype=numpy.uint8),
            unresolved_mask=numpy.zeros((8, 12), dtype=bool),
            sample_count=20,
            disagreement=41.5,
        )
        contract = plate_cache_contract("abc", [(0, 10)], 8, 48)
        store_plate(unstable, tmp_path, contract)
        restored = load_cached_plate(tmp_path, contract)
        assert restored.disagreement == pytest.approx(41.5)
        assert not restored.is_stable
