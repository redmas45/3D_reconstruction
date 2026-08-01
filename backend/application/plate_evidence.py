"""Turns tracked detections into the clean plate for a whole video (§5.3).

`clean_plate` knows how to take a masked temporal median. This module is the piece
that decides *what to mask*: it maps YOLO's detection records onto the frames the plate
samples, and it caches the result so a resumed run does not re-decode the video.

The one subtlety worth stating plainly. Detection runs on a stride — every eighth frame
by default — so most frames have no record of their own. Masking only exactly-detected
frames would leave actors unmasked on every other sample, and they would survive into
the median as ghosts. So each sample takes the union of every box seen within a window
around it. Over-masking is cheap here: with dozens of samples the median only needs a
majority of clean observations per pixel, and an over-wide mask costs nothing but a few
more unresolved pixels. Under-masking costs a ghost, which is visible in the output.

**Visible-only.** Sample frames come from the selection's visible ranges and are
re-checked against the hidden ranges before any frame is read, so the §2 evidence
contract holds for pixels as it does for structured evidence.
"""

import logging
from pathlib import Path

import cv2
import numpy

from application.clean_plate import (
    DEFAULT_SAMPLE_COUNT,
    CleanPlate,
    CleanPlateError,
    extract_clean_plate,
    select_sample_frames,
    write_clean_plate,
)
from domain.cancellation import CancellationCheck
from infrastructure.json_files import read_json_file, write_json_file


LOGGER = logging.getLogger(__name__)

PLATE_IMAGE_NAME = "clean_plate.png"
PLATE_REPORT_NAME = "clean_plate_report.json"
PLATE_MASK_NAME = "clean_plate_unresolved.png"

# Half-width of the window, as a multiple of the detection stride, over which boxes are
# unioned onto a sample frame. Two strides covers the sample's own lattice slot and its
# neighbours, so an actor moving between detections is still masked along its path.
BOX_WINDOW_STRIDE_MULTIPLIER = 2
MINIMUM_BOX_WINDOW_FRAMES = 4


class PlateEvidenceError(RuntimeError):
    """The plate could not be produced from this video's visible evidence."""


def _visible_only(frames: list[int], hidden_ranges: list[tuple[int, int]]) -> list[int]:
    hidden = [(int(start), int(end)) for start, end in hidden_ranges]
    return [
        frame for frame in frames
        if not any(start <= frame <= end for start, end in hidden)
    ]


def detection_boxes_by_frame(
    detections: list[dict],
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Group raw detection records by the frame they were measured on."""
    grouped: dict[int, list[tuple[float, float, float, float]]] = {}
    for record in detections:
        box = record.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        frame = int(record["frame"])
        grouped.setdefault(frame, []).append(tuple(float(value) for value in box[:4]))
    return grouped


def foreground_boxes_for_samples(
    detections: list[dict],
    sample_frames: list[int],
    window_frames: int,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Union every box observed near each sample frame onto that sample.

    Linear in the number of detections rather than quadratic in samples: each record is
    assigned to the samples whose window contains it.
    """
    grouped = detection_boxes_by_frame(detections)
    ordered_samples = sorted(sample_frames)
    boxes: dict[int, list[tuple[float, float, float, float]]] = {
        frame: [] for frame in ordered_samples
    }
    if not ordered_samples:
        return boxes
    sample_array = numpy.array(ordered_samples)
    for detection_frame, frame_boxes in grouped.items():
        first = int(numpy.searchsorted(sample_array, detection_frame - window_frames, "left"))
        last = int(numpy.searchsorted(sample_array, detection_frame + window_frames, "right"))
        for sample in ordered_samples[first:last]:
            boxes[sample].extend(frame_boxes)
    return boxes


def box_window_frames(detection_stride: int) -> int:
    return max(
        MINIMUM_BOX_WINDOW_FRAMES,
        int(detection_stride) * BOX_WINDOW_STRIDE_MULTIPLIER,
    )


def build_clean_plate(
    video_path: Path,
    visible_ranges: list[tuple[int, int]],
    hidden_ranges: list[tuple[int, int]],
    detections: list[dict],
    detection_stride: int,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    cancellation_check: CancellationCheck | None = None,
) -> CleanPlate:
    """Extract the background plate for one video."""
    sample_frames = _visible_only(
        select_sample_frames(visible_ranges, sample_count), hidden_ranges,
    )
    if not sample_frames:
        raise PlateEvidenceError(
            "No visible frames remain for plate extraction after excluding hidden ranges"
        )
    window = box_window_frames(detection_stride)
    boxes = foreground_boxes_for_samples(detections, sample_frames, window)
    try:
        return extract_clean_plate(
            video_path,
            visible_ranges,
            boxes,
            cancellation_check=cancellation_check,
            sample_frames=sample_frames,
        )
    except CleanPlateError as error:
        raise PlateEvidenceError(str(error)) from error


def load_cached_plate(plate_directory: Path, contract: dict) -> CleanPlate | None:
    """Return a previously extracted plate when it was built for this exact input."""
    report = read_json_file(plate_directory / PLATE_REPORT_NAME)
    if not isinstance(report, dict) or report.get("contract") != contract:
        return None
    image = cv2.imread(str(plate_directory / PLATE_IMAGE_NAME), cv2.IMREAD_COLOR)
    mask_image = cv2.imread(str(plate_directory / PLATE_MASK_NAME), cv2.IMREAD_GRAYSCALE)
    if image is None or mask_image is None or image.shape[:2] != mask_image.shape[:2]:
        return None
    return CleanPlate(
        image=image,
        unresolved_mask=mask_image > 0,
        sample_count=int(report.get("sample_count", 0)),
        disagreement=float(report.get("sample_disagreement", 0.0)),
    )


def store_plate(plate: CleanPlate, plate_directory: Path, contract: dict) -> Path:
    """Persist the plate and the contract it was built for."""
    image_path = write_clean_plate(plate, plate_directory / PLATE_IMAGE_NAME)
    mask_path = plate_directory / PLATE_MASK_NAME
    temporary_mask = mask_path.with_suffix(".writing.png")
    cv2.imwrite(str(temporary_mask), plate.unresolved_mask.astype(numpy.uint8) * 255)
    temporary_mask.replace(mask_path)
    # The report is written last: it is what `load_cached_plate` trusts, so it must never
    # name images that are not fully on disk yet.
    write_json_file(
        plate_directory / PLATE_REPORT_NAME, {**plate.report(), "contract": contract},
    )
    return image_path


def plate_cache_contract(
    video_sha256: str,
    visible_ranges: list[tuple[int, int]],
    detection_stride: int,
    sample_count: int,
) -> dict:
    return {
        "schema_version": 1,
        "video_sha256": video_sha256,
        "visible_ranges": [[int(start), int(end)] for start, end in visible_ranges],
        "detection_stride": int(detection_stride),
        "sample_count": int(sample_count),
        "box_window_frames": box_window_frames(detection_stride),
    }


def resolve_clean_plate(
    video_path: Path,
    plate_directory: Path,
    visible_ranges: list[tuple[int, int]],
    hidden_ranges: list[tuple[int, int]],
    detections: list[dict],
    detection_stride: int,
    video_sha256: str,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    reuse_work: bool = False,
    cancellation_check: CancellationCheck | None = None,
) -> CleanPlate:
    """Extract the plate, or reuse a cached one built from identical inputs."""
    contract = plate_cache_contract(
        video_sha256, visible_ranges, detection_stride, sample_count,
    )
    if reuse_work:
        cached = load_cached_plate(plate_directory, contract)
        if cached is not None:
            LOGGER.info("Reusing cached clean plate from %s", plate_directory)
            return cached
    plate = build_clean_plate(
        video_path,
        visible_ranges,
        hidden_ranges,
        detections,
        detection_stride,
        sample_count,
        cancellation_check,
    )
    store_plate(plate, plate_directory, contract)
    return plate
