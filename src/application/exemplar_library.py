"""Cuts real subjects out of the visible footage so gaps can be filled with real pixels.

The plate makes this cheap. Having recovered, per shot, what the scene looks like with
nobody in it, the foreground at any visible frame is just where that frame differs from
the plate. Inside a detection box that difference is an accurate matte — far better than
a rectangle, and obtained without a segmentation model, because the background it is
being separated from has already been measured rather than guessed.

What comes out is a bank of RGBA cut-outs per entity: the same person, photographed
repeatedly by the same camera under the same light, at a range of sizes and gait phases.
Filling a gap is then a matter of choosing one and drawing it in the right place.

**Visible-only.** Every frame read here is checked against the hidden ranges first, so
the evidence contract holds for these pixels as it does for the structured ledger. An
entity's appearance during a gap is never consulted — only its appearance before and
after, which is exactly what a reconstruction is entitled to use.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from domain.actor_placement import Observation, observation_velocities
from domain.cancellation import CancellationCheck, raise_if_cancelled


LOGGER = logging.getLogger(__name__)

# Most sightings of one entity that are kept. Enough to cover a full walk cycle at the
# stride detection runs on, without holding a whole video of crops in memory.
MAXIMUM_OBSERVATIONS_PER_ENTITY = 24
# A box smaller than this yields a cut-out too coarse to be worth drawing.
MINIMUM_OBSERVATION_HEIGHT_PIXELS = 24
# 8-bit difference from the plate above which a pixel counts as foreground. Low enough
# to catch dark clothing against dark shopfronts, high enough to ignore sensor noise and
# compression mush.
FOREGROUND_DIFFERENCE_THRESHOLD = 26
# A cut-out covering less of its box than this is usually a failed matte — the subject
# stood against something the same colour — and the box is used whole instead.
MINIMUM_MATTE_COVERAGE = 0.12
# Cleans speckle out of the matte and closes gaps inside a torso.
MATTE_OPEN_KERNEL = 3
MATTE_CLOSE_KERNEL = 7
# Softens the cut edge so it does not read as a sticker laid on the plate.
MATTE_FEATHER_SIGMA = 1.1
# Pixels of surrounding context kept around the detection box, because detections clip
# hair, heels and swinging hands.
BOX_PADDING_PIXELS = 6


@dataclass(frozen=True)
class ExemplarBank:
    """Every usable sighting of one entity, with its cut-outs."""

    entity_id: str
    observations: tuple[Observation, ...]
    cutouts: tuple[numpy.ndarray, ...]
    velocities: tuple[tuple[float, float], ...]

    def __bool__(self) -> bool:
        return bool(self.cutouts)


def _visible(frame_index: int, hidden_ranges) -> bool:
    return not any(int(start) <= frame_index <= int(end) for start, end in hidden_ranges)


def _select_observation_frames(
    detections: list[dict], hidden_ranges, shot_bounds: tuple[int, int] | None,
) -> list[Observation]:
    """The sightings worth cutting out, spread across the entity's visible life.

    Spread rather than consecutive: neighbouring detections show almost the same pose,
    so taking them all would fill the bank with near-duplicates and leave the rest of the
    walk cycle unrepresented.
    """
    usable = []
    for detection in detections:
        frame_index = int(detection["frame"])
        if not _visible(frame_index, hidden_ranges):
            continue
        if shot_bounds is not None and not shot_bounds[0] <= frame_index <= shot_bounds[1]:
            continue
        box = detection.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        left, top, right, bottom = (float(value) for value in box[:4])
        if bottom - top < MINIMUM_OBSERVATION_HEIGHT_PIXELS or right <= left:
            continue
        usable.append(Observation(frame_index, left, top, right, bottom))
    usable.sort(key=lambda item: item.source_frame)
    if len(usable) <= MAXIMUM_OBSERVATIONS_PER_ENTITY:
        return usable
    step = len(usable) / MAXIMUM_OBSERVATIONS_PER_ENTITY
    return [usable[int(index * step)] for index in range(MAXIMUM_OBSERVATIONS_PER_ENTITY)]


def matte_from_plate(
    frame_region: numpy.ndarray, plate_region: numpy.ndarray,
) -> numpy.ndarray:
    """Alpha for the subject standing in front of a known background.

    Returns full coverage rather than a sliver when the subject and the background are
    too alike to separate: drawing the whole box is a visible rectangle, but dropping
    most of a person is a hole with a head floating above it.
    """
    difference = cv2.absdiff(frame_region, plate_region)
    if difference.ndim == 3:
        difference = difference.max(axis=2)
    _, mask = cv2.threshold(
        difference, FOREGROUND_DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        numpy.ones((MATTE_OPEN_KERNEL, MATTE_OPEN_KERNEL), numpy.uint8),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        numpy.ones((MATTE_CLOSE_KERNEL, MATTE_CLOSE_KERNEL), numpy.uint8),
    )
    mask = _largest_component(mask)
    if float(numpy.count_nonzero(mask)) / mask.size < MINIMUM_MATTE_COVERAGE:
        mask = numpy.full(mask.shape, 255, numpy.uint8)
    return cv2.GaussianBlur(mask, (0, 0), MATTE_FEATHER_SIGMA)


def _largest_component(mask: numpy.ndarray) -> numpy.ndarray:
    """Keep only the biggest blob, dropping passers-by caught in the same box."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    # Label 0 is the background.
    largest = 1 + int(numpy.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return numpy.where(labels == largest, 255, 0).astype(numpy.uint8)


def _cutout(
    frame: numpy.ndarray, plate: numpy.ndarray, observation: Observation,
) -> numpy.ndarray | None:
    height, width = frame.shape[:2]
    left = max(0, int(observation.left) - BOX_PADDING_PIXELS)
    top = max(0, int(observation.top) - BOX_PADDING_PIXELS)
    right = min(width, int(observation.right) + BOX_PADDING_PIXELS)
    bottom = min(height, int(observation.bottom) + BOX_PADDING_PIXELS)
    if right - left < 2 or bottom - top < 2:
        return None
    region = frame[top:bottom, left:right]
    alpha = matte_from_plate(region, plate[top:bottom, left:right])
    cutout = numpy.dstack([region, alpha])
    return cutout


def build_exemplar_banks(
    video_path: Path,
    plate: numpy.ndarray,
    detections_by_entity: dict[str, list[dict]],
    hidden_ranges,
    shot_bounds: tuple[int, int] | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> dict[str, ExemplarBank]:
    """Cut every entity out of the visible frames it was seen in.

    Reads the video once, in order, collecting whatever is wanted from each frame:
    seeking to thousands of individual frames costs far more than a single pass.
    """
    wanted: dict[int, list[tuple[str, Observation]]] = {}
    selected: dict[str, list[Observation]] = {}
    for entity_id, detections in detections_by_entity.items():
        observations = _select_observation_frames(detections, hidden_ranges, shot_bounds)
        if not observations:
            continue
        selected[entity_id] = observations
        for observation in observations:
            wanted.setdefault(observation.source_frame, []).append((entity_id, observation))
    if not wanted:
        return {}

    collected: dict[str, dict[int, numpy.ndarray]] = {key: {} for key in selected}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {Path(video_path).name} to cut out actors")
    first_frame, last_frame = min(wanted), max(wanted)
    # Seek to the first frame of interest rather than decoding from the start: a take at
    # the end of a long video would otherwise pay for the whole file, once per take.
    # Reading is sequential from there, which is where decoding is actually fast.
    if first_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_index = first_frame
    try:
        while frame_index <= last_frame:
            raise_if_cancelled(cancellation_check)
            read_successfully, frame = capture.read()
            if not read_successfully:
                break
            for entity_id, observation in wanted.get(frame_index, ()):
                cutout = _cutout(frame, plate, observation)
                if cutout is not None:
                    collected[entity_id][frame_index] = cutout
            frame_index += 1
    finally:
        capture.release()

    banks: dict[str, ExemplarBank] = {}
    for entity_id, observations in selected.items():
        kept = [
            observation for observation in observations
            if observation.source_frame in collected[entity_id]
        ]
        if not kept:
            continue
        banks[entity_id] = ExemplarBank(
            entity_id=entity_id,
            observations=tuple(kept),
            cutouts=tuple(collected[entity_id][item.source_frame] for item in kept),
            velocities=tuple(observation_velocities(kept)),
        )
    LOGGER.info(
        "Cut out %d entities from visible footage (%d sightings)",
        len(banks), sum(len(bank.cutouts) for bank in banks.values()),
    )
    return banks
