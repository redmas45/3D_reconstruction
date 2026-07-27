"""Recovers a photographic background from the visible 75% (§5.3).

M2 composites actors onto real footage rather than rendering a synthetic environment.
The plate is that footage with people and vehicles removed: sample visible frames,
mask every tracked foreground box, and take the per-pixel median of what remains.

The median is the point. A mean smears moving actors into ghosts — the exact defect
that made the old compositing renderer unusable. A median over enough samples returns
the value the pixel held most of the time, which for a static camera is the background.

**Visible-only.** Every sampled frame is checked against the hidden ranges before it is
read, so the §2 evidence contract holds here as it does for the structured ledger.
Pixels never observed unoccluded are reported rather than quietly invented.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from domain.cancellation import CancellationCheck, raise_if_cancelled


LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLE_COUNT = 48
MINIMUM_SAMPLE_COUNT = 8
# Below this many unmasked observations a pixel's median is not trustworthy.
MINIMUM_OBSERVATIONS_PER_PIXEL = 3
# Boxes are grown before masking because detections clip limbs, and a sliver of leftover
# actor is far more visible in the plate than a slightly over-masked background.
BOX_DILATION_FRACTION = 0.08
BOX_DILATION_MINIMUM_PIXELS = 4
# Median is computed in horizontal strips so peak memory stays bounded on Colab.
STRIP_HEIGHT_PIXELS = 64
# Mean absolute deviation of unmasked samples from the plate, in 8-bit levels. Sensor
# noise and compression on a locked-off camera sit well under this; a camera that moves
# sits far above it, because each sample shows different scenery at the same pixel.
MAXIMUM_STABLE_DISAGREEMENT = 12.0
# Per-pixel spread above which that pixel's median is not background. Set well above the
# whole-plate limit so it flags only pixels that clearly never settled, rather than
# condemning ordinary sensor noise.
MAXIMUM_PIXEL_DEVIATION = 26.0
# Widening blurs used to diffuse recovered background into unresolved pixels. Narrow
# first so a pixel is filled from the nearest real background that can reach it, and only
# widened for pixels sitting deep inside a large unrecovered region.
FILL_KERNEL_SIZES = (9, 25, 61, 141, 301)
# How much genuine background a blur must have covered before its average is trusted.
FILL_MINIMUM_COVERAGE = 0.02


class CleanPlateError(RuntimeError):
    """The plate could not be built from the available visible evidence."""


@dataclass(frozen=True)
class CleanPlate:
    image: numpy.ndarray
    unresolved_mask: numpy.ndarray
    sample_count: int
    disagreement: float = 0.0

    @property
    def unresolved_fraction(self) -> float:
        return float(numpy.count_nonzero(self.unresolved_mask)) / self.unresolved_mask.size

    @property
    def is_stable(self) -> bool:
        """Whether the sampled frames actually agreed on what the background is.

        A locked-off camera produces near-identical unmasked pixels across samples, so
        the median is the background and `disagreement` stays at sensor-noise level.
        High disagreement means the samples showed different things at the same pixel,
        which has two plausible causes: the camera moved, or something moved through the
        frame that tracking never masked. Both make the plate a blend rather than a
        scene, so both are worth reporting, and the metric does not claim to tell them
        apart.

        Measured directly from the samples rather than inferred from the camera-motion
        estimator, which is a separate mechanism that can fail independently.
        """
        return self.disagreement <= MAXIMUM_STABLE_DISAGREEMENT

    def report(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "unresolved_pixel_fraction": round(self.unresolved_fraction, 6),
            "sample_disagreement": round(self.disagreement, 4),
            "stable": self.is_stable,
            "height": int(self.image.shape[0]),
            "width": int(self.image.shape[1]),
        }


def select_sample_frames(
    visible_ranges: list[tuple[int, int]],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> list[int]:
    """Spread samples across visible footage in proportion to each range's length.

    Sampling evenly over frame index would over-weight whichever visible stretch happens
    to be longest; proportional allocation keeps every part of the scene represented.
    """
    spans = [
        (int(start), int(end)) for start, end in visible_ranges if int(end) >= int(start)
    ]
    if not spans:
        raise CleanPlateError("No visible ranges are available for plate extraction")
    total_frames = sum(end - start + 1 for start, end in spans)
    frames: list[int] = []
    for start, end in spans:
        length = end - start + 1
        allocation = max(1, round(sample_count * length / total_frames))
        frames.extend(
            int(round(start + index * (length - 1) / max(1, allocation - 1)))
            for index in range(allocation)
        )
    return sorted(set(frames))


def _dilated_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = (float(value) for value in box)
    horizontal = max(BOX_DILATION_MINIMUM_PIXELS, (right - left) * BOX_DILATION_FRACTION)
    vertical = max(BOX_DILATION_MINIMUM_PIXELS, (bottom - top) * BOX_DILATION_FRACTION)
    return (
        max(0, int(left - horizontal)),
        max(0, int(top - vertical)),
        min(width, int(right + horizontal)),
        min(height, int(bottom + vertical)),
    )


def foreground_mask(
    boxes: list[tuple[float, float, float, float]], width: int, height: int,
) -> numpy.ndarray:
    """True where a tracked entity occludes the background."""
    mask = numpy.zeros((height, width), dtype=bool)
    for box in boxes:
        left, top, right, bottom = _dilated_box(box, width, height)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _read_samples(
    video_path: Path,
    sample_frames: list[int],
    foreground_boxes: dict[int, list[tuple[float, float, float, float]]],
    cancellation_check: CancellationCheck | None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise CleanPlateError(f"Cannot open video for plate extraction: {video_path.name}")
    images: list[numpy.ndarray] = []
    masks: list[numpy.ndarray] = []
    try:
        for frame_index in sample_frames:
            raise_if_cancelled(cancellation_check)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            height, width = frame.shape[:2]
            images.append(frame)
            masks.append(foreground_mask(foreground_boxes.get(frame_index, []), width, height))
    finally:
        capture.release()
    if len(images) < MINIMUM_SAMPLE_COUNT:
        raise CleanPlateError(
            f"Only {len(images)} visible frames could be sampled; "
            f"at least {MINIMUM_SAMPLE_COUNT} are required for a stable plate"
        )
    return numpy.stack(images), numpy.stack(masks)


def _masked_median(
    images: numpy.ndarray, masks: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
    """Per-pixel median over unmasked samples, computed in strips to bound memory.

    Also returns how far the unmasked samples sat from that median, both per pixel and
    overall. The overall figure says whether the plate means anything at all (see
    `CleanPlate.is_stable`); the per-pixel map says *where* it does not, which is what
    separates recovered background from a lingering crowd.
    """
    sample_count, height, width, channels = images.shape
    plate = numpy.zeros((height, width, channels), dtype=numpy.uint8)
    observations = numpy.zeros((height, width), dtype=numpy.int32)
    pixel_deviation = numpy.zeros((height, width), dtype=numpy.float32)
    deviation_total = 0.0
    deviation_count = 0
    for top in range(0, height, STRIP_HEIGHT_PIXELS):
        bottom = min(height, top + STRIP_HEIGHT_PIXELS)
        strip = images[:, top:bottom].astype(numpy.float32)
        strip_mask = masks[:, top:bottom]
        strip[strip_mask] = numpy.nan
        observations[top:bottom] = (~strip_mask).sum(axis=0)
        strip_plate = _strip_median(strip, images[:, top:bottom])
        plate[top:bottom] = strip_plate
        # NaN entries are the masked samples and drop out of both sums, so the deviation
        # measures only pixels that were genuinely observed as background.
        deviations = numpy.abs(strip - strip_plate.astype(numpy.float32))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            pixel_deviation[top:bottom] = numpy.nan_to_num(
                numpy.nanmean(deviations, axis=(0, 3)), nan=0.0,
            )
        deviation_total += float(numpy.nansum(deviations))
        deviation_count += int(numpy.count_nonzero(~numpy.isnan(deviations)))
    disagreement = deviation_total / deviation_count if deviation_count else 0.0
    return plate, observations, pixel_deviation, disagreement


def _strip_median(strip: numpy.ndarray, raw_strip: numpy.ndarray) -> numpy.ndarray:
    """Median ignoring masked samples; falls back to the plain median where all are masked."""
    with warnings.catch_warnings():
        # A fully-occluded pixel is an expected, handled outcome — it becomes an
        # unresolved-mask entry below rather than an error.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median = numpy.nanmedian(strip, axis=0)
    fully_masked = numpy.isnan(median)
    if fully_masked.any():
        # Every sample was occluded here. Use the ordinary median so the plate still has
        # a plausible value; the pixel is reported as unresolved rather than trusted.
        fallback = numpy.median(raw_strip.astype(numpy.float32), axis=0)
        median = numpy.where(fully_masked, fallback, median)
    return numpy.clip(median, 0, 255).astype(numpy.uint8)


def _unreliable_pixels(
    observations: numpy.ndarray, pixel_deviation: numpy.ndarray,
) -> numpy.ndarray:
    """Where the median is not the background, for either of the two reasons it can fail.

    A pixel can be occluded in nearly every sample, which the observation count catches.
    It can also be *unoccluded* in plenty of samples and still not settle — a crowd
    standing outside a shop is different people, not one person moving on, so more than
    half the samples show somebody and the median returns a person-coloured smear. The
    observation count says nothing about that; the spread of the samples around the
    median says it plainly.
    """
    return (observations < MINIMUM_OBSERVATIONS_PER_PIXEL) | (
        pixel_deviation > MAXIMUM_PIXEL_DEVIATION
    )


def _fill_unresolved(plate: numpy.ndarray, unresolved: numpy.ndarray) -> numpy.ndarray:
    """Replace unrecoverable pixels with a continuation of what surrounds them.

    Neither this nor the smear it replaces is evidence, and both are reported through
    `unresolved_mask`. The difference is what a viewer reads: a softened patch of
    pavement is quietly wrong, while a half-transparent stranger standing in the road
    looks like a reconstruction claiming somebody was there.

    Done by diffusing recovered background inward under progressively wider blurs, each
    weighted by how much real background it covered. Structural inpainting was tried
    first and is wrong for this: it propagates along edges, so a band of unresolved
    pixels the width of a pavement comes back covered in radiating diamond streaks that
    are far more conspicuous than the ghosts they replaced. A weighted blur has no
    preferred direction and simply fades the surroundings in.
    """
    known = ~unresolved
    if not known.any():
        return plate
    weights = known.astype(numpy.float32)
    values = plate.astype(numpy.float32) * weights[:, :, None]
    filled = plate.astype(numpy.float32)
    remaining = unresolved.copy()
    for kernel in FILL_KERNEL_SIZES:
        if not remaining.any():
            break
        size = (kernel, kernel)
        blurred = cv2.blur(values, size)
        covered = cv2.blur(weights, size)
        usable = remaining & (covered > FILL_MINIMUM_COVERAGE)
        if usable.any():
            filled[usable] = blurred[usable] / covered[usable][:, None]
            remaining[usable] = False
    if remaining.any():
        # Nothing nearby was ever recovered. The average of what was is the least
        # misleading thing left to put there.
        filled[remaining] = plate[known].mean(axis=0)
    return numpy.clip(filled, 0, 255).astype(numpy.uint8)


def extract_clean_plate(
    video_path: Path,
    visible_ranges: list[tuple[int, int]],
    foreground_boxes: dict[int, list[tuple[float, float, float, float]]],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    cancellation_check: CancellationCheck | None = None,
    sample_frames: list[int] | None = None,
) -> CleanPlate:
    """Build the background plate from visible frames only.

    `sample_frames` lets a caller that has already chosen and filtered its samples pass
    them straight through, rather than round-tripping them back through range selection.
    """
    if sample_frames is None:
        sample_frames = select_sample_frames(visible_ranges, sample_count)
    images, masks = _read_samples(video_path, sample_frames, foreground_boxes, cancellation_check)
    plate, observations, pixel_deviation, disagreement = _masked_median(images, masks)
    unresolved = _unreliable_pixels(observations, pixel_deviation)
    if unresolved.any():
        LOGGER.info(
            "Clean plate could not resolve %d pixels (%.1f%%); filling them from their "
            "surroundings",
            int(numpy.count_nonzero(unresolved)),
            100.0 * numpy.count_nonzero(unresolved) / unresolved.size,
        )
        plate = _fill_unresolved(plate, unresolved)
    extracted = CleanPlate(
        image=plate,
        unresolved_mask=unresolved,
        sample_count=len(images),
        disagreement=disagreement,
    )
    if not extracted.is_stable:
        LOGGER.warning(
            "Clean plate is unstable: sampled frames differ from it by %.1f levels on "
            "average (limit %.1f). Either the camera moves or unmasked motion crosses "
            "the frame, so the recovered background is a blend rather than one scene.",
            disagreement, MAXIMUM_STABLE_DISAGREEMENT,
        )
    return extracted


def write_clean_plate(plate: CleanPlate, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".writing.png")
    if not cv2.imwrite(str(temporary_path), plate.image):
        raise CleanPlateError(f"Could not write clean plate to {output_path}")
    temporary_path.replace(output_path)
    return output_path
