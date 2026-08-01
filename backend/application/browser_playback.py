"""Builds the browser playback video without leaking hidden source frames.

The browser renderer needs one ordinary, frame-complete video under its transparent
Three.js layer. Visible frames are copied from the uploaded source. Frames selected as
gaps are replaced with a background plate recovered from visible evidence only; the
Three.js client then draws the inferred actors over those frames while the video plays.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy

from domain.cancellation import CancellationCheck, raise_if_cancelled


PlateProvider = Callable[[int], numpy.ndarray | None]
BOUNDARY_WIPE_FEATHER = 0.08


class BrowserPlaybackError(RuntimeError):
    """The evidence-safe playback base could not be written."""


def write_browser_playback_video(
    source_path: Path,
    output_path: Path,
    hidden_ranges: Sequence[tuple[int, int]],
    plates: PlateProvider | None,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    """Write a complete timeline with source frames and evidence-safe gap bases.

    The writer deliberately never reads a hidden source frame into the output. It only
    reads the source sequentially to advance the decoder, replacing hidden frames with
    a cached clean plate or a transition between the two visible boundary frames.
    """
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise BrowserPlaybackError(f"Cannot open source video: {source_path.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1 or fps <= 0.0 or expected_frames < 1:
        capture.release()
        raise BrowserPlaybackError("Source video has an invalid frame contract")
    ranges = _normalize_ranges(hidden_ranges, expected_frames)
    boundary_frames = _read_visible_boundary_frames(
        source_path, ranges, expected_frames, cancellation_check,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.writing{output_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise BrowserPlaybackError(f"Cannot create playback video: {output_path.name}")

    frame_index = 0
    last_visible_frame: numpy.ndarray | None = None
    try:
        while True:
            raise_if_cancelled(cancellation_check)
            success, source_frame = capture.read()
            if not success:
                break
            gap_index = _gap_index_for_frame(ranges, frame_index)
            if gap_index is not None:
                gap_start, gap_end = ranges[gap_index]
                frame = _plate_frame(
                    plates,
                    gap_index,
                    frame_index,
                    gap_start,
                    gap_end,
                    last_visible_frame,
                    boundary_frames.get(gap_index),
                    source_frame,
                )
            else:
                frame = source_frame
                last_visible_frame = source_frame.copy()
            writer.write(_fit_frame(frame, width, height))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index != expected_frames:
        temporary_path.unlink(missing_ok=True)
        raise BrowserPlaybackError(
            f"Playback decoded {frame_index} of {expected_frames} source frames"
        )
    temporary_path.replace(output_path)
    return output_path


def _normalize_ranges(
    hidden_ranges: Sequence[tuple[int, int]], total_frames: int,
) -> list[tuple[int, int]]:
    ranges = sorted((int(start), int(end)) for start, end in hidden_ranges)
    previous_end = -1
    for start, end in ranges:
        if start < 0 or end < start or end >= total_frames or start <= previous_end:
            raise BrowserPlaybackError("Hidden playback ranges overlap or exceed the source")
        previous_end = end
    return ranges


def _gap_index_for_frame(
    ranges: list[tuple[int, int]], frame_index: int,
) -> int | None:
    for gap_index, (start, end) in enumerate(ranges):
        if start <= frame_index <= end:
            return gap_index
        if frame_index < start:
            return None
    return None


def _read_visible_boundary_frames(
    source_path: Path,
    ranges: list[tuple[int, int]],
    total_frames: int,
    cancellation_check: CancellationCheck | None,
) -> dict[int, tuple[numpy.ndarray | None, numpy.ndarray | None]]:
    """Cache only the visible frames immediately surrounding each hidden range."""
    requested: dict[int, tuple[int | None, int | None]] = {
        index: (
            start - 1 if start > 0 else None,
            end + 1 if end + 1 < total_frames else None,
        )
        for index, (start, end) in enumerate(ranges)
    }
    wanted = {
        frame_index
        for before_after in requested.values()
        for frame_index in before_after
        if frame_index is not None
    }
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return {}
    captured: dict[int, numpy.ndarray] = {}
    try:
        frame_index = 0
        while frame_index <= max(wanted):
            raise_if_cancelled(cancellation_check)
            success, frame = capture.read()
            if not success:
                break
            if frame_index in wanted:
                captured[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()
    return {
        gap_index: (
            captured.get(before_index) if before_index is not None else None,
            captured.get(after_index) if after_index is not None else None,
        )
        for gap_index, (before_index, after_index) in requested.items()
    }


def _plate_frame(
    plates: PlateProvider | None,
    gap_index: int,
    frame_index: int,
    gap_start: int,
    gap_end: int,
    last_visible_frame: numpy.ndarray | None,
    boundary_frames: tuple[numpy.ndarray | None, numpy.ndarray | None] | None,
    source_frame: numpy.ndarray,
) -> numpy.ndarray:
    if plates is not None:
        try:
            plate = plates(gap_index)
            if isinstance(plate, numpy.ndarray) and plate.size:
                return plate.copy()
        except (KeyError, OSError, RuntimeError, ValueError):
            pass
    before_frame, after_frame = boundary_frames or (None, None)
    if before_frame is None:
        before_frame = last_visible_frame
    if before_frame is not None and after_frame is not None:
        hidden_length = max(1, gap_end - gap_start + 1)
        transition = (frame_index - gap_start + 1) / (hidden_length + 1)
        return _boundary_wipe(before_frame, after_frame, transition)
    if before_frame is not None:
        return before_frame.copy()
    if after_frame is not None:
        return after_frame.copy()
    # Gap placement normally keeps visible context before every interval. This neutral
    # fallback is still safer than accidentally exposing a hidden source frame when a
    # caller supplies a gap beginning at frame zero.
    return numpy.full_like(source_frame, (30, 42, 52))


def _boundary_wipe(
    before_frame: numpy.ndarray,
    after_frame: numpy.ndarray,
    transition: float,
) -> numpy.ndarray:
    """Move one visible boundary across the frame without a full-frame ghost dissolve."""
    height, width = before_frame.shape[:2]
    horizontal, vertical = numpy.meshgrid(
        numpy.linspace(0.0, 1.0, width, dtype=numpy.float32),
        numpy.linspace(0.0, 1.0, height, dtype=numpy.float32),
    )
    diagonal_position = (horizontal + vertical) * 0.5
    feather = max(0.01, BOUNDARY_WIPE_FEATHER)
    boundary = -feather + float(transition) * (1.0 + 2.0 * feather)
    alpha = numpy.clip((boundary - diagonal_position) / feather, 0.0, 1.0)[:, :, None]
    blended = (
        before_frame.astype(numpy.float32) * (1.0 - alpha)
        + after_frame.astype(numpy.float32) * alpha
    )
    return numpy.clip(blended, 0.0, 255.0).astype(numpy.uint8)


def _fit_frame(frame: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
