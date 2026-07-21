"""Inspects source-video metadata and computes stable source identities."""

import hashlib
import math
from pathlib import Path

import cv2

from domain.cancellation import CancellationCheck, raise_if_cancelled


FILE_HASH_CHUNK_BYTES = 4 * 1024 * 1024
MAXIMUM_SOURCE_DIMENSION = 4_096
MAXIMUM_SOURCE_PIXELS = 3_840 * 2_160
MAXIMUM_SOURCE_DURATION_SECONDS = 600.0
MAXIMUM_SOURCE_FPS = 120.0


def inspect_source_video(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")
    try:
        video_metadata = _read_video_metadata(capture)
        first_frame_read, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, video_metadata["frames"] - 1))
        last_frame_read, _ = capture.read()
    finally:
        capture.release()
    _validate_decodable_source(video_path, video_metadata, first_frame_read, last_frame_read)
    return video_metadata


def _read_video_metadata(capture: cv2.VideoCapture) -> dict:
    try:
        raw_values = {
            "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": float(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Video metadata is invalid") from error
    if any(not math.isfinite(value) for value in raw_values.values()):
        raise ValueError("Video metadata is invalid")
    video_metadata = {
        "width": int(raw_values["width"]),
        "height": int(raw_values["height"]),
        "fps": raw_values["fps"],
        "frames": int(raw_values["frames"]),
    }
    validate_source_resource_limits(video_metadata)
    return video_metadata


def _validate_decodable_source(
    video_path: Path,
    video_metadata: dict,
    first_frame_read: bool,
    last_frame_read: bool,
) -> None:
    if not _has_valid_decode_contract(video_metadata, first_frame_read, last_frame_read):
        raise ValueError(f"Video is unreadable or too short: {video_path.name}")
    width, height = video_metadata["width"], video_metadata["height"]
    if width % 2 or height % 2:
        raise ValueError(
            "Video width and height must be even for compatible H.264 output; "
            f"{video_path.name} is {width}x{height}"
        )


def _has_valid_decode_contract(video_metadata: dict, first_frame_read: bool, last_frame_read: bool) -> bool:
    return all((
        video_metadata["frames"] >= 4,
        video_metadata["width"] >= 1,
        video_metadata["height"] >= 1,
        video_metadata["fps"] > 0.0,
        first_frame_read,
        last_frame_read,
    ))


def validate_source_resource_limits(video_metadata: dict) -> None:
    width, height = int(video_metadata["width"]), int(video_metadata["height"])
    fps, frame_count = float(video_metadata["fps"]), int(video_metadata["frames"])
    if max(width, height) > MAXIMUM_SOURCE_DIMENSION or width * height > MAXIMUM_SOURCE_PIXELS:
        raise ValueError("Video resolution exceeds the supported 4K pixel budget")
    if fps > MAXIMUM_SOURCE_FPS:
        raise ValueError(f"Video frame rate exceeds the supported {MAXIMUM_SOURCE_FPS:g} fps limit")
    maximum_frame_count = math.floor(MAXIMUM_SOURCE_DURATION_SECONDS * fps) if fps > 0.0 else None
    if maximum_frame_count is not None and frame_count > maximum_frame_count:
        raise ValueError("Video duration exceeds the supported 10-minute limit")


def source_video_sha256(video_path: Path, cancellation_check: CancellationCheck | None) -> str:
    digest = hashlib.sha256()
    with video_path.open("rb") as source_file:
        while chunk := source_file.read(FILE_HASH_CHUNK_BYTES):
            raise_if_cancelled(cancellation_check)
            digest.update(chunk)
    return digest.hexdigest()
