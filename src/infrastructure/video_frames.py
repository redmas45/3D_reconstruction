import warnings
from pathlib import Path

import cv2
import numpy as np

from domain.cancellation import CancellationCheck, raise_if_cancelled


JPEG_QUALITY = 92
CONTEXT_DETECTION_FRAME_TOLERANCE = 12
CONTEXT_MASK_PADDING_FRACTION = 0.10
CONTEXT_INPAINT_RADIUS_PIXELS = 7.0
STATIC_CONTEXT_SAMPLE_COUNT = 31
STATIC_CAMERA_CLASSIFICATION = "static_camera"
MEDIAN_ROW_CHUNK_SIZE = 64
CONTEXT_MASK_FEATHER_SIGMA_PIXELS = 9.0
StaticBackgroundCacheKey = tuple[
    str,
    int,
    int,
    tuple[tuple[int, int], ...],
]
_STATIC_BACKGROUND_CACHE: dict[StaticBackgroundCacheKey, np.ndarray] = {}


def export_video_frame(
    video_path: Path,
    frame_index: int,
    output_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    frame = _read_video_frame(video_path, frame_index, cancellation_check)
    return _write_frame(frame, output_path)


def export_forensic_context_frame(
    video_path: Path,
    frame_index: int,
    scene_report: dict,
    output_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    frame = _read_video_frame(video_path, frame_index, cancellation_check)
    mask = _foreground_mask(frame.shape, scene_report, frame_index)
    context_frame = _remove_foreground(
        video_path,
        frame,
        mask,
        scene_report,
        cancellation_check,
    )
    return _write_frame(context_frame, output_path)


def _remove_foreground(
    video_path: Path,
    boundary_frame: np.ndarray,
    mask: np.ndarray,
    scene_report: dict,
    cancellation_check: CancellationCheck | None,
) -> np.ndarray:
    if not cv2.countNonZero(mask):
        return boundary_frame
    if _camera_is_static(scene_report):
        median_frame = _static_background_median(
            video_path,
            scene_report,
            cancellation_check,
        )
        if median_frame is not None:
            return _blend_reconstructed_background(
                boundary_frame,
                median_frame,
                mask,
            )
    return cv2.inpaint(
        boundary_frame,
        mask,
        CONTEXT_INPAINT_RADIUS_PIXELS,
        cv2.INPAINT_TELEA,
    )


def _blend_reconstructed_background(
    boundary_frame: np.ndarray,
    background_frame: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    feathered_mask = cv2.GaussianBlur(
        foreground_mask,
        (0, 0),
        CONTEXT_MASK_FEATHER_SIGMA_PIXELS,
    )
    alpha = feathered_mask.astype(np.float32)[:, :, None] / 255.0
    blended = (
        boundary_frame.astype(np.float32) * (1.0 - alpha)
        + background_frame.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _camera_is_static(scene_report: dict) -> bool:
    motion_report = scene_report.get("camera_motion_report", {})
    return motion_report.get("classification") == STATIC_CAMERA_CLASSIFICATION


def _static_background_median(
    video_path: Path,
    scene_report: dict,
    cancellation_check: CancellationCheck | None,
) -> np.ndarray | None:
    maximum_frame_index = int(scene_report.get("video", {}).get("frames", 0)) - 1
    cache_key = _static_background_cache_key(video_path, scene_report)
    cached_background = _STATIC_BACKGROUND_CACHE.get(cache_key)
    if cached_background is not None:
        return cached_background
    sample_indexes = _distributed_visible_context_indexes(
        maximum_frame_index,
        scene_report.get("hidden_ranges", []),
    )
    frames = []
    foreground_masks = []
    for sample_index in sample_indexes:
        sample_frame = _read_video_frame(
            video_path,
            sample_index,
            cancellation_check,
        )
        frames.append(sample_frame)
        foreground_masks.append(
            _foreground_mask(sample_frame.shape, scene_report, sample_index),
        )
    if len(frames) < 3:
        return None
    background = _masked_frame_median(frames, foreground_masks)
    _STATIC_BACKGROUND_CACHE[cache_key] = background
    return background


def _static_background_cache_key(
    video_path: Path,
    scene_report: dict,
) -> StaticBackgroundCacheKey:
    hidden_ranges = tuple(
        (int(item[0]), int(item[1]))
        for item in scene_report.get("hidden_ranges", [])
        if isinstance(item, (list, tuple)) and len(item) == 2
    )
    return (
        str(video_path.resolve()),
        video_path.stat().st_mtime_ns,
        len(scene_report.get("tracks", [])),
        hidden_ranges,
    )


def _distributed_visible_context_indexes(
    maximum_frame_index: int,
    hidden_ranges: list,
) -> list[int]:
    if maximum_frame_index < 0:
        return []
    candidate_indexes = np.linspace(
        0,
        maximum_frame_index,
        num=STATIC_CONTEXT_SAMPLE_COUNT * 2,
        dtype=int,
    )
    visible_indexes = [
        int(candidate)
        for candidate in candidate_indexes
        if not _frame_is_hidden(int(candidate), hidden_ranges)
    ]
    if len(visible_indexes) <= STATIC_CONTEXT_SAMPLE_COUNT:
        return visible_indexes
    selected_positions = np.linspace(
        0,
        len(visible_indexes) - 1,
        num=STATIC_CONTEXT_SAMPLE_COUNT,
        dtype=int,
    )
    return [visible_indexes[int(position)] for position in selected_positions]


def _masked_frame_median(
    frames: list[np.ndarray],
    foreground_masks: list[np.ndarray],
) -> np.ndarray:
    frame_stack = np.stack(frames, axis=0)
    mask_stack = np.stack(foreground_masks, axis=0)
    fallback = np.median(frame_stack, axis=0).astype(np.uint8)
    result = fallback.copy()
    for row_start in range(0, frame_stack.shape[1], MEDIAN_ROW_CHUNK_SIZE):
        row_end = min(frame_stack.shape[1], row_start + MEDIAN_ROW_CHUNK_SIZE)
        frame_chunk = frame_stack[:, row_start:row_end].astype(np.float32)
        mask_chunk = mask_stack[:, row_start:row_end, :, None] > 0
        frame_chunk[mask_chunk.repeat(3, axis=3)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_chunk = np.nanmedian(frame_chunk, axis=0)
        valid_pixels = ~np.isnan(median_chunk)
        result_chunk = result[row_start:row_end]
        result_chunk[valid_pixels] = median_chunk[valid_pixels].astype(np.uint8)
    return result


def _frame_is_hidden(frame_index: int, hidden_ranges: list) -> bool:
    return any(
        int(hidden_range[0]) <= frame_index <= int(hidden_range[1])
        for hidden_range in hidden_ranges
        if isinstance(hidden_range, (list, tuple)) and len(hidden_range) == 2
    )


def _read_video_frame(
    video_path: Path,
    frame_index: int,
    cancellation_check: CancellationCheck | None,
) -> np.ndarray:
    raise_if_cancelled(cancellation_check)
    if frame_index < 0:
        raise ValueError("Frame index cannot be negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video: {video_path.name}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        read_successfully, frame = capture.read()
    finally:
        capture.release()
    if not read_successfully:
        raise ValueError(f"Cannot read visible evidence frame {frame_index}")
    raise_if_cancelled(cancellation_check)
    return frame


def _foreground_mask(
    frame_shape: tuple[int, ...],
    scene_report: dict,
    frame_index: int,
) -> np.ndarray:
    frame_height, frame_width = frame_shape[:2]
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for track in scene_report.get("tracks", []):
        detection = _nearest_detection(track.get("detections", []), frame_index)
        if detection is None:
            continue
        _mask_detection(mask, detection.get("bbox", []), frame_width, frame_height)
    return mask


def _nearest_detection(detections: list[dict], frame_index: int) -> dict | None:
    candidates = [
        detection for detection in detections
        if abs(int(detection.get("frame", -1)) - frame_index)
        <= CONTEXT_DETECTION_FRAME_TOLERANCE
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(int(item["frame"]) - frame_index))


def _mask_detection(
    mask: np.ndarray,
    bbox: list,
    frame_width: int,
    frame_height: int,
) -> None:
    if len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    padding_x = round(max(1, x2 - x1) * CONTEXT_MASK_PADDING_FRACTION)
    padding_y = round(max(1, y2 - y1) * CONTEXT_MASK_PADDING_FRACTION)
    start = (max(0, x1 - padding_x), max(0, y1 - padding_y))
    end = (min(frame_width - 1, x2 + padding_x), min(frame_height - 1, y2 + padding_y))
    cv2.rectangle(mask, start, end, 255, -1)


def _write_frame(frame: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_successfully = cv2.imwrite(
        str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not write_successfully:
        raise OSError(f"Cannot write visible evidence frame to {output_path}")
    return output_path
