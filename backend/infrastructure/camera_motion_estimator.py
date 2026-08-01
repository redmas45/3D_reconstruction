import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from domain.cancellation import CancellationCheck, raise_if_cancelled


CAMERA_MOTION_SAMPLE_LIMIT = 12
PAIR_FRAME_DISTANCE = 5
MAXIMUM_ANALYSIS_WIDTH = 720
MINIMUM_FEATURE_MATCHES = 12
STATIC_TRANSLATION_PIXELS_PER_FRAME = 0.75
STATIC_ROTATION_DEGREES_PER_FRAME = 0.08
STATIC_SCALE_CHANGE_PER_FRAME = 0.002
MAXIMUM_FIT_RESIDUAL_PIXELS = 4.0
FOREGROUND_MASK_PADDING_PIXELS = 8


def estimate_camera_motion(
    video_path: Path,
    scene_report: dict,
    cancellation_check: CancellationCheck | None = None,
) -> dict:
    sample_pairs = _sample_pairs(scene_report.get("visible_ranges", []))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot inspect camera motion in {video_path.name}")
    pair_reports: list[dict] = []
    try:
        for first_frame, second_frame in sample_pairs:
            raise_if_cancelled(cancellation_check)
            pair_report = _measure_pair(capture, scene_report, first_frame, second_frame)
            if pair_report is not None:
                pair_reports.append(pair_report)
    finally:
        capture.release()
    return summarize_camera_motion(pair_reports, len(sample_pairs))


def _sample_pairs(visible_ranges: list[dict]) -> list[tuple[int, int]]:
    eligible_ranges = [
        item for item in visible_ranges
        if int(item["end"]) - int(item["start"]) >= PAIR_FRAME_DISTANCE * 2
    ]
    if not eligible_ranges:
        return []
    indexes = _evenly_spaced_indexes(len(eligible_ranges), CAMERA_MOTION_SAMPLE_LIMIT)
    pairs: list[tuple[int, int]] = []
    for range_index in indexes:
        visible_range = eligible_ranges[range_index]
        midpoint = (int(visible_range["start"]) + int(visible_range["end"])) // 2
        pairs.append((midpoint - PAIR_FRAME_DISTANCE, midpoint))
    return pairs


def _measure_pair(
    capture: cv2.VideoCapture,
    scene_report: dict,
    first_frame_index: int,
    second_frame_index: int,
) -> dict | None:
    first_frame = _read_frame(capture, first_frame_index)
    second_frame = _read_frame(capture, second_frame_index)
    if first_frame is None or second_frame is None:
        return None
    first_gray, scale = _analysis_frame(first_frame)
    second_gray, _ = _analysis_frame(second_frame)
    first_mask = _background_mask(first_gray.shape, scene_report, first_frame_index, scale)
    second_mask = _background_mask(second_gray.shape, scene_report, second_frame_index, scale)
    return _feature_transform(first_gray, second_gray, first_mask, second_mask, first_frame_index, second_frame_index)


def _feature_transform(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_frame: int,
    second_frame: int,
) -> dict | None:
    detector = cv2.ORB_create(nfeatures=1_200, fastThreshold=12)
    first_keypoints, first_descriptors = detector.detectAndCompute(first_gray, first_mask)
    second_keypoints, second_descriptors = detector.detectAndCompute(second_gray, second_mask)
    if first_descriptors is None or second_descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(first_descriptors, second_descriptors)
    matches = sorted(matches, key=lambda match: match.distance)[:240]
    if len(matches) < MINIMUM_FEATURE_MATCHES:
        return None
    source_points = np.float32([first_keypoints[match.queryIdx].pt for match in matches])
    target_points = np.float32([second_keypoints[match.trainIdx].pt for match in matches])
    transform, inliers = cv2.estimateAffinePartial2D(
        source_points, target_points, method=cv2.RANSAC, ransacReprojThreshold=2.5
    )
    if transform is None or inliers is None:
        return None
    return _pair_report(transform, source_points, target_points, inliers, first_frame, second_frame)


def _pair_report(
    transform: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    inliers: np.ndarray,
    first_frame: int,
    second_frame: int,
) -> dict:
    frame_distance = max(1, second_frame - first_frame)
    inlier_mask = inliers.ravel().astype(bool)
    transformed_points = cv2.transform(source_points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    residuals = np.linalg.norm(transformed_points[inlier_mask] - target_points[inlier_mask], axis=1)
    scale = math.sqrt(float(transform[0, 0] ** 2 + transform[0, 1] ** 2))
    translation = math.hypot(float(transform[0, 2]), float(transform[1, 2]))
    rotation = abs(math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0]))))
    return {
        "first_frame": first_frame,
        "second_frame": second_frame,
        "translation_pixels_per_frame": round(translation / frame_distance, 5),
        "rotation_degrees_per_frame": round(rotation / frame_distance, 5),
        "scale_change_per_frame": round(abs(scale - 1.0) / frame_distance, 7),
        "inlier_ratio": round(float(inlier_mask.mean()), 5),
        "fit_residual_pixels": round(float(np.median(residuals)) if residuals.size else 99.0, 5),
    }


def summarize_camera_motion(pair_reports: list[dict], requested_samples: int) -> dict:
    if not pair_reports:
        return _unclassified_motion_report(requested_samples)
    translation = _median_metric(pair_reports, "translation_pixels_per_frame", 99.0)
    rotation = _median_metric(pair_reports, "rotation_degrees_per_frame", 99.0)
    scale_change = _median_metric(pair_reports, "scale_change_per_frame", 99.0)
    inlier_ratio = _median_metric(pair_reports, "inlier_ratio", 0.0)
    residual = _median_metric(pair_reports, "fit_residual_pixels", 99.0)
    static_camera = (
        translation <= STATIC_TRANSLATION_PIXELS_PER_FRAME
        and rotation <= STATIC_ROTATION_DEGREES_PER_FRAME
        and scale_change <= STATIC_SCALE_CHANGE_PER_FRAME
    )
    fit_score = max(0.0, min(1.0, 1.0 - residual / MAXIMUM_FIT_RESIDUAL_PIXELS))
    return {
        "classification": "static_camera" if static_camera else "dynamic_camera",
        "render_transform_available": False,
        "sample_count": len(pair_reports),
        "requested_sample_count": requested_samples,
        "median_translation_pixels_per_frame": round(translation, 5),
        "median_rotation_degrees_per_frame": round(rotation, 5),
        "median_scale_change_per_frame": round(scale_change, 7),
        "static_feature_inlier_score": round(inlier_ratio, 4),
        "camera_motion_fit_score": round(fit_score, 4),
        "pair_reports": pair_reports,
    }


def _unclassified_motion_report(requested_samples: int) -> dict:
    return {
        "classification": "unclassified",
        "render_transform_available": False,
        "sample_count": 0,
        "requested_sample_count": requested_samples,
        "median_translation_pixels_per_frame": None,
        "median_rotation_degrees_per_frame": None,
        "median_scale_change_per_frame": None,
        "static_feature_inlier_score": 0.0,
        "camera_motion_fit_score": 0.0,
        "pair_reports": [],
    }


def _background_mask(
    image_shape: tuple[int, ...], scene_report: dict, frame_index: int, scale: float,
) -> np.ndarray:
    mask = np.full(image_shape, 255, dtype=np.uint8)
    for track in scene_report.get("tracks", []):
        detection = _nearest_detection(track.get("detections", []), frame_index)
        if detection is None:
            continue
        x1, y1, x2, y2 = [round(float(value) * scale) for value in detection["bbox"]]
        padding = round(FOREGROUND_MASK_PADDING_PIXELS * scale)
        cv2.rectangle(mask, (max(0, x1 - padding), max(0, y1 - padding)), (x2 + padding, y2 + padding), 0, -1)
    return mask


def _nearest_detection(detections: list[dict], frame_index: int) -> dict | None:
    candidates = [item for item in detections if abs(int(item["frame"]) - frame_index) <= PAIR_FRAME_DISTANCE]
    return min(candidates, key=lambda item: abs(int(item["frame"]) - frame_index)) if candidates else None


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    read_successfully, frame = capture.read()
    return frame if read_successfully else None


def _analysis_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, MAXIMUM_ANALYSIS_WIDTH / float(frame_width))
    resized = cv2.resize(frame, (round(frame_width * scale), round(frame_height * scale)))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), scale


def _median_metric(reports: list[dict], key: str, fallback: float) -> float:
    values = [float(report[key]) for report in reports]
    return median(values) if values else fallback


def _evenly_spaced_indexes(item_count: int, limit: int) -> list[int]:
    selected_count = min(item_count, limit)
    if selected_count == 1:
        return [0]
    return [round(index * (item_count - 1) / (selected_count - 1)) for index in range(selected_count)]
