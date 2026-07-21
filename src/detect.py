import gc
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO

from domain.cancellation import CancellationCheck, raise_if_cancelled


RELEVANT_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    43: "knife",
    67: "cell phone",
}


def _scale_frame(frame: np.ndarray, downscale_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if not downscale_width or width <= downscale_width:
        return frame, 1.0
    scale = downscale_width / width
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return resized, scale


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    ]


def _appearance_descriptor(frame: np.ndarray, bbox: list[int]) -> list[float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * 32
    crop_height, crop_width = crop.shape[:2]
    margin_x = int(crop_width * 0.12)
    margin_y = int(crop_height * 0.08)
    interior = crop[margin_y:max(margin_y + 1, crop_height - margin_y), margin_x:max(margin_x + 1, crop_width - margin_x)]
    hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return [round(float(value), 6) for value in histogram.flatten()]


def _reset_tracker(model: YOLO) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []) if predictor else []:
        tracker.reset()


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _box_detection(
    box: object,
    frame: np.ndarray,
    frame_index: int,
    segment_index: int,
    scale: float,
    class_names: dict[int, str],
) -> dict:
    class_id = int(box.cls[0])
    coordinates = box.xyxy[0].cpu().numpy().tolist()
    bbox = [int(value / scale) for value in coordinates]
    return {
        "frame": frame_index,
        "segment_index": segment_index,
        "source_track_id": int(box.id[0]) if box.id is not None else -1,
        "class_id": class_id,
        "class_name": class_names.get(class_id, str(class_id)),
        "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
        "bbox": bbox,
        "appearance": _appearance_descriptor(frame, bbox),
    }


def _detect_range(
    model: YOLO,
    capture: cv2.VideoCapture,
    frame_range: tuple[int, int],
    segment_index: int,
    settings: dict,
    cancellation_check: CancellationCheck | None,
) -> tuple[list[dict], int]:
    start, end = frame_range
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    detections: list[dict] = []
    processed_frames = 0
    for frame_index in range(start, end + 1):
        raise_if_cancelled(cancellation_check)
        success, frame = capture.read()
        if not success:
            raise ValueError(f"Video decoding stopped at evidence frame {frame_index}")
        if (frame_index - start) % settings["frame_stride"] != 0:
            continue
        resized, scale = _scale_frame(frame, settings["downscale_width"])
        results = model.track(resized, persist=True, verbose=False, **settings["track_args"])
        raise_if_cancelled(cancellation_check)
        processed_frames += 1
        if not results or len(results[0].boxes) == 0:
            continue
        for box in results[0].boxes:
            detections.append(_box_detection(box, frame, frame_index, segment_index, scale, model.names))
    return detections, processed_frames


def detect_scene_objects(
    video_path: str,
    visible_ranges: list[tuple[int, int]],
    model_name: str = "yolo26m.pt",
    class_ids: list[int] | None = None,
    frame_stride: int = 8,
    downscale_width: int = 960,
    conf: float = 0.25,
    tracker_config: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> list[dict]:
    selected_classes = class_ids or sorted(RELEVANT_COCO_CLASSES)
    print(f"[Detector] Initializing {model_name} for sequential scene tracking...")
    raise_if_cancelled(cancellation_check)
    model = YOLO(model_name)
    raise_if_cancelled(cancellation_check)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    track_args = {"classes": selected_classes, "conf": conf}
    if tracker_config:
        track_args["tracker"] = str(Path(tracker_config).resolve())
    settings = {
        "frame_stride": max(1, int(frame_stride)),
        "downscale_width": max(0, int(downscale_width)),
        "track_args": track_args,
    }
    detections: list[dict] = []
    processed_frames = 0
    segment_total = len(visible_ranges)
    try:
        for segment_index, frame_range in enumerate(visible_ranges):
            raise_if_cancelled(cancellation_check)
            if segment_index:
                _reset_tracker(model)
            segment_detections, segment_frames = _detect_range(
                model, capture, frame_range, segment_index, settings, cancellation_check
            )
            detections.extend(segment_detections)
            processed_frames += segment_frames
            print(f"[Detector] Segment {segment_index + 1}: {segment_frames} sampled frames.")
            if progress_callback is not None:
                progress_callback(segment_index + 1, segment_total)
    finally:
        capture.release()
        del model
        _release_cuda_cache()
    print(f"[Detector] Finished: {len(detections)} detections from {processed_frames} sampled frames.")
    return detections
