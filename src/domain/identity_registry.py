import hashlib
import json
import random
from pathlib import Path
from statistics import median

import cv2
import numpy as np


IDENTITY_SCHEMA_VERSION = 1
IDENTITY_GENERATOR_VERSION = "procedural_forensic_v1"
APPEARANCE_SAMPLE_LIMIT = 3
DEFAULT_UPPER_COLOR = [0.13, 0.48, 0.54]
DEFAULT_LOWER_COLOR = [0.16, 0.19, 0.24]
RENDERABLE_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def build_identity_registry(scene_report: dict, video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot sample visible appearance from {video_path.name}")
    try:
        identities = {
            track["id"]: _identity_for_track(capture, video_path.stem, track)
            for track in scene_report.get("tracks", [])
            if track.get("class_name") in RENDERABLE_CLASSES
        }
    finally:
        capture.release()
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "generator_version": IDENTITY_GENERATOR_VERSION,
        "video_id": video_path.stem,
        "identities": identities,
    }


def write_identity_registry(registry: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as registry_file:
        json.dump(registry, registry_file, indent=2)


def _identity_for_track(capture: cv2.VideoCapture, video_id: str, track: dict) -> dict:
    seed = _identity_seed(video_id, str(track["id"]))
    random_generator = random.Random(seed)
    sampled_colors = _sample_appearance_colors(capture, track.get("detections", []))
    upper_color = sampled_colors[0] if sampled_colors else _jitter_color(DEFAULT_UPPER_COLOR, random_generator)
    lower_color = sampled_colors[1] if sampled_colors else _jitter_color(DEFAULT_LOWER_COLOR, random_generator)
    return {
        "id": track["id"],
        "kind": track["class_name"],
        "seed": seed,
        "body_proportions": {
            "height_scale": round(random_generator.uniform(0.92, 1.08), 4),
            "shoulder_scale": round(random_generator.uniform(0.88, 1.12), 4),
            "limb_scale": round(random_generator.uniform(0.94, 1.06), 4),
        },
        "appearance": {
            "upper_color": upper_color,
            "lower_color": lower_color,
            "vehicle_color": upper_color,
            "source": "visible_evidence" if sampled_colors else "deterministic_fallback",
        },
        "animation_phase": round(random_generator.random(), 4),
        "associated_objects": list(track.get("associated_objects", [])),
        "evidence_count": len(track.get("detections", [])),
        "appearance_confidence": round(float(track.get("avg_confidence", 0.0)), 4),
    }


def _sample_appearance_colors(capture: cv2.VideoCapture, detections: list[dict]) -> tuple[list[float], list[float]] | None:
    selected = _evenly_spaced(detections, APPEARANCE_SAMPLE_LIMIT)
    upper_samples: list[list[float]] = []
    lower_samples: list[list[float]] = []
    for detection in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(detection["frame"]))
        success, frame = capture.read()
        if not success:
            continue
        sample = _colors_from_bbox(frame, detection.get("bbox", []))
        if sample is not None:
            upper_samples.append(sample[0])
            lower_samples.append(sample[1])
    if not upper_samples:
        return None
    return _median_color(upper_samples), _median_color(lower_samples)


def _colors_from_bbox(frame: np.ndarray, bbox: list[int]) -> tuple[list[float], list[float]] | None:
    if len(bbox) != 4:
        return None
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(frame_width, x2)
    y1, y2 = max(0, y1), min(frame_height, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    person_height = y2 - y1
    margin = max(1, int((x2 - x1) * 0.16))
    upper = frame[y1 + int(person_height * 0.16):y1 + int(person_height * 0.50), x1 + margin:x2 - margin]
    lower = frame[y1 + int(person_height * 0.55):y1 + int(person_height * 0.88), x1 + margin:x2 - margin]
    if upper.size == 0 or lower.size == 0:
        return None
    return _robust_rgb(upper), _robust_rgb(lower)


def _robust_rgb(crop: np.ndarray) -> list[float]:
    pixels = crop.reshape(-1, 3)
    brightness = pixels.mean(axis=1)
    retained = pixels[(brightness >= np.percentile(brightness, 15)) & (brightness <= np.percentile(brightness, 85))]
    color_bgr = np.median(retained if len(retained) else pixels, axis=0)
    return [round(float(channel) / 255.0, 4) for channel in color_bgr[::-1]]


def _median_color(colors: list[list[float]]) -> list[float]:
    return [round(median(color[channel] for color in colors), 4) for channel in range(3)]


def _evenly_spaced(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return list(items)
    indexes = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indexes]


def _identity_seed(video_id: str, track_id: str) -> int:
    digest = hashlib.sha256(f"{video_id}:{track_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _jitter_color(base_color: list[float], random_generator: random.Random) -> list[float]:
    return [round(max(0.04, min(0.92, value + random_generator.uniform(-0.12, 0.12))), 4) for value in base_color]
