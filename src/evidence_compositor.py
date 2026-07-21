import bisect
import warnings
from pathlib import Path

import cv2
import numpy as np

from domain.cancellation import CancellationCheck, raise_if_cancelled


DEFAULT_PLATE_SAMPLES = 9
DEFAULT_PLATE_WINDOW_SECONDS = 0.75
DEFAULT_TRANSITION_SECONDS = 0.18


def _read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    success, frame = capture.read()
    if not success:
        raise ValueError(f"Could not read source frame {frame_number}")
    return frame


def _alignment_matrix(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray | None:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 640.0 / reference.shape[1])
    size = (int(reference.shape[1] * scale), int(reference.shape[0] * scale))
    reference_small = cv2.resize(reference_gray, size)
    candidate_small = cv2.resize(candidate_gray, size)
    matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-4)
    try:
        cv2.findTransformECC(reference_small, candidate_small, matrix, cv2.MOTION_EUCLIDEAN, criteria)
    except cv2.error:
        return None
    matrix[0, 2] /= scale
    matrix[1, 2] /= scale
    return matrix


def _align_to_reference(frame: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = _alignment_matrix(reference, frame)
    if matrix is None:
        return frame, None
    height, width = reference.shape[:2]
    aligned = cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, matrix


def _bbox_at(track: dict, frame_number: int, maximum_gap: int) -> list[int] | None:
    detections = track.get("detections", [])
    frames = [item["frame"] for item in detections]
    position = bisect.bisect_left(frames, frame_number)
    if position < len(frames) and frames[position] == frame_number:
        return detections[position]["bbox"]
    before = detections[position - 1] if position else None
    after = detections[position] if position < len(detections) else None
    if before and after and after["frame"] - before["frame"] <= maximum_gap:
        progress = (frame_number - before["frame"]) / max(1, after["frame"] - before["frame"])
        return [int(round(first * (1.0 - progress) + second * progress)) for first, second in zip(before["bbox"], after["bbox"])]
    nearest = before if before and frame_number - before["frame"] <= maximum_gap else after
    return nearest["bbox"] if nearest and abs(nearest["frame"] - frame_number) <= maximum_gap else None


def _foreground_mask(frame_number: int, tracks: list[dict], width: int, height: int, maximum_gap: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for track in tracks:
        bbox = _bbox_at(track, frame_number, maximum_gap)
        if bbox is None:
            continue
        x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
        margin_x = max(2, int((x2 - x1) * 0.08))
        margin_y = max(2, int((y2 - y1) * 0.05))
        cv2.rectangle(
            mask,
            (max(0, x1 - margin_x), max(0, y1 - margin_y)),
            (min(width - 1, x2 + margin_x), min(height - 1, y2 + margin_y)),
            255,
            -1,
        )
    return mask


def _aligned_mask(mask: np.ndarray, matrix: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if matrix is None:
        return mask
    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _sample_indexes(start: int, end: int, maximum_samples: int) -> list[int]:
    if end < start:
        return []
    count = min(maximum_samples, end - start + 1)
    return sorted({int(round(value)) for value in np.linspace(start, end, count)})


def _background_plate(
    capture: cv2.VideoCapture,
    frame_range: tuple[int, int],
    reference_frame: int,
    maximum_samples: int,
    tracks: list[dict],
    detection_maximum_gap: int,
) -> np.ndarray:
    reference = _read_frame(capture, reference_frame)
    aligned_frames = []
    masked_frames = []
    for frame_number in _sample_indexes(frame_range[0], frame_range[1], maximum_samples):
        frame = _read_frame(capture, frame_number)
        aligned, matrix = _align_to_reference(frame, reference)
        mask = _foreground_mask(frame_number, tracks, frame.shape[1], frame.shape[0], detection_maximum_gap)
        aligned_mask = _aligned_mask(mask, matrix, reference.shape[1], reference.shape[0])
        masked = aligned.astype(np.float32)
        masked[aligned_mask > 0] = np.nan
        aligned_frames.append(aligned)
        masked_frames.append(masked)
    if not aligned_frames:
        return reference
    fallback = np.median(np.stack(aligned_frames), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plate = np.nanmedian(np.stack(masked_frames), axis=0)
    plate = np.where(np.isfinite(plate), plate, fallback)
    return np.clip(plate, 0, 255).astype(np.uint8)


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(1, min(width, int(x2))),
        max(1, min(height, int(y2))),
    ]


def _fallback_alpha(height: int, width: int) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, int(width * 0.46)), max(1, int(height * 0.49)))
    cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(alpha, (0, 0), 1.2)


def _foreground_alpha(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    if min(height, width) < 8:
        return _fallback_alpha(height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    inset = max(1, min(width, height) // 35)
    rectangle = (inset, inset, max(1, width - 2 * inset), max(1, height - 2 * inset))
    try:
        cv2.grabCut(crop, mask, rectangle, background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return _fallback_alpha(height, width)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(alpha, (0, 0), 1.1)


def _extract_asset(capture: cv2.VideoCapture, reference: dict | None) -> dict | None:
    if reference is None:
        return None
    frame = _read_frame(capture, int(reference["frame"]))
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(reference["bbox"], width, height)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return {
        "image": crop,
        "alpha": _foreground_alpha(crop),
        "confidence": float(reference.get("confidence", 0.0)),
    }


def _entity_assets(video_path: str, entities: list[dict]) -> dict[str, dict]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    assets = {}
    for entity in entities:
        assets[entity["id"]] = {
            "before": _extract_asset(capture, entity.get("reference_before")),
            "after": _extract_asset(capture, entity.get("reference_after")),
        }
    capture.release()
    return assets


def _resized_asset(asset: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(asset["image"], (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(asset["alpha"], (width, height), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32), alpha.astype(np.float32) / 255.0


def _selected_asset(assets: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    before, after = assets.get("before"), assets.get("after")
    if before and after:
        strongest = before if before["confidence"] >= after["confidence"] else after
        return _resized_asset(strongest, width, height)
    available = before or after
    return _resized_asset(available, width, height) if available else None


def _draw_shadow(frame: np.ndarray, bbox: list[int], opacity: float) -> None:
    x1, _, x2, y2 = bbox
    overlay = np.zeros_like(frame)
    center = ((x1 + x2) // 2, y2)
    axes = (max(3, (x2 - x1) // 3), max(2, (x2 - x1) // 12))
    cv2.ellipse(overlay, center, axes, 0, 0, 360, (30, 30, 30), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), max(2.0, axes[0] * 0.12))
    cv2.addWeighted(overlay, 0.45 * opacity, frame, 1.0, 0, dst=frame)


def _paste_asset(frame: np.ndarray, asset: tuple[np.ndarray, np.ndarray], bbox: list[int], opacity: float) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return
    image, alpha = asset
    source_height, source_width = alpha.shape
    image = image[: min(source_height, y2 - y1), : min(source_width, x2 - x1)]
    alpha = alpha[: image.shape[0], : image.shape[1]] * opacity
    target = frame[y1:y1 + image.shape[0], x1:x1 + image.shape[1]].astype(np.float32)
    blended = image * alpha[..., None] + target * (1.0 - alpha[..., None])
    frame[y1:y1 + image.shape[0], x1:x1 + image.shape[1]] = blended.astype(np.uint8)


def _draw_dashed_path(overlay: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for index in range(1, len(points)):
        if index % 4 < 2:
            cv2.line(overlay, points[index - 1], points[index], color, 1, cv2.LINE_AA)


def _draw_uncertainty(frame: np.ndarray, entities: list[dict], enabled: bool) -> None:
    if not enabled:
        return
    overlay = frame.copy()
    for entity in entities:
        if entity["confidence"] >= 0.62:
            continue
        offset = entity["alternative_path_offset_px"]
        centers = [((item["bbox"][0] + item["bbox"][2]) // 2, item["bbox"][3]) for item in entity["path"]]
        _draw_dashed_path(overlay, [(x - offset, y) for x, y in centers], (0, 190, 255))
        _draw_dashed_path(overlay, [(x + offset, y) for x, y in centers], (0, 190, 255))
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, dst=frame)


def _draw_inference_hud(frame: np.ndarray, confidence: float, duration_seconds: float, opacity: float) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 62), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, dst=frame)
    cv2.putText(frame, "AI-INFERRED EVIDENCE VISUALIZATION", (18, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
    subtitle = f"NOT GROUND TRUTH | confidence {confidence:.0%} | inferred gap {duration_seconds:.2f}s"
    cv2.putText(frame, subtitle, (18, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)


def _boundary_blend(
    frame: np.ndarray,
    before_boundary: np.ndarray,
    after_boundary: np.ndarray,
    frame_offset: int,
    total_frames: int,
    transition_frames: int,
) -> np.ndarray:
    if transition_frames <= 0:
        return frame
    if frame_offset < transition_frames:
        amount = frame_offset / transition_frames
        return cv2.addWeighted(frame, amount, before_boundary, 1.0 - amount, 0)
    remaining = total_frames - frame_offset - 1
    if remaining < transition_frames:
        amount = remaining / transition_frames
        return cv2.addWeighted(frame, amount, after_boundary, 1.0 - amount, 0)
    return frame


def render_evidence_reconstruction(
    output_path: str,
    video_path: str,
    plan: dict,
    scene_report: dict,
    width: int,
    height: int,
    fps: float,
    visual_config: dict | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> None:
    settings = visual_config or {}
    hidden_start = int(plan["hidden_range"]["start"])
    hidden_end = int(plan["hidden_range"]["end"])
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    window_frames = max(1, int(round(settings.get("plate_window_seconds", DEFAULT_PLATE_WINDOW_SECONDS) * fps)))
    detection_maximum_gap = max(1, int(settings.get("plate_detection_max_gap_frames", 16)))
    tracks = scene_report.get("tracks", [])
    before_boundary = _read_frame(capture, hidden_start - 1)
    after_boundary = _read_frame(capture, hidden_end + 1)
    before_plate = _background_plate(
        capture, (max(0, hidden_start - window_frames), hidden_start - 1), hidden_start - 1,
        DEFAULT_PLATE_SAMPLES, tracks, detection_maximum_gap,
    )
    after_plate = _background_plate(
        capture, (hidden_end + 1, hidden_end + window_frames), hidden_end + 1,
        DEFAULT_PLATE_SAMPLES, tracks, detection_maximum_gap,
    )
    capture.release()

    entities = plan.get("entities", [])
    assets = _entity_assets(video_path, entities)
    path_maps = {entity["id"]: {item["frame"]: item for item in entity["path"]} for entity in entities}
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Cannot create inferred gap video: {Path(output_path).name}")
    total_frames = hidden_end - hidden_start + 1
    transition_frames = int(round(settings.get("transition_seconds", DEFAULT_TRANSITION_SECONDS) * fps))
    try:
        for frame_number in range(hidden_start, hidden_end + 1):
            raise_if_cancelled(cancellation_check)
            frame_offset = frame_number - hidden_start
            progress = frame_offset / max(1, total_frames - 1)
            frame = cv2.addWeighted(after_plate, progress, before_plate, 1.0 - progress, 0)
            _draw_uncertainty(frame, entities, settings.get("show_uncertainty_paths", True))
            ordered = sorted(entities, key=lambda item: path_maps[item["id"]][frame_number]["bbox"][3])
            for entity in ordered:
                point = path_maps[entity["id"]][frame_number]
                bbox = point["bbox"]
                target_width = max(2, bbox[2] - bbox[0])
                target_height = max(2, bbox[3] - bbox[1])
                asset = _selected_asset(assets[entity["id"]], target_width, target_height)
                if asset is None:
                    continue
                opacity = float(point["opacity"]) * min(1.0, 0.90 + entity["confidence"] * 0.10)
                _draw_shadow(frame, bbox, opacity)
                _paste_asset(frame, asset, bbox, opacity)
            frame = _boundary_blend(
                frame, before_boundary, after_boundary, frame_offset, total_frames, transition_frames
            )
            _draw_inference_hud(
                frame,
                plan.get("overall_confidence", 0.0),
                total_frames / fps,
                settings.get("hud_opacity", 0.46),
            )
            writer.write(frame)
    finally:
        writer.release()
    rendered_path = Path(output_path)
    if not rendered_path.is_file() or rendered_path.stat().st_size == 0:
        raise OSError(f"Inferred gap video was not written: {rendered_path.name}")
