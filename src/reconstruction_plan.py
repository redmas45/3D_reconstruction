from scene_intelligence import bbox_area


RENDER_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
DEFAULT_REFERENCE_AGE_SECONDS = 2.0


def _nearest_detection(track: dict, target_frame: int, side: str) -> dict | None:
    detections = sorted(track.get("detections", []), key=lambda item: item["frame"])
    if side == "before":
        candidates = [item for item in detections if item["frame"] <= target_frame]
        return candidates[-1] if candidates else None
    candidates = [item for item in detections if item["frame"] >= target_frame]
    return candidates[0] if candidates else None


def _boundary_velocity(track: dict, boundary_frame: int, side: str) -> list[float]:
    detections = sorted(track.get("detections", []), key=lambda item: item["frame"])
    if side == "after":
        sample = [item for item in detections if item["frame"] >= boundary_frame][:3]
    else:
        sample = [item for item in detections if item["frame"] <= boundary_frame][-3:]
    if len(sample) < 2:
        return [0.0] * 4
    first, last = sample[0], sample[-1]
    elapsed = max(1, last["frame"] - first["frame"])
    return [(last["bbox"][index] - first["bbox"][index]) / elapsed for index in range(4)]


def _extrapolated_bbox(bbox: list[int], velocity: list[float], frame_delta: int) -> list[int]:
    max_displacement = 120
    return [
        int(round(value + max(-max_displacement, min(max_displacement, velocity[index] * frame_delta))))
        for index, value in enumerate(bbox)
    ]


def _path_for_entity(
    track: dict,
    before: dict | None,
    after: dict | None,
    hidden_start: int,
    hidden_end: int,
    confidence: float,
) -> list[dict]:
    frame_count = max(1, hidden_end - hidden_start)
    before_velocity = _boundary_velocity(track, hidden_start - 1, "before")
    after_velocity = _boundary_velocity(track, hidden_end + 1, "after")
    path = []
    for frame_number in range(hidden_start, hidden_end + 1):
        progress = (frame_number - hidden_start) / frame_count
        bbox, opacity = _path_point(before, after, before_velocity, after_velocity, progress, frame_count)
        uncertainty = int(round((1.0 - confidence) * max(24, bbox[3] - bbox[1]) * 0.65))
        path.append({"frame": frame_number, "bbox": bbox, "opacity": opacity, "uncertainty_px": uncertainty})
    return path


def _path_point(
    before: dict | None,
    after: dict | None,
    before_velocity: list[float],
    after_velocity: list[float],
    progress: float,
    frame_count: int,
) -> tuple[list[int], float]:
    if before:
        bbox = _extrapolated_bbox(before["bbox"], before_velocity, int(progress * frame_count))
        opacity = 1.0 if after else max(0.0, 1.0 - progress)
        return bbox, opacity
    if after:
        delta = -int((1.0 - progress) * frame_count)
        bbox = _extrapolated_bbox(after["bbox"], after_velocity, delta)
        return bbox, max(0.0, progress)
    raise ValueError("A path requires before or after evidence")


def _valid_reference(reference: dict | None, boundary: int, fps: float) -> dict | None:
    if reference is None:
        return None
    maximum_age = max(1, int(round(DEFAULT_REFERENCE_AGE_SECONDS * fps)))
    return reference if abs(reference["frame"] - boundary) <= maximum_age else None


def _entity_confidence(track: dict, before: dict | None, after: dict | None, gap_seconds: float) -> float:
    detector_confidence = float(track.get("avg_confidence", 0.0))
    continuity = float(track.get("continuity_confidence", 0.0)) if before and after else 0.35
    duration_factor = max(0.55, 1.0 - max(0.0, gap_seconds - 1.0) * 0.12)
    return round(max(0.05, min(0.99, (0.55 * detector_confidence + 0.45 * continuity) * duration_factor)), 3)


def _planned_entity(track: dict, hidden_start: int, hidden_end: int, fps: float) -> dict | None:
    before = _valid_reference(_nearest_detection(track, hidden_start - 1, "before"), hidden_start - 1, fps)
    after = _valid_reference(_nearest_detection(track, hidden_end + 1, "after"), hidden_end + 1, fps)
    if before is None and after is None:
        return None
    gap_seconds = (hidden_end - hidden_start + 1) / fps
    confidence = _entity_confidence(track, before, after, gap_seconds)
    path = _path_for_entity(track, before, after, hidden_start, hidden_end, confidence)
    return {
        "id": track["id"],
        "class_name": track["class_name"],
        "direction": track["direction"],
        "confidence": confidence,
        "visible_before_gap": before is not None,
        "visible_after_gap": after is not None,
        "path_constraint_mode": "forward_prediction" if before else "reverse_entry_prediction",
        "post_gap_observation_role": (
            "soft_consistency_check" if before and after else "entry_boundary_evidence"
        ),
        "reference_before": before,
        "reference_after": after,
        "associated_objects": track.get("associated_objects", []),
        "path": path,
        "alternative_path_offset_px": max(8, path[len(path) // 2]["uncertainty_px"]),
        "render_priority": bbox_area(path[len(path) // 2]["bbox"]),
    }


def build_reconstruction_plan(
    scene_report: dict,
    hidden_range: tuple[int, int],
    fps: float,
    max_entities: int = 8,
    min_track_frames: int = 3,
) -> dict:
    hidden_start, hidden_end = hidden_range
    entities = []
    for track in scene_report.get("tracks", []):
        if track["class_name"] not in RENDER_CLASSES:
            continue
        if track["frames_seen"] < min_track_frames or track["avg_area"] < 180:
            continue
        entity = _planned_entity(track, hidden_start, hidden_end, fps)
        if entity:
            entities.append(entity)
    entities.sort(key=lambda item: (item["visible_before_gap"] and item["visible_after_gap"], item["render_priority"]), reverse=True)
    entities = entities[:max_entities]
    overall_confidence = round(sum(item["confidence"] for item in entities) / max(1, len(entities)), 3)
    return {
        "hidden_range": {"start": hidden_start, "end": hidden_end},
        "fps": fps,
        "strategy": "evidence_grounded_2_5d_compositing",
        "overall_confidence": overall_confidence,
        "entities": entities,
        "notes": [
            "Visible chunks remain original evidence with detection overlays.",
            "The hidden gap uses only visible boundary frames, tracked entity crops, and inferred paths.",
            "Confidence and alternative paths communicate uncertainty; this is not recovered ground truth.",
        ],
    }
