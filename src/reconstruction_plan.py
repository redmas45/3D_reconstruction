from scene_intelligence import bbox_area


RENDER_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def _nearest_detection(track: dict, target_frame: int, side: str):
    detections = sorted(track.get("detections", []), key=lambda item: item["frame"])
    if not detections:
        return None
    if side == "before":
        candidates = [det for det in detections if det["frame"] <= target_frame]
        return candidates[-1] if candidates else None
    if side == "after":
        candidates = [det for det in detections if det["frame"] >= target_frame]
        return candidates[0] if candidates else None
    return min(detections, key=lambda det: abs(det["frame"] - target_frame))


def _velocity_from_track(track: dict, fps: float):
    detections = sorted(track.get("detections", []), key=lambda item: item["frame"])
    if len(detections) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    first = detections[0]
    last = detections[-1]
    dt = max(1, last["frame"] - first["frame"])
    return [(last["bbox"][i] - first["bbox"][i]) / dt for i in range(4)]


def _lerp_bbox(a, b, t):
    return [int(round((1.0 - t) * a[i] + t * b[i])) for i in range(4)]


def _extrapolate_bbox(bbox, velocity, frame_delta, max_delta=160):
    out = []
    for idx, value in enumerate(bbox):
        delta = max(-max_delta, min(max_delta, velocity[idx] * frame_delta))
        out.append(int(round(value + delta)))
    return out


def _path_for_entity(track: dict, hidden_start: int, hidden_end: int, fps: float):
    before = _nearest_detection(track, hidden_start - 1, "before")
    after = _nearest_detection(track, hidden_end + 1, "after")
    velocity = _velocity_from_track(track, fps)
    total = max(1, hidden_end - hidden_start)

    path = []
    for frame_no in range(hidden_start, hidden_end + 1):
        if before and after:
            t = (frame_no - hidden_start) / total
            bbox = _lerp_bbox(before["bbox"], after["bbox"], t)
        elif before:
            bbox = _extrapolate_bbox(before["bbox"], velocity, frame_no - before["frame"])
        elif after:
            bbox = _extrapolate_bbox(after["bbox"], velocity, frame_no - after["frame"])
        else:
            return []
        path.append({"frame": frame_no, "bbox": bbox})
    return path


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
        if track["frames_seen"] < min_track_frames:
            continue
        if track["avg_area"] < 180:
            continue

        before = _nearest_detection(track, hidden_start - 1, "before")
        after = _nearest_detection(track, hidden_end + 1, "after")
        appears_in_gap = before is not None or after is not None
        if not appears_in_gap:
            continue

        path = _path_for_entity(track, hidden_start, hidden_end, fps)
        if not path:
            continue

        entities.append(
            {
                "id": track["id"],
                "class_name": track["class_name"],
                "direction": track["direction"],
                "confidence": track["avg_confidence"],
                "visible_before_gap": track["visible_before_gap"],
                "visible_after_gap": track["visible_after_gap"],
                "reference_before": before,
                "reference_after": after,
                "path": path,
                "render_priority": bbox_area(path[len(path) // 2]["bbox"]),
            }
        )

    entities.sort(
        key=lambda item: (
            item["visible_before_gap"] and item["visible_after_gap"],
            item["render_priority"],
        ),
        reverse=True,
    )
    entities = entities[:max_entities]
    entities.sort(key=lambda item: item["render_priority"])
    return {
        "hidden_range": {"start": hidden_start, "end": hidden_end},
        "fps": fps,
        "strategy": "yolo_tracks_to_animated_3d_reconstruction",
        "entities": entities,
        "notes": [
            "Visible chunks remain original frames.",
            "Hidden chunk is reconstructed from visible detections, tracks, and entity crops.",
            "No skeleton overlay is rendered in the final video.",
        ],
    }
