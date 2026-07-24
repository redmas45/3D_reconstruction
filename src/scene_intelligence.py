import math
from collections import Counter, defaultdict

import numpy as np


CARRIED_CLASSES = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
REIDENTIFICATION_THRESHOLD = 0.48
MINIMUM_APPEARANCE_SIMILARITY = 0.35


def bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: list[int]) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def bbox_diag(bbox: list[int]) -> float:
    return math.hypot(max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _mean_descriptor(detections: list[dict]) -> list[float]:
    descriptors = [item.get("appearance", []) for item in detections]
    descriptors = [item for item in descriptors if item and any(item)]
    if not descriptors:
        return [0.0] * 32
    return np.mean(np.asarray(descriptors, dtype=np.float32), axis=0).tolist()


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    first_vector = np.asarray(first, dtype=np.float32)
    second_vector = np.asarray(second, dtype=np.float32)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(first_vector, second_vector) / denominator)))


def _local_tracks(detections: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    anonymous_index = 0
    for detection in sorted(detections, key=lambda item: item["frame"]):
        source_id = int(detection.get("source_track_id", -1))
        if source_id < 0:
            anonymous_index += 1
            source_id = -anonymous_index
        key = (int(detection.get("segment_index", 0)), detection["class_name"], source_id)
        grouped[key].append(detection)
    return [
        {
            "segment_index": key[0],
            "class_name": key[1],
            "class_id": items[0]["class_id"],
            "detections": sorted(items, key=lambda item: item["frame"]),
            "appearance": _mean_descriptor(items),
        }
        for key, items in grouped.items()
    ]


def _center_velocity(track: dict, side: str) -> tuple[float, float]:
    detections = track["detections"]
    sample = detections[:3] if side == "start" else detections[-3:]
    if len(sample) < 2:
        return 0.0, 0.0
    first, last = sample[0], sample[-1]
    elapsed_frames = max(1, last["frame"] - first["frame"])
    first_center = bbox_center(first["bbox"])
    last_center = bbox_center(last["bbox"])
    return (
        (last_center[0] - first_center[0]) / elapsed_frames,
        (last_center[1] - first_center[1]) / elapsed_frames,
    )


def _reidentification_score(before: dict, after: dict) -> dict:
    before_last = before["detections"][-1]
    after_first = after["detections"][0]
    elapsed = max(1, after_first["frame"] - before_last["frame"])
    velocity = _center_velocity(before, "end")
    before_center = bbox_center(before_last["bbox"])
    predicted = (before_center[0] + velocity[0] * elapsed, before_center[1] + velocity[1] * elapsed)
    spatial_scale = max(50.0, (bbox_diag(before_last["bbox"]) + bbox_diag(after_first["bbox"])) * 1.5)
    spatial = math.exp(-distance(predicted, bbox_center(after_first["bbox"])) / spatial_scale)
    area_ratio = min(bbox_area(before_last["bbox"]), bbox_area(after_first["bbox"])) / max(
        1.0, max(bbox_area(before_last["bbox"]), bbox_area(after_first["bbox"]))
    )
    appearance = _cosine_similarity(before["appearance"], after["appearance"])
    appearance_conflict = appearance < MINIMUM_APPEARANCE_SIMILARITY
    score = 0.0 if appearance_conflict else (0.55 * appearance) + (0.30 * spatial) + (0.15 * area_ratio)
    return {
        "score": round(score, 4),
        "appearance": round(appearance, 4),
        "spatial": round(spatial, 4),
        "appearance_conflict": appearance_conflict,
    }


def _find_root(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], first: int, second: int) -> None:
    first_root = _find_root(parents, first)
    second_root = _find_root(parents, second)
    if first_root != second_root:
        parents[second_root] = first_root


def _components_have_compatible_appearance(
    parents: list[int], tracks: list[dict], first: int, second: int,
) -> bool:
    first_root = _find_root(parents, first)
    second_root = _find_root(parents, second)
    if first_root == second_root:
        return True
    first_component = [index for index in range(len(tracks)) if _find_root(parents, index) == first_root]
    second_component = [index for index in range(len(tracks)) if _find_root(parents, index) == second_root]
    return all(
        _cosine_similarity(tracks[left]["appearance"], tracks[right]["appearance"])
        >= MINIMUM_APPEARANCE_SIMILARITY
        for left in first_component
        for right in second_component
    )


def _boundary_matches(local_tracks: list[dict], before_segment: int, after_segment: int) -> list[tuple[int, int, dict]]:
    before_indexes = [index for index, track in enumerate(local_tracks) if track["segment_index"] == before_segment]
    after_indexes = [index for index, track in enumerate(local_tracks) if track["segment_index"] == after_segment]
    candidates: list[tuple[float, int, int, dict]] = []
    for before_index in before_indexes:
        for after_index in after_indexes:
            if local_tracks[before_index]["class_name"] != local_tracks[after_index]["class_name"]:
                continue
            evidence = _reidentification_score(local_tracks[before_index], local_tracks[after_index])
            if evidence["appearance_conflict"]:
                continue
            candidates.append((evidence["score"], before_index, after_index, evidence))
    return _select_one_to_one_matches(sorted(candidates, reverse=True))


def _select_one_to_one_matches(candidates: list[tuple[float, int, int, dict]]) -> list[tuple[int, int, dict]]:
    used_before: set[int] = set()
    used_after: set[int] = set()
    matches = []
    for score, before_index, after_index, evidence in candidates:
        if score < REIDENTIFICATION_THRESHOLD:
            break
        if before_index in used_before or after_index in used_after:
            continue
        matches.append((before_index, after_index, evidence))
        used_before.add(before_index)
        used_after.add(after_index)
    return matches


def _combine_component(tracks: list[dict], evidence: list[dict]) -> dict:
    detections = sorted(
        [item for track in tracks for item in track["detections"]],
        key=lambda item: item["frame"],
    )
    scores = [item["score"] for item in evidence]
    return {
        "segment_index": min(track["segment_index"] for track in tracks),
        "class_name": tracks[0]["class_name"],
        "class_id": tracks[0]["class_id"],
        "detections": detections,
        "appearance": _mean_descriptor(detections),
        "continuity_confidence": round(float(np.mean(scores)), 4) if scores else None,
        "reidentification": evidence,
    }


def _merge_across_gaps(local_tracks: list[dict]) -> list[dict]:
    if not local_tracks:
        return []
    parents = list(range(len(local_tracks)))
    match_evidence: list[tuple[int, int, dict]] = []
    maximum_segment = max(track["segment_index"] for track in local_tracks)
    for segment_index in range(maximum_segment):
        for before_index, after_index, evidence in _boundary_matches(local_tracks, segment_index, segment_index + 1):
            if not _components_have_compatible_appearance(
                parents, local_tracks, before_index, after_index,
            ):
                continue
            _union(parents, before_index, after_index)
            match_evidence.append((before_index, after_index, evidence))
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(local_tracks)):
        components[_find_root(parents, index)].append(index)
    merged = []
    for indexes in components.values():
        evidence = [item for first, second, item in match_evidence if first in indexes and second in indexes]
        merged.append(_combine_component([local_tracks[index] for index in indexes], evidence))
    return merged


def _direction(detections: list[dict]) -> str:
    first_center = bbox_center(detections[0]["bbox"])
    last_center = bbox_center(detections[-1]["bbox"])
    dx = last_center[0] - first_center[0]
    dy = last_center[1] - first_center[1]
    average_width = np.mean([max(1, item["bbox"][2] - item["bbox"][0]) for item in detections])
    if abs(dx) < max(12, average_width * 0.2) and abs(dy) < max(12, average_width * 0.2):
        return "stationary"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down/toward" if dy > 0 else "up/away"


def _public_detection(detection: dict) -> dict:
    public_detection = {
        "frame": detection["frame"],
        "bbox": detection["bbox"],
        "confidence": round(float(detection["confidence"]), 4),
    }
    if isinstance(detection.get("pose_evidence"), dict):
        public_detection["pose_evidence"] = detection["pose_evidence"]
    return public_detection


def _summarize_track(raw_track: dict, track_id: str, fps: float) -> dict:
    detections = sorted(raw_track["detections"], key=lambda item: item["frame"])
    first, last = detections[0], detections[-1]
    first_center, last_center = bbox_center(first["bbox"]), bbox_center(last["bbox"])
    duration = max(1.0 / fps, (last["frame"] - first["frame"]) / fps)
    return {
        "id": track_id,
        "class_name": raw_track["class_name"],
        "class_id": raw_track["class_id"],
        "frames_seen": len(detections),
        "first_frame": first["frame"],
        "last_frame": last["frame"],
        "first_bbox": first["bbox"],
        "last_bbox": last["bbox"],
        "avg_confidence": round(float(np.mean([item["confidence"] for item in detections])), 3),
        "avg_area": round(float(np.mean([bbox_area(item["bbox"]) for item in detections])), 1),
        "direction": _direction(detections),
        "speed_px_sec": round(distance(first_center, last_center) / duration, 2),
        "continuity_confidence": raw_track.get("continuity_confidence"),
        "reidentification": raw_track.get("reidentification"),
        "detections": [_public_detection(item) for item in detections],
    }


def build_tracks(detections: list[dict], fps: float) -> list[dict]:
    local_tracks = [track for track in _local_tracks(detections) if len(track["detections"]) >= 2]
    merged_tracks = _merge_across_gaps(local_tracks)
    class_indexes: dict[str, int] = defaultdict(int)
    tracks: list[dict] = []
    for raw_track in merged_tracks:
        class_name = raw_track["class_name"]
        class_indexes[class_name] += 1
        tracks.append(_summarize_track(raw_track, f"{class_name}_{class_indexes[class_name]}", fps))
    tracks.sort(key=lambda track: (track["class_name"] != "person", -track["frames_seen"], -track["avg_area"]))
    return tracks


def _association_distance(person: dict, carried_object: dict) -> float | None:
    person_by_frame = {item["frame"]: item for item in person["detections"]}
    distances: list[float] = []
    for item in carried_object["detections"]:
        person_item = person_by_frame.get(item["frame"])
        if person_item:
            distances.append(distance(bbox_center(item["bbox"]), bbox_center(person_item["bbox"])))
    return float(np.median(distances)) if distances else None


def associate_objects(tracks: list[dict]) -> dict[str, list[str]]:
    people = [track for track in tracks if track["class_name"] == "person"]
    objects = [track for track in tracks if track["class_name"] in CARRIED_CLASSES]
    associations: dict[str, list[str]] = defaultdict(list)
    for carried_object in objects:
        candidates = []
        for person in people:
            separation = _association_distance(person, carried_object)
            threshold = max(80.0, bbox_diag(person["first_bbox"]) * 0.8)
            if separation is not None and separation <= threshold:
                candidates.append((separation, person["id"]))
        if candidates:
            associations[min(candidates)[1]].append(carried_object["id"])
    return dict(associations)


def _peak_class_count(detections: list[dict], class_name: str) -> int:
    counts = Counter(item["frame"] for item in detections if item["class_name"] == class_name)
    return max(counts.values(), default=0)


def summarize_scene(
    detections: list[dict],
    fps: float,
    frame_width: int,
    hidden_ranges: list[tuple[int, int]],
) -> dict:
    del frame_width
    tracks = build_tracks(detections, fps=fps)
    associations = associate_objects(tracks)
    people = [track for track in tracks if track["class_name"] == "person"]
    vehicles = [track for track in tracks if track["class_name"] in VEHICLE_CLASSES]
    carried = [track for track in tracks if track["class_name"] in CARRIED_CLASSES]
    for person in people:
        person["associated_objects"] = associations.get(person["id"], [])
    gap_entities = {
        str(index): [
            track["id"]
            for track in tracks
            if track["first_frame"] < hidden_start and track["last_frame"] > hidden_end
        ]
        for index, (hidden_start, hidden_end) in enumerate(hidden_ranges)
    }
    return {
        "people_count": len(people),
        "peak_people_visible": _peak_class_count(detections, "person"),
        "vehicle_count": len(vehicles),
        "carried_object_count": len(carried),
        "class_counts": dict(Counter(track["class_name"] for track in tracks)),
        "hidden_ranges": [{"start": start, "end": end} for start, end in hidden_ranges],
        "likely_gap_entities": gap_entities,
        "people": [_person_summary(person) for person in people],
        "vehicles": [_vehicle_summary(vehicle) for vehicle in vehicles],
        "tracks": tracks,
    }


def _person_summary(person: dict) -> dict:
    keys = [
        "id", "direction", "frames_seen", "continuity_confidence", "associated_objects",
    ]
    return {key: person[key] for key in keys}


def _vehicle_summary(vehicle: dict) -> dict:
    keys = ["id", "class_name", "direction", "frames_seen", "continuity_confidence"]
    return {key: vehicle[key] for key in keys}
