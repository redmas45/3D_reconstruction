import math
from collections import Counter, defaultdict


CARRIED_CLASSES = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_diag(bbox):
    return math.sqrt(max(1, bbox[2] - bbox[0]) ** 2 + max(1, bbox[3] - bbox[1]) ** 2)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _make_track(det, track_idx):
    return {
        "internal_id": track_idx,
        "class_name": det["class_name"],
        "class_id": det["class_id"],
        "detections": [det],
        "last_frame": det["frame"],
        "last_bbox": det["bbox"],
    }


def _assign_tracks(detections: list) -> list:
    tracks = []
    for det in sorted(detections, key=lambda item: (item["frame"], item["class_name"], -item["confidence"])):
        det_center = bbox_center(det["bbox"])
        best = None
        for track in tracks:
            if track["class_name"] != det["class_name"]:
                continue
            if track["last_frame"] == det["frame"]:
                continue
            gap = det["frame"] - track["last_frame"]
            if gap < 0:
                continue
            last_center = bbox_center(track["last_bbox"])
            avg_diag = (bbox_diag(track["last_bbox"]) + bbox_diag(det["bbox"])) / 2.0
            max_distance = min(260.0, max(55.0, avg_diag * 1.25 + gap * 0.55))
            dist = distance(det_center, last_center)
            if dist <= max_distance:
                score = dist + gap * 0.08
                if best is None or score < best[0]:
                    best = (score, track)
        if best is None:
            tracks.append(_make_track(det, len(tracks) + 1))
        else:
            track = best[1]
            track["detections"].append(det)
            track["last_frame"] = det["frame"]
            track["last_bbox"] = det["bbox"]
    return tracks


def build_tracks(detections: list, fps: float, frame_width: int, hidden_range: tuple[int, int]) -> list:
    tracks = []
    assigned = _assign_tracks(detections)
    hidden_start, hidden_end = hidden_range
    for raw_track in assigned:
        items = sorted(raw_track["detections"], key=lambda item: item["frame"])
        if not items:
            continue
        first = items[0]
        last = items[-1]
        first_center = bbox_center(first["bbox"])
        last_center = bbox_center(last["bbox"])
        dx = last_center[0] - first_center[0]
        dy = last_center[1] - first_center[1]
        avg_width = sum(max(1, item["bbox"][2] - item["bbox"][0]) for item in items) / len(items)

        if abs(dx) < max(12, avg_width * 0.2) and abs(dy) < max(12, avg_width * 0.2):
            direction = "stationary"
        elif abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down/toward" if dy > 0 else "up/away"

        duration_sec = max(1.0 / max(1.0, fps), (last["frame"] - first["frame"]) / max(1.0, fps))
        speed_px_sec = distance(first_center, last_center) / duration_sec
        visible_before_gap = any(item["frame"] < hidden_start for item in items)
        visible_after_gap = any(item["frame"] > hidden_end for item in items)

        tracks.append(
            {
                "id": f"{first['class_name']}_{raw_track['internal_id']}",
                "class_name": first["class_name"],
                "class_id": first["class_id"],
                "frames_seen": len(items),
                "first_frame": first["frame"],
                "last_frame": last["frame"],
                "first_bbox": first["bbox"],
                "last_bbox": last["bbox"],
                "avg_confidence": round(sum(item["confidence"] for item in items) / len(items), 3),
                "avg_area": round(sum(bbox_area(item["bbox"]) for item in items) / len(items), 1),
                "direction": direction,
                "speed_px_sec": round(speed_px_sec, 2),
                "visible_before_gap": visible_before_gap,
                "visible_after_gap": visible_after_gap,
                "detections": items,
            }
        )

    tracks.sort(key=lambda track: (track["class_name"] != "person", -track["frames_seen"], -track["avg_area"]))
    return tracks


def associate_objects(tracks: list) -> dict:
    people = [track for track in tracks if track["class_name"] == "person"]
    objects = [track for track in tracks if track["class_name"] in CARRIED_CLASSES]
    associations = defaultdict(list)

    for obj in objects:
        obj_center = bbox_center(obj["first_bbox"])
        best = None
        for person in people:
            person_center = bbox_center(person["first_bbox"])
            threshold = max(80.0, bbox_diag(person["first_bbox"]) * 0.7)
            dist = distance(obj_center, person_center)
            if dist <= threshold and (best is None or dist < best[0]):
                best = (dist, person)
        if best:
            associations[best[1]["id"]].append(obj["id"])

    return dict(associations)


def summarize_scene(detections: list, fps: float, frame_width: int, hidden_range: tuple[int, int]) -> dict:
    tracks = build_tracks(detections, fps=fps, frame_width=frame_width, hidden_range=hidden_range)
    class_counts = Counter(track["class_name"] for track in tracks)
    associations = associate_objects(tracks)

    people = [track for track in tracks if track["class_name"] == "person"]
    vehicles = [track for track in tracks if track["class_name"] in VEHICLE_CLASSES]
    carried = [track for track in tracks if track["class_name"] in CARRIED_CLASSES]

    for person in people:
        person["associated_objects"] = associations.get(person["id"], [])

    likely_gap_entities = [
        track["id"]
        for track in tracks
        if track["visible_before_gap"] and track["visible_after_gap"]
    ]

    report = {
        "people_count": len(people),
        "vehicle_count": len(vehicles),
        "carried_object_count": len(carried),
        "class_counts": dict(class_counts),
        "hidden_range": {"start": hidden_range[0], "end": hidden_range[1]},
        "likely_gap_entities": likely_gap_entities,
        "people": [
            {
                "id": person["id"],
                "direction": person["direction"],
                "frames_seen": person["frames_seen"],
                "visible_before_gap": person["visible_before_gap"],
                "visible_after_gap": person["visible_after_gap"],
                "associated_objects": person.get("associated_objects", []),
            }
            for person in people
        ],
        "vehicles": [
            {
                "id": vehicle["id"],
                "class_name": vehicle["class_name"],
                "direction": vehicle["direction"],
                "frames_seen": vehicle["frames_seen"],
            }
            for vehicle in vehicles
        ],
        "tracks": tracks,
    }
    return report
