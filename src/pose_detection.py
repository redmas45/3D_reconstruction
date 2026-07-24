from collections.abc import Iterable


MINIMUM_POSE_ASSOCIATION_IOU = 0.30
EXPECTED_COCO_KEYPOINT_COUNT = 17


def pose_candidates(result: object, scale: float) -> list[dict]:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or boxes is None or len(boxes) == 0:
        return []
    coordinates = keypoints.xy.cpu().numpy()
    confidences = _keypoint_confidences(keypoints, len(coordinates))
    candidates = []
    for index, box in enumerate(boxes):
        if index >= len(coordinates):
            break
        bbox = [int(value / scale) for value in box.xyxy[0].cpu().numpy().tolist()]
        candidates.append({
            "bbox": bbox,
            "keypoints": _normalized_keypoints(
                coordinates[index],
                confidences[index],
                bbox,
                scale,
            ),
        })
    return candidates


def attach_pose_evidence(
    detections: list[dict],
    candidates: list[dict],
) -> list[dict]:
    available_candidates = set(range(len(candidates)))
    enriched = []
    for detection in detections:
        updated = dict(detection)
        if detection.get("class_name") == "person":
            match = _best_candidate(detection["bbox"], candidates, available_candidates)
            if match is not None:
                updated["pose_evidence"] = {
                    "schema_version": 1,
                    "format": "coco17_bbox_normalized",
                    "keypoints": candidates[match]["keypoints"],
                }
                available_candidates.remove(match)
        enriched.append(updated)
    return enriched


def _keypoint_confidences(keypoints: object, pose_count: int) -> list[list[float]]:
    confidence_tensor = getattr(keypoints, "conf", None)
    if confidence_tensor is None:
        return [[1.0] * EXPECTED_COCO_KEYPOINT_COUNT for _ in range(pose_count)]
    return confidence_tensor.cpu().numpy().tolist()


def _normalized_keypoints(
    coordinates: Iterable[Iterable[float]],
    confidences: Iterable[float],
    bbox: list[int],
    scale: float,
) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    return [
        [
            round(_clamp((float(point[0]) / scale - x1) / width), 5),
            round(_clamp((float(point[1]) / scale - y1) / height), 5),
            round(_clamp(float(confidence)), 5),
        ]
        for point, confidence in zip(coordinates, confidences)
    ]


def _best_candidate(
    detection_bbox: list[int],
    candidates: list[dict],
    available_candidates: set[int],
) -> int | None:
    ranked = sorted(
        (
            (_intersection_over_union(detection_bbox, candidates[index]["bbox"]), index)
            for index in available_candidates
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < MINIMUM_POSE_ASSOCIATION_IOU:
        return None
    return ranked[0][1]


def _intersection_over_union(first: list[int], second: list[int]) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
