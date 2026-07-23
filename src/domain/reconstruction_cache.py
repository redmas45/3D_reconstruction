"""Defines the persisted selection and detection cache contracts."""

import math


def source_video_contract(video_metadata: dict) -> dict:
    return {
        "width": int(video_metadata["width"]),
        "height": int(video_metadata["height"]),
        "frames": int(video_metadata["frames"]),
        "fps": float(video_metadata["fps"]),
        "sha256": str(video_metadata.get("sha256", "")),
    }


def gap_cache_configuration(gap_configuration: dict) -> dict:
    return {
        "missing_fraction": float(gap_configuration.get("missing_fraction", 0.25)),
        "min_seconds": float(gap_configuration.get("min_seconds", 5.0)),
        "max_seconds": float(gap_configuration.get("max_seconds", 7.0)),
        "compact_min_seconds": float(gap_configuration.get("compact_min_seconds", 1.0)),
        "compact_max_seconds": float(gap_configuration.get("compact_max_seconds", 3.0)),
        "review_profile_min_video_seconds": float(
            gap_configuration.get("review_profile_min_video_seconds", 60.0)
        ),
        "context_seconds": float(gap_configuration.get("context_seconds", 2.0)),
    }


def selection_cache_is_compatible(
    selection: object,
    video_metadata: dict,
    gap_configuration: dict,
) -> bool:
    if not isinstance(selection, dict) or not _selection_timeline_is_valid(
        selection, int(video_metadata["frames"]),
    ):
        return False
    return all((
        selection.get("policy") in {
            "distributed_review_evidence_gaps",
            "distributed_compact_evidence_gaps",
        },
        selection.get("source_video_contract") == source_video_contract(video_metadata),
        selection.get("gap_configuration") == gap_cache_configuration(gap_configuration),
    ))


def _selection_timeline_is_valid(selection: dict, total_frames: int) -> bool:
    ranges = _timeline_ranges(selection.get("timeline"), total_frames)
    if ranges is None:
        return False
    hidden_ranges, visible_ranges = ranges
    return all((
        _normalize_ranges(selection.get("hidden_ranges")) == hidden_ranges,
        _normalize_ranges(selection.get("visible_ranges")) == visible_ranges,
        selection.get("gap_count") == len(hidden_ranges),
        selection.get("missing_frames") == sum(end - start + 1 for start, end in hidden_ranges),
    ))


def _timeline_ranges(
    timeline: object,
    total_frames: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if not isinstance(timeline, list) or not timeline:
        return None
    ranges: dict[str, list[tuple[int, int]]] = {"hidden": [], "visible": []}
    next_frame = 0
    for item_index, segment in enumerate(timeline):
        expected_kind = "visible" if item_index % 2 == 0 else "hidden"
        if not _timeline_segment_is_valid(segment, expected_kind, len(ranges[expected_kind]), next_frame):
            return None
        start_frame, end_frame = int(segment["start"]), int(segment["end"])
        ranges[expected_kind].append((start_frame, end_frame))
        next_frame = end_frame + 1
    if next_frame != total_frames or timeline[-1].get("kind") != "visible":
        return None
    return ranges["hidden"], ranges["visible"]


def _timeline_segment_is_valid(segment: object, kind: str, index: int, start_frame: int) -> bool:
    if not isinstance(segment, dict):
        return False
    end_frame = segment.get("end")
    return all((
        segment.get("kind") == kind,
        type(segment.get("index")) is int and segment.get("index") == index,
        type(segment.get("start")) is int and segment.get("start") == start_frame,
        type(end_frame) is int and end_frame >= start_frame,
        segment.get("frame_count") == (end_frame - start_frame + 1 if type(end_frame) is int else None),
    ))


def _normalize_ranges(value: object) -> list[tuple[int, int]] | None:
    if not isinstance(value, list):
        return None
    normalized_ranges: list[tuple[int, int]] = []
    for frame_range in value:
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            return None
        if any(type(frame_index) is not int for frame_index in frame_range):
            return None
        normalized_ranges.append((frame_range[0], frame_range[1]))
    return normalized_ranges


def cached_detections_are_valid(payload: object, visible_ranges: object) -> bool:
    if not isinstance(payload, list):
        return False
    normalized_ranges = _normalize_ranges(visible_ranges)
    if normalized_ranges is None:
        return False
    return all(_cached_detection_is_valid(detection, normalized_ranges) for detection in payload)


def _cached_detection_is_valid(detection: object, visible_ranges: list[tuple[int, int]]) -> bool:
    if not isinstance(detection, dict):
        return False
    frame_index = detection.get("frame")
    segment_index = detection.get("segment_index")
    if type(frame_index) is not int or type(segment_index) is not int:
        return False
    if segment_index < 0 or segment_index >= len(visible_ranges):
        return False
    start_frame, end_frame = visible_ranges[segment_index]
    return all((
        start_frame <= frame_index <= end_frame,
        _finite_number_list(detection.get("bbox"), expected_length=4),
        _finite_number_list(detection.get("appearance")),
        type(detection.get("class_id")) is int,
        type(detection.get("source_track_id")) is int,
        isinstance(detection.get("class_name"), str),
        _is_finite_number(detection.get("confidence")),
    ))


def _finite_number_list(value: object, expected_length: int | None = None) -> bool:
    if not isinstance(value, list) or (expected_length is not None and len(value) != expected_length):
        return False
    return all(_is_finite_number(item) for item in value)


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))
