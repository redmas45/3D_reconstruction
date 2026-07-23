import math
import random


DEFAULT_MISSING_FRACTION = 0.25
DEFAULT_MIN_GAP_SECONDS = 5.0
DEFAULT_MAX_GAP_SECONDS = 7.0
DEFAULT_COMPACT_MIN_GAP_SECONDS = 1.0
DEFAULT_COMPACT_MAX_GAP_SECONDS = 3.0
DEFAULT_REVIEW_PROFILE_MIN_VIDEO_SECONDS = 60.0
DEFAULT_CONTEXT_SECONDS = 2.0
REVIEW_GAP_POLICY = "distributed_review_evidence_gaps"
COMPACT_GAP_POLICY = "distributed_compact_evidence_gaps"


def _seconds_to_frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def _gap_durations(target_frames: int, minimum_frames: int, maximum_frames: int, rng: random.Random) -> list[int]:
    average_frames = (minimum_frames + maximum_frames) / 2.0
    minimum_gap_count = max(1, math.ceil(target_frames / maximum_frames))
    maximum_gap_count = max(1, target_frames // minimum_frames)
    gap_count = max(minimum_gap_count, min(maximum_gap_count, int(round(target_frames / average_frames))))
    durations = [minimum_frames] * gap_count
    remaining = target_frames - sum(durations)
    while remaining:
        available = [index for index, duration in enumerate(durations) if duration < maximum_frames]
        if not available:
            raise ValueError("Could not allocate configured missing duration")
        durations[rng.choice(available)] += 1
        remaining -= 1
    rng.shuffle(durations)
    return durations


def _gap_profile(
    video_duration_seconds: float,
    review_minimum_seconds: float,
    review_maximum_seconds: float,
    compact_minimum_seconds: float,
    compact_maximum_seconds: float,
    review_profile_minimum_video_seconds: float,
) -> tuple[str, float, float]:
    if video_duration_seconds >= review_profile_minimum_video_seconds:
        return REVIEW_GAP_POLICY, review_minimum_seconds, review_maximum_seconds
    return COMPACT_GAP_POLICY, compact_minimum_seconds, compact_maximum_seconds


def _visible_durations(
    visible_frames: int,
    gap_count: int,
    minimum_context_frames: int,
    rng: random.Random,
) -> list[int]:
    minimum_total = minimum_context_frames * (gap_count + 1)
    if minimum_total > visible_frames:
        minimum_context_frames = max(1, visible_frames // (gap_count + 1))
        minimum_total = minimum_context_frames * (gap_count + 1)
    durations = [minimum_context_frames] * (gap_count + 1)
    remaining = visible_frames - minimum_total
    weights = [rng.random() + 0.2 for _ in durations]
    weight_total = sum(weights)
    allocated = 0
    for index, weight in enumerate(weights):
        addition = int(remaining * weight / weight_total)
        durations[index] += addition
        allocated += addition
    for index in rng.sample(range(len(durations)), remaining - allocated):
        durations[index] += 1
    return durations


def _timeline(visible_durations: list[int], gap_durations: list[int]) -> list[dict]:
    timeline: list[dict] = []
    next_frame = 0
    for visible_index, visible_frames in enumerate(visible_durations):
        timeline.append(_segment("visible", visible_index, next_frame, visible_frames))
        next_frame += visible_frames
        if visible_index < len(gap_durations):
            timeline.append(_segment("hidden", visible_index, next_frame, gap_durations[visible_index]))
            next_frame += gap_durations[visible_index]
    return timeline


def _segment(kind: str, index: int, start: int, frame_count: int) -> dict:
    return {
        "kind": kind,
        "index": index,
        "start": start,
        "end": start + frame_count - 1,
        "frame_count": frame_count,
    }


def choose_hidden_gaps(
    total_frames: int,
    fps: float,
    rng: random.Random,
    missing_fraction: float = DEFAULT_MISSING_FRACTION,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    compact_min_gap_seconds: float = DEFAULT_COMPACT_MIN_GAP_SECONDS,
    compact_max_gap_seconds: float = DEFAULT_COMPACT_MAX_GAP_SECONDS,
    review_profile_min_video_seconds: float = DEFAULT_REVIEW_PROFILE_MIN_VIDEO_SECONDS,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
) -> dict:
    if total_frames < 3 or fps <= 0:
        raise ValueError("Video and FPS must be valid")
    if not 0 < missing_fraction < 1:
        raise ValueError("Missing fraction must be between zero and one")
    if min_gap_seconds <= 0 or max_gap_seconds < min_gap_seconds:
        raise ValueError("Gap duration must satisfy 0 < min <= max")
    if compact_min_gap_seconds <= 0 or compact_max_gap_seconds < compact_min_gap_seconds:
        raise ValueError("Compact gap duration must satisfy 0 < min <= max")
    if review_profile_min_video_seconds <= 0:
        raise ValueError("Review profile minimum video duration must be positive")

    target_frames = max(1, int(round(total_frames * missing_fraction)))
    policy, selected_minimum_seconds, selected_maximum_seconds = _gap_profile(
        total_frames / fps,
        min_gap_seconds,
        max_gap_seconds,
        compact_min_gap_seconds,
        compact_max_gap_seconds,
        review_profile_min_video_seconds,
    )
    minimum_frames = _seconds_to_frames(selected_minimum_seconds, fps)
    maximum_frames = _seconds_to_frames(selected_maximum_seconds, fps)
    if target_frames < minimum_frames:
        minimum_video_seconds = selected_minimum_seconds / missing_fraction
        raise ValueError(
            "Video is too short for the configured gap policy; "
            f"use at least {minimum_video_seconds:.2f} seconds of footage"
        )
    gap_durations = _gap_durations(target_frames, minimum_frames, maximum_frames, rng)
    visible_frames = total_frames - target_frames
    if visible_frames < len(gap_durations) + 1:
        raise ValueError("Video does not contain enough visible evidence around the configured gaps")
    context_frames = _seconds_to_frames(context_seconds, fps)
    visible_durations = _visible_durations(visible_frames, len(gap_durations), context_frames, rng)
    timeline = _timeline(visible_durations, gap_durations)
    hidden_ranges = [(item["start"], item["end"]) for item in timeline if item["kind"] == "hidden"]
    visible_ranges = [(item["start"], item["end"]) for item in timeline if item["kind"] == "visible"]
    return {
        "policy": policy,
        "profile": "review" if policy == REVIEW_GAP_POLICY else "compact",
        "missing_fraction_target": missing_fraction,
        "missing_fraction_actual": round(target_frames / total_frames, 6),
        "missing_frames": target_frames,
        "gap_count": len(hidden_ranges),
        "timeline": timeline,
        "hidden_ranges": hidden_ranges,
        "visible_ranges": visible_ranges,
        "gap_durations_seconds": [round(duration / fps, 3) for duration in gap_durations],
        "selected_gap_bounds_seconds": {
            "minimum": selected_minimum_seconds,
            "maximum": selected_maximum_seconds,
        },
    }
