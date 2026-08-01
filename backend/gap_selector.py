"""Chooses which stretches of a video to hide, and where the evidence for them lives.

A gap is only reconstructable from the footage around it, and "around it" has to mean
the same scene. A gap placed across a cut has its before-context in one camera setup and
its after-context in another, so the trajectory drawn between them crosses a scene change
and the background recovered for it is a blend of two places. Every gap is therefore
placed wholly inside one shot, with visible context on both sides *of that shot*.

Videos with no cuts are the ordinary case and behave exactly as before: one shot spanning
the file, gaps distributed across it.

Where the shot structure cannot host the configured missing fraction — a montage of short
clips has far less usable room than its frame count suggests — the shortfall is reported
rather than silently absorbed by relaxing the constraint.
"""

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


def _normalize_shots(shots, total_frames: int) -> list[tuple[int, int]]:
    """A video with no reported shot structure is one shot covering all of it."""
    if not shots:
        return [(0, total_frames - 1)]
    spans = [(int(start), int(end)) for start, end in shots]
    for start, end in spans:
        if start < 0 or end >= total_frames or start > end:
            raise ValueError(
                f"Shot ({start}, {end}) does not fit a video of {total_frames} frames"
            )
    return sorted(spans)


def _free_frames(span: tuple[int, int], assigned: list[int], context_frames: int) -> int:
    """Room left in a shot for one more gap.

    A shot holding `k` gaps needs `k + 1` stretches of visible context — one before the
    first, one after the last, and one between each neighbouring pair — so adding a gap
    costs its own length plus one more context stretch.
    """
    length = span[1] - span[0] + 1
    return length - sum(assigned) - (len(assigned) + 2) * context_frames


def _assign_gaps_to_shots(
    spans: list[tuple[int, int]],
    gap_durations: list[int],
    context_frames: int,
) -> tuple[dict[int, list[int]], list[int]]:
    """Give every gap a shot roomy enough to hold it, longest gap first.

    Longest first because a long gap fits in fewer shots; placing the easy short ones
    first would fill the only shots the long ones could have used. Each gap goes to the
    shot with the most room left, which spreads them across the video instead of packing
    them into whichever shot happens to come first.
    """
    assigned: dict[int, list[int]] = {index: [] for index in range(len(spans))}
    unplaced: list[int] = []
    for duration in sorted(gap_durations, reverse=True):
        best_index, best_slack = None, -1
        for index, span in enumerate(spans):
            slack = _free_frames(span, assigned[index], context_frames) - duration
            if slack >= 0 and slack > best_slack:
                best_index, best_slack = index, slack
        if best_index is None:
            unplaced.append(duration)
            continue
        assigned[best_index].append(duration)
    return assigned, unplaced


def _spread(total: int, parts: int, rng: random.Random) -> list[int]:
    """Split `total` spare frames across `parts` context stretches, unevenly."""
    if parts <= 0:
        return []
    if total <= 0:
        return [0] * parts
    weights = [rng.random() + 0.2 for _ in range(parts)]
    weight_total = sum(weights)
    shares = [int(total * weight / weight_total) for weight in weights]
    for index in rng.sample(range(parts), total - sum(shares)):
        shares[index] += 1
    return shares


def _place_within_shot(
    span: tuple[int, int], durations: list[int], context_frames: int, rng: random.Random,
) -> list[tuple[int, int]]:
    """Lay a shot's gaps out along it, keeping every context stretch at least the
    configured minimum and scattering the remaining slack between them."""
    if not durations:
        return []
    ordered = list(durations)
    rng.shuffle(ordered)
    length = span[1] - span[0] + 1
    slack = length - sum(ordered) - (len(ordered) + 1) * context_frames
    extra = _spread(max(0, slack), len(ordered) + 1, rng)
    ranges: list[tuple[int, int]] = []
    cursor = span[0]
    for index, duration in enumerate(ordered):
        cursor += context_frames + extra[index]
        ranges.append((cursor, cursor + duration - 1))
        cursor += duration
    return ranges


def _segment(kind: str, index: int, start: int, frame_count: int) -> dict:
    return {
        "kind": kind,
        "index": index,
        "start": start,
        "end": start + frame_count - 1,
        "frame_count": frame_count,
    }


def _timeline_from_hidden(hidden_ranges: list[tuple[int, int]], total_frames: int) -> list[dict]:
    """Fill in the visible segments around the gaps so the timeline covers every frame.

    Transition frames between shots are never hidden and so always land in a visible
    segment, which is what lets the stitched output stay frame-for-frame complete.
    """
    timeline: list[dict] = []
    cursor = 0
    for hidden_index, (start, end) in enumerate(hidden_ranges):
        if start <= cursor - 1:
            raise ValueError(f"Gap {hidden_index} overlaps the previous segment")
        timeline.append(_segment("visible", len(timeline) // 2, cursor, start - cursor))
        timeline.append(_segment("hidden", hidden_index, start, end - start + 1))
        cursor = end + 1
    timeline.append(_segment("visible", len(timeline) // 2, cursor, total_frames - cursor))
    for segment in timeline:
        if segment["frame_count"] <= 0:
            raise ValueError(
                "Gap placement produced an empty visible segment; every gap must keep "
                "context on both sides"
            )
    return timeline


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
    shots=None,
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
    if total_frames - target_frames < len(gap_durations) + 1:
        raise ValueError("Video does not contain enough visible evidence around the configured gaps")

    spans = _normalize_shots(shots, total_frames)
    context_frames = _seconds_to_frames(context_seconds, fps)
    assigned, unplaced = _assign_gaps_to_shots(spans, gap_durations, context_frames)
    if all(not durations for durations in assigned.values()):
        raise ValueError(
            f"No shot is long enough to hold a {selected_minimum_seconds:.2f} second gap "
            f"with {context_seconds:.2f} seconds of context on each side; the longest is "
            f"{max(end - start + 1 for start, end in spans) / fps:.2f} seconds"
        )
    hidden_ranges = sorted(
        placement
        for index, durations in assigned.items()
        for placement in _place_within_shot(spans[index], durations, context_frames, rng)
    )
    timeline = _timeline_from_hidden(hidden_ranges, total_frames)
    placed_frames = sum(end - start + 1 for start, end in hidden_ranges)
    return {
        "policy": policy,
        "profile": "review" if policy == REVIEW_GAP_POLICY else "compact",
        "missing_fraction_target": missing_fraction,
        "missing_fraction_actual": round(placed_frames / total_frames, 6),
        "missing_frames": placed_frames,
        "gap_count": len(hidden_ranges),
        "timeline": timeline,
        "hidden_ranges": hidden_ranges,
        "visible_ranges": [
            (item["start"], item["end"]) for item in timeline if item["kind"] == "visible"
        ],
        "gap_durations_seconds": [
            round((end - start + 1) / fps, 3) for start, end in hidden_ranges
        ],
        "selected_gap_bounds_seconds": {
            "minimum": selected_minimum_seconds,
            "maximum": selected_maximum_seconds,
        },
        "shot_placement": {
            "shot_count": len(spans),
            "shots_hosting_gaps": sum(1 for durations in assigned.values() if durations),
            "requested_frames": target_frames,
            "unplaced_gap_count": len(unplaced),
            "unplaced_frames": sum(unplaced),
            "gaps_per_shot": {
                str(index): len(durations) for index, durations in sorted(assigned.items())
            },
        },
    }
