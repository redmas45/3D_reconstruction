"""Ranks reconstruction candidates using measured evidence and hard constraints.

The language model may explain a choice, but it must not invent geometry.  This
module is the deterministic safety layer between the bounded candidate library and
the renderer.  Every score is derived from visible boundary measurements carried by
the entity contract; missing measurements reduce confidence rather than being guessed.
"""

import math


MAXIMUM_SCORE = 1.0
MINIMUM_RENDERABLE_SCORE = 0.30
SAFE_SELECTION_MARGIN = 0.12
DEFAULT_POST_RESIDUAL_SCALE_METRES = 2.0
ANGLE_SUPPORT_LIMIT_DEGREES = 60.0
LOW_SPEED_METRES_PER_SECOND = 0.20
REFERENCE_FPS = 30.0
MINIMUM_TIME_STEP_SECONDS = 1.0 / REFERENCE_FPS


def score_hypothesis(
    entity: dict,
    hypothesis_type: str,
    path: list[dict],
    speed_meters_per_second: float,
) -> dict:
    """Return a bounded score and an auditable component breakdown."""
    boundary = entity.get("boundary_evidence", {})
    lifecycle = str(entity.get("lifecycle", "uncertain"))
    confidence = _clamp(entity.get("confidence", 0.0))
    continuity_value = entity.get("continuity_confidence")
    continuity = _clamp(confidence if continuity_value is None else continuity_value)
    disagreement = float(boundary.get("heading_disagreement_degrees", 0.0))
    residual = _endpoint_residual(boundary, path)
    components = {
        "boundary_confidence": confidence,
        "identity_continuity": continuity,
        "lifecycle_fit": _lifecycle_fit(lifecycle, hypothesis_type),
        "heading_fit": _heading_fit(disagreement, hypothesis_type),
        "endpoint_fit": _endpoint_fit(residual, entity),
        "kinematic_fit": _kinematic_fit(entity, path, speed_meters_per_second),
    }
    total = (
        0.24 * components["boundary_confidence"]
        + 0.16 * components["identity_continuity"]
        + 0.18 * components["lifecycle_fit"]
        + 0.14 * components["heading_fit"]
        + 0.16 * components["endpoint_fit"]
        + 0.12 * components["kinematic_fit"]
    )
    hard_constraints = _hard_constraints(entity, hypothesis_type, disagreement)
    eligible = not hard_constraints and total >= MINIMUM_RENDERABLE_SCORE
    return {
        "selection_score": round(_clamp(total), 4),
        "score_components": {key: round(_clamp(value), 4) for key, value in components.items()},
        "hard_constraints": hard_constraints,
        "render_eligibility": eligible,
        "endpoint_residual_meters": round(residual, 4) if residual is not None else None,
        "selection_policy": "deterministic_visible_evidence_ranker_v1",
    }


def choose_safe_hypothesis(
    hypotheses: list[dict],
    requested_id: str,
) -> tuple[dict, bool]:
    """Keep an LLM choice only when it is safe and close to the measured best."""
    if not hypotheses:
        raise ValueError("At least one hypothesis is required")
    eligible = [item for item in hypotheses if item.get("render_eligibility", True)]
    candidates = eligible or hypotheses
    best = max(candidates, key=lambda item: float(item.get("selection_score", item.get("prior", 0.0))))
    requested = next((item for item in candidates if item.get("id") == requested_id), None)
    if requested is None:
        return best, True
    requested_score = float(requested.get("selection_score", requested.get("prior", 0.0)))
    best_score = float(best.get("selection_score", best.get("prior", 0.0)))
    overridden = requested_score + SAFE_SELECTION_MARGIN < best_score
    return (best if overridden else requested), overridden


def _lifecycle_fit(lifecycle: str, hypothesis_type: str) -> float:
    if hypothesis_type == "identity_unresolved_proxy":
        return 1.0 if lifecycle == "uncertain" else 0.62
    if hypothesis_type == "remain_occluded":
        return 1.0 if lifecycle == "uncertain" else 0.35
    if hypothesis_type == "exit_visible_region":
        return 1.0 if lifecycle == "exits" else 0.10
    if hypothesis_type == "enter_visible_region":
        return 1.0 if lifecycle == "enters" else 0.10
    if lifecycle == "continuous":
        return 0.92 if hypothesis_type == "boundary_consistent_motion" else 0.82
    return 0.58


def _heading_fit(disagreement: float, hypothesis_type: str) -> float:
    if hypothesis_type == "follow_supported_turn":
        return _clamp(1.0 - abs(disagreement - 25.0) / 60.0)
    if hypothesis_type in {"hold_position", "remain_occluded", "identity_unresolved_proxy"}:
        return _clamp(disagreement / 90.0)
    return _clamp(1.0 - disagreement / 120.0)


def _endpoint_fit(residual: float | None, entity: dict) -> float:
    if residual is None:
        return 0.55
    uncertainty = float(entity.get("uncertainty", {}).get("position_radius_meters", 0.0))
    scale = max(DEFAULT_POST_RESIDUAL_SCALE_METRES, uncertainty)
    return math.exp(-max(0.0, residual) / scale)


def _kinematic_fit(entity: dict, path: list[dict], speed: float) -> float:
    limits = entity.get("kinematics", {})
    maximum_speed = max(0.01, float(limits.get("maximum_speed_meters_per_second", 8.0)))
    maximum_acceleration = max(
        0.01, float(limits.get("maximum_acceleration_meters_per_second_squared", 4.0)),
    )
    maximum_turn = max(0.01, float(limits.get("maximum_turn_rate_degrees_per_second", 45.0)))
    speed_fit = min(1.0, maximum_speed / max(maximum_speed, abs(float(speed))))
    duration_seconds = max(MINIMUM_TIME_STEP_SECONDS, float(entity.get("kinematics", {}).get("duration_seconds", 1.0)))
    segment_speeds = _segment_speeds(path, duration_seconds)
    acceleration = _maximum_acceleration(segment_speeds, path, duration_seconds)
    turn_rate = _maximum_turn_rate(path, duration_seconds)
    acceleration_fit = min(1.0, maximum_acceleration / max(maximum_acceleration, acceleration))
    turn_fit = min(1.0, maximum_turn / max(maximum_turn, turn_rate))
    return 0.50 * speed_fit + 0.30 * acceleration_fit + 0.20 * turn_fit


def _hard_constraints(entity: dict, hypothesis_type: str, disagreement: float) -> list[str]:
    lifecycle = str(entity.get("lifecycle", "uncertain"))
    constraints: list[str] = []
    if hypothesis_type == "exit_visible_region" and lifecycle != "exits":
        constraints.append("exit_requires_exit_lifecycle")
    if hypothesis_type == "enter_visible_region" and lifecycle != "enters":
        constraints.append("entry_requires_entry_lifecycle")
    if hypothesis_type == "follow_supported_turn" and disagreement > ANGLE_SUPPORT_LIMIT_DEGREES:
        constraints.append("turn_heading_disagreement_exceeds_support")
    if hypothesis_type == "hold_position" and lifecycle == "continuous":
        speed = float(entity.get("animation", {}).get("speed_meters_per_second", 0.0))
        if speed > LOW_SPEED_METRES_PER_SECOND and disagreement < 45.0:
            constraints.append("hold_conflicts_with_measured_motion")
    if hypothesis_type == "remain_occluded" and lifecycle != "uncertain":
        constraints.append("occlusion_not_supported_by_lifecycle")
    return constraints


def _endpoint_residual(boundary: dict, path: list[dict]) -> float | None:
    observed = boundary.get("post_gap_world")
    if not isinstance(observed, list) or len(observed) != 3 or not path:
        return None
    endpoint = path[-1].get("world")
    if not isinstance(endpoint, list) or len(endpoint) != 3:
        return None
    return math.sqrt(sum((float(endpoint[index]) - float(observed[index])) ** 2 for index in range(3)))


def _segment_speeds(path: list[dict], duration_seconds: float) -> list[float]:
    speeds: list[float] = []
    total_frames = max(1, int(path[-1].get("frame", 0)) - int(path[0].get("frame", 0))) if path else 1
    seconds_per_frame = duration_seconds / total_frames
    for first, second in zip(path, path[1:]):
        first_world, second_world = first.get("world", []), second.get("world", [])
        duration = max(MINIMUM_TIME_STEP_SECONDS, (int(second.get("frame", 0)) - int(first.get("frame", 0))) * seconds_per_frame)
        if len(first_world) != 3 or len(second_world) != 3:
            continue
        distance = math.sqrt(sum((float(second_world[index]) - float(first_world[index])) ** 2 for index in range(3)))
        speeds.append(distance / duration)
    return speeds or [0.0]


def _maximum_acceleration(speeds: list[float], path: list[dict], duration_seconds: float) -> float:
    if len(speeds) < 2:
        return 0.0
    total_frames = max(1, int(path[-1].get("frame", 0)) - int(path[0].get("frame", 0)))
    seconds_per_frame = duration_seconds / total_frames
    values = []
    for first, second, speed_first, speed_second in zip(path, path[1:], speeds, speeds[1:]):
        duration = max(MINIMUM_TIME_STEP_SECONDS, (int(second.get("frame", 0)) - int(first.get("frame", 0))) * seconds_per_frame)
        values.append(abs(speed_second - speed_first) / duration)
    return max(values, default=0.0)


def _maximum_turn_rate(path: list[dict], duration_seconds: float) -> float:
    headings: list[tuple[float, float]] = []
    for first, second in zip(path, path[1:]):
        a, b = first.get("world", []), second.get("world", [])
        if len(a) == 3 and len(b) == 3:
            headings.append((float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
    total_frames = max(1, int(path[-1].get("frame", 0)) - int(path[0].get("frame", 0)))
    seconds_per_frame = duration_seconds / total_frames
    rates: list[float] = []
    for index, (first, second) in enumerate(zip(headings, headings[1:])):
        first_angle = math.degrees(math.atan2(first[1], first[0]))
        second_angle = math.degrees(math.atan2(second[1], second[0]))
        difference = abs((second_angle - first_angle + 180.0) % 360.0 - 180.0)
        frame_delta = max(1, int(path[index + 2].get("frame", 0)) - int(path[index + 1].get("frame", 0)))
        rates.append(difference / max(MINIMUM_TIME_STEP_SECONDS, frame_delta * seconds_per_frame))
    return max(rates, default=0.0)


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(MAXIMUM_SCORE, float(value)))
    except (TypeError, ValueError):
        return 0.0
