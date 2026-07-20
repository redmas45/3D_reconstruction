import math

from domain.camera_calibration import image_point_to_world


BOUNDARY_WINDOW_SECONDS = 0.35
MINIMUM_BOUNDARY_CONFIDENCE = 0.35
SUPPORTED_HEADING_DISAGREEMENT = 30.0
PLAUSIBLE_TURN_DISAGREEMENT = 60.0
AMBIGUOUS_HEADING_DISAGREEMENT = 100.0
SUPPORTED_POSITION_RESIDUAL_METERS = 1.5
PLAUSIBLE_POSITION_RESIDUAL_METERS = 3.5
AMBIGUOUS_POSITION_RESIDUAL_METERS = 7.0


def build_entity_prediction(
    track: dict,
    hidden_range: tuple[int, int],
    fps: float,
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> dict | None:
    hidden_start, hidden_end = hidden_range
    before = _boundary_detections(track, hidden_start, "before", fps)
    after = _boundary_detections(track, hidden_end, "after", fps)
    lifecycle = _lifecycle(before, after)
    if lifecycle is None:
        return None
    image_waypoints = _predicted_image_waypoints(before, after, hidden_range, lifecycle)
    world_waypoints = _world_waypoints(image_waypoints, hidden_range, frame_size, camera_contract)
    boundary_evidence = _boundary_evidence(before, after, world_waypoints, frame_size, camera_contract)
    lifecycle = _resolved_lifecycle(lifecycle, boundary_evidence)
    confidence = _prediction_confidence(track, before, after, boundary_evidence, lifecycle)
    return {
        "lifecycle": lifecycle,
        "confidence": confidence,
        "boundary_evidence": boundary_evidence,
        "path_prediction": {
            "method": "centripetal_catmull_rom",
            "constraint_mode": "forward_prediction" if before else "reverse_entry_prediction",
            "post_gap_observation_role": "soft_consistency_check",
            "waypoints": world_waypoints,
        },
        "speed_meters_per_second": _path_speed(world_waypoints, hidden_range, fps),
    }


def heading_confidence_multiplier(disagreement_degrees: float) -> float:
    if disagreement_degrees <= SUPPORTED_HEADING_DISAGREEMENT:
        return 1.0
    if disagreement_degrees <= PLAUSIBLE_TURN_DISAGREEMENT:
        return 0.82
    if disagreement_degrees <= AMBIGUOUS_HEADING_DISAGREEMENT:
        return 0.58
    return 0.35


def fidelity_tier(entity_confidence: float, calibration_confidence: float) -> str:
    effective_confidence = entity_confidence
    if calibration_confidence < 0.50:
        effective_confidence -= 0.25
    if effective_confidence >= 0.75:
        return "supported"
    if effective_confidence >= 0.50:
        return "plausible"
    return "weak"


def _boundary_detections(track: dict, boundary_frame: int, side: str, fps: float) -> list[dict]:
    window_frames = max(2, round(BOUNDARY_WINDOW_SECONDS * fps))
    detections = track.get("detections", [])
    if side == "before":
        eligible = [item for item in detections if boundary_frame - window_frames <= item["frame"] < boundary_frame]
        return eligible[-3:]
    eligible = [item for item in detections if boundary_frame < item["frame"] <= boundary_frame + window_frames]
    return eligible[:3]


def _lifecycle(before: list[dict], after: list[dict]) -> str | None:
    valid_before = bool(before) and float(before[-1].get("confidence", 0.0)) >= MINIMUM_BOUNDARY_CONFIDENCE
    valid_after = bool(after) and float(after[0].get("confidence", 0.0)) >= MINIMUM_BOUNDARY_CONFIDENCE
    if valid_before and valid_after:
        return "continuous"
    if valid_before:
        return "exits"
    if valid_after:
        return "enters"
    return None


def _predicted_image_waypoints(
    before: list[dict],
    after: list[dict],
    hidden_range: tuple[int, int],
    lifecycle: str,
) -> list[tuple[float, float]]:
    hidden_start, hidden_end = hidden_range
    midpoint_frame = (hidden_start + hidden_end) // 2
    if before:
        anchor = _ground_point(before[-1])
        velocity = _image_velocity(before)
        return [
            _advance(anchor, velocity, hidden_start - before[-1]["frame"]),
            _advance(anchor, velocity, midpoint_frame - before[-1]["frame"]),
            _advance(anchor, velocity, hidden_end - before[-1]["frame"]),
        ]
    anchor = _ground_point(after[0])
    velocity = _image_velocity(after)
    if lifecycle == "enters" and len(after) < 2:
        velocity = (0.0, 0.0)
    return [
        _advance(anchor, velocity, hidden_start - after[0]["frame"]),
        _advance(anchor, velocity, midpoint_frame - after[0]["frame"]),
        _advance(anchor, velocity, hidden_end - after[0]["frame"]),
    ]


def _world_waypoints(
    points: list[tuple[float, float]],
    hidden_range: tuple[int, int],
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> list[dict]:
    hidden_start, hidden_end = hidden_range
    frames = [hidden_start, (hidden_start + hidden_end) // 2, hidden_end]
    roles = ["start", "inferred_midpoint", "predicted_end"]
    width, height = frame_size
    return [
        {"role": role, "frame": frame, "world": image_point_to_world(point[0], point[1], width, height, camera_contract)}
        for role, frame, point in zip(roles, frames, points)
    ]


def _boundary_evidence(
    before: list[dict],
    after: list[dict],
    world_waypoints: list[dict],
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> dict:
    pre_heading = _heading(_image_velocity(before)) if len(before) >= 2 else 0.0
    post_heading = _heading(_image_velocity(after)) if len(after) >= 2 else pre_heading
    disagreement = _angle_difference(pre_heading, post_heading)
    residual = None
    if after:
        post_point = _ground_point(after[0])
        post_world = image_point_to_world(post_point[0], post_point[1], frame_size[0], frame_size[1], camera_contract)
        residual = _distance(world_waypoints[-1]["world"], post_world)
    residual_multiplier = position_residual_confidence_multiplier(residual)
    return {
        "pre_gap_heading_degrees": round(pre_heading, 3),
        "post_gap_heading_degrees": round(post_heading, 3),
        "heading_disagreement_degrees": round(disagreement, 3),
        "post_gap_position_residual_meters": round(residual, 4) if residual is not None else None,
        "post_gap_residual_confidence_multiplier": residual_multiplier,
        "before_frame": before[-1]["frame"] if before else None,
        "after_frame": after[0]["frame"] if after else None,
    }


def _prediction_confidence(
    track: dict,
    before: list[dict],
    after: list[dict],
    boundary_evidence: dict,
    lifecycle: str,
) -> float:
    evidence_scores = [float(item.get("confidence", 0.0)) for item in before[-1:] + after[:1]]
    evidence_score = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
    continuity_score = float(track.get("continuity_confidence", track.get("avg_confidence", 0.0)))
    lifecycle_factor = {"continuous": 1.0, "enters": 0.75, "exits": 0.75, "uncertain": 0.50}[lifecycle]
    heading_factor = heading_confidence_multiplier(boundary_evidence["heading_disagreement_degrees"])
    residual_factor = float(boundary_evidence["post_gap_residual_confidence_multiplier"])
    confidence = (0.55 * evidence_score + 0.45 * continuity_score)
    return round(max(0.0, min(1.0, confidence * lifecycle_factor * heading_factor * residual_factor)), 4)


def _resolved_lifecycle(lifecycle: str, boundary_evidence: dict) -> str:
    heading_disagreement = float(boundary_evidence["heading_disagreement_degrees"])
    if lifecycle == "continuous" and heading_disagreement > AMBIGUOUS_HEADING_DISAGREEMENT:
        return "uncertain"
    return lifecycle


def position_residual_confidence_multiplier(residual_meters: float | None) -> float:
    if residual_meters is None or residual_meters <= SUPPORTED_POSITION_RESIDUAL_METERS:
        return 1.0
    if residual_meters <= PLAUSIBLE_POSITION_RESIDUAL_METERS:
        return 0.82
    if residual_meters <= AMBIGUOUS_POSITION_RESIDUAL_METERS:
        return 0.58
    return 0.30


def _image_velocity(detections: list[dict]) -> tuple[float, float]:
    if len(detections) < 2:
        return 0.0, 0.0
    first, last = detections[0], detections[-1]
    frame_delta = max(1, last["frame"] - first["frame"])
    first_point, last_point = _ground_point(first), _ground_point(last)
    return (last_point[0] - first_point[0]) / frame_delta, (last_point[1] - first_point[1]) / frame_delta


def _ground_point(detection: dict) -> tuple[float, float]:
    x1, _, x2, y2 = detection["bbox"]
    return (x1 + x2) / 2.0, float(y2)


def _advance(point: tuple[float, float], velocity: tuple[float, float], frames: int) -> tuple[float, float]:
    return point[0] + velocity[0] * frames, point[1] + velocity[1] * frames


def _heading(velocity: tuple[float, float]) -> float:
    return math.degrees(math.atan2(velocity[1], velocity[0])) % 360.0


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _path_speed(waypoints: list[dict], hidden_range: tuple[int, int], fps: float) -> float:
    duration = max(1.0 / fps, (hidden_range[1] - hidden_range[0] + 1) / fps)
    distance = _distance(waypoints[0]["world"], waypoints[-1]["world"])
    return round(distance / duration, 4)
