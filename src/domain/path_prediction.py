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
UNMATCHED_CONTINUITY_FACTOR = 0.65
LONG_GAP_CONFIDENCE_START_SECONDS = 3.0
LONG_GAP_CONFIDENCE_DECAY_PER_SECOND = 0.08
MINIMUM_DURATION_CONFIDENCE_MULTIPLIER = 0.55
WAYPOINT_INTERVAL_SECONDS = 1.25
MAXIMUM_PATH_WAYPOINTS = 8
DEFAULT_MAXIMUM_SPEED_METERS_PER_SECOND = 8.0
MAXIMUM_SPEED_METERS_PER_SECOND = {
    "person": 3.0,
    "bicycle": 12.0,
    "motorcycle": 30.0,
    "car": 35.0,
    "truck": 28.0,
    "bus": 25.0,
}
MAXIMUM_ACCELERATION_METERS_PER_SECOND_SQUARED = {
    "person": 3.0,
    "bicycle": 4.0,
    "motorcycle": 7.0,
    "car": 6.0,
    "truck": 4.0,
    "bus": 3.0,
}
MAXIMUM_TURN_RATE_DEGREES_PER_SECOND = {
    "person": 120.0,
    "bicycle": 70.0,
    "motorcycle": 55.0,
    "car": 45.0,
    "truck": 30.0,
    "bus": 25.0,
}


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
    waypoint_frames = _waypoint_frames(hidden_range, fps)
    image_points = _predicted_image_points(before, after, waypoint_frames, lifecycle)
    world_waypoints = _world_waypoints(
        image_points, waypoint_frames, frame_size, camera_contract,
    )
    kind = str(track.get("class_name", "unknown"))
    world_waypoints = _speed_limited_waypoints(
        world_waypoints, hidden_range, fps, kind,
    )
    boundary = _boundary_evidence(
        before, after, world_waypoints, frame_size, camera_contract,
    )
    lifecycle = _resolved_lifecycle(lifecycle, boundary)
    duration_seconds = _gap_duration_seconds(hidden_range, fps)
    confidence = _prediction_confidence(
        track, before, after, boundary, lifecycle, duration_seconds,
    )
    speed = _path_speed(world_waypoints, hidden_range, fps)
    return _prediction_contract(
        before, lifecycle, confidence, boundary, world_waypoints,
        _kinematic_contract(kind, speed, duration_seconds), speed,
    )


def heading_confidence_multiplier(disagreement_degrees: float) -> float:
    if disagreement_degrees <= SUPPORTED_HEADING_DISAGREEMENT:
        return 1.0
    if disagreement_degrees <= PLAUSIBLE_TURN_DISAGREEMENT:
        return 0.82
    if disagreement_degrees <= AMBIGUOUS_HEADING_DISAGREEMENT:
        return 0.58
    return 0.35


def position_residual_confidence_multiplier(residual_meters: float | None) -> float:
    if residual_meters is None or residual_meters <= SUPPORTED_POSITION_RESIDUAL_METERS:
        return 1.0
    if residual_meters <= PLAUSIBLE_POSITION_RESIDUAL_METERS:
        return 0.82
    if residual_meters <= AMBIGUOUS_POSITION_RESIDUAL_METERS:
        return 0.58
    return 0.30


def duration_confidence_multiplier(duration_seconds: float) -> float:
    excess_seconds = max(0.0, duration_seconds - LONG_GAP_CONFIDENCE_START_SECONDS)
    multiplier = math.exp(-LONG_GAP_CONFIDENCE_DECAY_PER_SECOND * excess_seconds)
    return round(max(MINIMUM_DURATION_CONFIDENCE_MULTIPLIER, multiplier), 4)


def fidelity_tier(entity_confidence: float, calibration_confidence: float) -> str:
    effective_confidence = entity_confidence
    if calibration_confidence < 0.50:
        effective_confidence -= 0.25
    if effective_confidence >= 0.75:
        return "supported"
    if effective_confidence >= 0.50:
        return "plausible"
    return "weak"


def _prediction_contract(
    before: list[dict],
    lifecycle: str,
    confidence: float,
    boundary_evidence: dict,
    world_waypoints: list[dict],
    kinematics: dict,
    speed: float,
) -> dict:
    return {
        "lifecycle": lifecycle,
        "confidence": confidence,
        "boundary_evidence": boundary_evidence,
        "path_prediction": {
            "method": "centripetal_catmull_rom",
            "constraint_mode": "forward_prediction" if before else "reverse_entry_prediction",
            "post_gap_observation_role": (
                "soft_consistency_check" if before else "entry_boundary_evidence"
            ),
            "waypoints": world_waypoints,
        },
        "kinematics": kinematics,
        "speed_meters_per_second": speed,
    }


def _boundary_detections(
    track: dict,
    boundary_frame: int,
    side: str,
    fps: float,
) -> list[dict]:
    window_frames = max(2, round(BOUNDARY_WINDOW_SECONDS * fps))
    detections = track.get("detections", [])
    if side == "before":
        eligible = [
            item for item in detections
            if boundary_frame - window_frames <= item["frame"] < boundary_frame
        ]
        return eligible[-3:]
    eligible = [
        item for item in detections
        if boundary_frame < item["frame"] <= boundary_frame + window_frames
    ]
    return eligible[:3]


def _lifecycle(before: list[dict], after: list[dict]) -> str | None:
    valid_before = (
        bool(before)
        and float(before[-1].get("confidence", 0.0)) >= MINIMUM_BOUNDARY_CONFIDENCE
    )
    valid_after = (
        bool(after)
        and float(after[0].get("confidence", 0.0)) >= MINIMUM_BOUNDARY_CONFIDENCE
    )
    if valid_before and valid_after:
        return "continuous"
    if valid_before:
        return "exits"
    if valid_after:
        return "enters"
    return None


def _predicted_image_points(
    before: list[dict],
    after: list[dict],
    waypoint_frames: list[int],
    lifecycle: str,
) -> list[tuple[float, float]]:
    if before:
        anchor = _ground_point(before[-1])
        velocity = _image_velocity(before)
        return [
            _advance(anchor, velocity, frame_index - before[-1]["frame"])
            for frame_index in waypoint_frames
        ]
    anchor = _ground_point(after[0])
    velocity = _image_velocity(after)
    if lifecycle == "enters" and len(after) < 2:
        velocity = (0.0, 0.0)
    return [
        _advance(anchor, velocity, frame_index - after[0]["frame"])
        for frame_index in waypoint_frames
    ]


def _world_waypoints(
    points: list[tuple[float, float]],
    waypoint_frames: list[int],
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> list[dict]:
    width, height = frame_size
    return [
        {
            "role": _waypoint_role(index, len(waypoint_frames)),
            "frame": frame_index,
            "world": image_point_to_world(
                point[0], point[1], width, height, camera_contract,
            ),
        }
        for index, (frame_index, point) in enumerate(zip(waypoint_frames, points))
    ]


def _boundary_evidence(
    before: list[dict],
    after: list[dict],
    world_waypoints: list[dict],
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> dict:
    pre_heading = _heading(_image_velocity(before)) if len(before) >= 2 else None
    post_heading = _heading(_image_velocity(after)) if len(after) >= 2 else None
    disagreement = (
        _angle_difference(pre_heading, post_heading)
        if pre_heading is not None and post_heading is not None
        else 0.0
    )
    residual = _post_gap_residual(
        before, after, world_waypoints, frame_size, camera_contract,
    )
    return {
        "pre_gap_heading_degrees": round(pre_heading, 3) if pre_heading is not None else None,
        "post_gap_heading_degrees": round(post_heading, 3) if post_heading is not None else None,
        "heading_disagreement_degrees": round(disagreement, 3),
        "post_gap_position_residual_meters": round(residual, 4) if residual is not None else None,
        "post_gap_residual_confidence_multiplier": (
            position_residual_confidence_multiplier(residual)
        ),
        "before_frame": before[-1]["frame"] if before else None,
        "after_frame": after[0]["frame"] if after else None,
    }


def _post_gap_residual(
    before: list[dict],
    after: list[dict],
    world_waypoints: list[dict],
    frame_size: tuple[int, int],
    camera_contract: dict,
) -> float | None:
    if not before or not after:
        return None
    post_point = _ground_point(after[0])
    post_world = image_point_to_world(
        post_point[0], post_point[1], frame_size[0], frame_size[1], camera_contract,
    )
    return _distance(world_waypoints[-1]["world"], post_world)


def _prediction_confidence(
    track: dict,
    before: list[dict],
    after: list[dict],
    boundary_evidence: dict,
    lifecycle: str,
    duration_seconds: float,
) -> float:
    evidence_scores = [
        float(item.get("confidence", 0.0)) for item in before[-1:] + after[:1]
    ]
    evidence_score = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
    continuity_score = _continuity_score(track)
    lifecycle_factor = {
        "continuous": 1.0, "enters": 0.75, "exits": 0.75, "uncertain": 0.50,
    }[lifecycle]
    heading_factor = heading_confidence_multiplier(
        boundary_evidence["heading_disagreement_degrees"],
    )
    residual_factor = float(
        boundary_evidence["post_gap_residual_confidence_multiplier"],
    )
    duration_factor = duration_confidence_multiplier(duration_seconds)
    confidence = 0.55 * evidence_score + 0.45 * continuity_score
    combined = confidence * lifecycle_factor * heading_factor * residual_factor * duration_factor
    return round(max(0.0, min(1.0, combined)), 4)


def _continuity_score(track: dict) -> float:
    continuity_value = track.get("continuity_confidence")
    if continuity_value is not None:
        return float(continuity_value)
    return float(track.get("avg_confidence", 0.0)) * UNMATCHED_CONTINUITY_FACTOR


def _resolved_lifecycle(lifecycle: str, boundary_evidence: dict) -> str:
    heading_disagreement = float(boundary_evidence["heading_disagreement_degrees"])
    if lifecycle == "continuous" and heading_disagreement > AMBIGUOUS_HEADING_DISAGREEMENT:
        return "uncertain"
    return lifecycle


def _image_velocity(detections: list[dict]) -> tuple[float, float]:
    if len(detections) < 2:
        return 0.0, 0.0
    first, last = detections[0], detections[-1]
    frame_delta = max(1, last["frame"] - first["frame"])
    first_point, last_point = _ground_point(first), _ground_point(last)
    return (
        (last_point[0] - first_point[0]) / frame_delta,
        (last_point[1] - first_point[1]) / frame_delta,
    )


def _ground_point(detection: dict) -> tuple[float, float]:
    x1, _, x2, y2 = detection["bbox"]
    return (x1 + x2) / 2.0, float(y2)


def _advance(
    point: tuple[float, float],
    velocity: tuple[float, float],
    frames: int,
) -> tuple[float, float]:
    return point[0] + velocity[0] * frames, point[1] + velocity[1] * frames


def _heading(velocity: tuple[float, float]) -> float:
    return math.degrees(math.atan2(velocity[1], velocity[0])) % 360.0


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum(
        (first[index] - second[index]) ** 2 for index in range(3)
    ))


def _path_speed(
    waypoints: list[dict],
    hidden_range: tuple[int, int],
    fps: float,
) -> float:
    path_distance = sum(
        _distance(waypoints[index - 1]["world"], waypoints[index]["world"])
        for index in range(1, len(waypoints))
    )
    return round(path_distance / _gap_duration_seconds(hidden_range, fps), 4)


def _gap_duration_seconds(hidden_range: tuple[int, int], fps: float) -> float:
    return max(1.0 / fps, (hidden_range[1] - hidden_range[0] + 1) / fps)


def _waypoint_frames(hidden_range: tuple[int, int], fps: float) -> list[int]:
    hidden_start, hidden_end = hidden_range
    duration_seconds = _gap_duration_seconds(hidden_range, fps)
    waypoint_count = min(
        MAXIMUM_PATH_WAYPOINTS,
        max(3, math.ceil(duration_seconds / WAYPOINT_INTERVAL_SECONDS) + 1),
    )
    return [
        hidden_start + round(index * (hidden_end - hidden_start) / (waypoint_count - 1))
        for index in range(waypoint_count)
    ]


def _waypoint_role(index: int, waypoint_count: int) -> str:
    if index == 0:
        return "start"
    if index == waypoint_count - 1:
        return "predicted_end"
    return f"inferred_{index:02d}"


def _speed_limited_waypoints(
    waypoints: list[dict],
    hidden_range: tuple[int, int],
    fps: float,
    kind: str,
) -> list[dict]:
    maximum_speed = MAXIMUM_SPEED_METERS_PER_SECOND.get(
        kind, DEFAULT_MAXIMUM_SPEED_METERS_PER_SECOND,
    )
    measured_speed = _path_speed(waypoints, hidden_range, fps)
    if measured_speed <= maximum_speed:
        return waypoints
    anchor = waypoints[0]["world"]
    scale = maximum_speed / measured_speed
    return [
        {
            **waypoint,
            "world": [
                round(anchor[axis] + (coordinate - anchor[axis]) * scale, 5)
                for axis, coordinate in enumerate(waypoint["world"])
            ],
        }
        for waypoint in waypoints
    ]


def _kinematic_contract(kind: str, speed: float, duration_seconds: float) -> dict:
    return {
        "model": "ground_plane_kinematic",
        "duration_seconds": round(duration_seconds, 4),
        "predicted_speed_meters_per_second": speed,
        "maximum_speed_meters_per_second": MAXIMUM_SPEED_METERS_PER_SECOND.get(
            kind, DEFAULT_MAXIMUM_SPEED_METERS_PER_SECOND,
        ),
        "maximum_acceleration_meters_per_second_squared": (
            MAXIMUM_ACCELERATION_METERS_PER_SECOND_SQUARED.get(kind, 4.0)
        ),
        "maximum_turn_rate_degrees_per_second": (
            MAXIMUM_TURN_RATE_DEGREES_PER_SECOND.get(kind, 45.0)
        ),
        "ground_contact_required": True,
    }
