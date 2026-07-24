import math


LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_KNEE_INDEX = 13
RIGHT_KNEE_INDEX = 14
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16
MOTION_KEYPOINT_INDEXES = (
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    LEFT_HIP_INDEX,
    RIGHT_HIP_INDEX,
    LEFT_KNEE_INDEX,
    RIGHT_KNEE_INDEX,
    LEFT_ANKLE_INDEX,
    RIGHT_ANKLE_INDEX,
)
MINIMUM_POSE_CONFIDENCE = 0.35
IDLE_SPEED_METERS_PER_SECOND = 0.20
BRISK_WALK_SPEED_METERS_PER_SECOND = 1.50
RUN_SPEED_METERS_PER_SECOND = 2.40
MINIMUM_CADENCE_SCALE = 0.70
MAXIMUM_CADENCE_SCALE = 1.45
MOTION_BLEND_SECONDS = 0.18
NOMINAL_CLIP_SPEEDS = {
    "idle": 0.0,
    "walk": 1.20,
    "brisk_walk": 1.85,
    "run": 3.20,
}


def build_motion_profile(
    track: dict,
    hidden_range: tuple[int, int],
    speed_meters_per_second: float,
    fallback_phase_offset: float,
) -> dict:
    boundary_poses = _boundary_pose_evidence(track, hidden_range)
    pose_confidence = _mean_pose_confidence(boundary_poses)
    usable_pose_evidence = (
        boundary_poses
        if pose_confidence >= MINIMUM_POSE_CONFIDENCE else []
    )
    phase_offset = _phase_offset(
        usable_pose_evidence,
        fallback_phase_offset,
        pose_confidence,
    )
    clip = motion_clip("idle" if speed_meters_per_second < IDLE_SPEED_METERS_PER_SECOND else "walk",
                       speed_meters_per_second)
    return {
        "schema_version": 1,
        "source": (
            "yolo_pose_visible_boundaries"
            if usable_pose_evidence else "kinematic_fallback"
        ),
        "clip": clip,
        "phase_offset": round(phase_offset, 4),
        "cadence_scale": cadence_scale(clip, speed_meters_per_second),
        "blend_seconds": MOTION_BLEND_SECONDS,
        "pose_confidence": round(pose_confidence, 4),
        "evidence": [
            {
                "frame": int(item["frame"]),
                "side": str(item["side"]),
                "confidence": round(_pose_confidence(item["pose_evidence"]), 4),
            }
            for item in usable_pose_evidence
        ],
    }


def synchronize_motion_profile(entity: dict) -> None:
    profile = entity.get("motion_profile")
    animation = entity.get("animation")
    if not isinstance(profile, dict) or not isinstance(animation, dict):
        return
    speed = float(animation.get("speed_meters_per_second", 0.0))
    clip = motion_clip(str(animation.get("state", "idle")), speed)
    profile["clip"] = clip
    profile["cadence_scale"] = cadence_scale(clip, speed)


def motion_clip(animation_state: str, speed_meters_per_second: float) -> str:
    if animation_state == "idle" or speed_meters_per_second < IDLE_SPEED_METERS_PER_SECOND:
        return "idle"
    if speed_meters_per_second >= RUN_SPEED_METERS_PER_SECOND:
        return "run"
    if speed_meters_per_second >= BRISK_WALK_SPEED_METERS_PER_SECOND:
        return "brisk_walk"
    return "walk"


def cadence_scale(clip: str, speed_meters_per_second: float) -> float:
    nominal_speed = NOMINAL_CLIP_SPEEDS.get(clip, NOMINAL_CLIP_SPEEDS["walk"])
    if nominal_speed <= 0.0:
        return 1.0
    requested_scale = speed_meters_per_second / nominal_speed
    return round(
        max(MINIMUM_CADENCE_SCALE, min(MAXIMUM_CADENCE_SCALE, requested_scale)),
        4,
    )


def _boundary_pose_evidence(
    track: dict,
    hidden_range: tuple[int, int],
) -> list[dict]:
    hidden_start, hidden_end = hidden_range
    detections = [
        item for item in track.get("detections", [])
        if isinstance(item.get("pose_evidence"), dict)
    ]
    before = [item for item in detections if int(item["frame"]) < hidden_start]
    after = [item for item in detections if int(item["frame"]) > hidden_end]
    selected = []
    if before:
        selected.append({**max(before, key=lambda item: int(item["frame"])), "side": "before"})
    if after:
        selected.append({**min(after, key=lambda item: int(item["frame"])), "side": "after"})
    return selected


def _mean_pose_confidence(boundary_poses: list[dict]) -> float:
    if not boundary_poses:
        return 0.0
    return sum(
        _pose_confidence(item["pose_evidence"])
        for item in boundary_poses
    ) / len(boundary_poses)


def _pose_confidence(pose_evidence: dict) -> float:
    keypoints = pose_evidence.get("keypoints", [])
    if not isinstance(keypoints, list):
        return 0.0
    confidences = [
        float(keypoints[index][2])
        for index in MOTION_KEYPOINT_INDEXES
        if index < len(keypoints) and _valid_keypoint(keypoints[index])
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _phase_offset(
    boundary_poses: list[dict],
    fallback_phase_offset: float,
    pose_confidence: float,
) -> float:
    if not boundary_poses or pose_confidence < MINIMUM_POSE_CONFIDENCE:
        return float(fallback_phase_offset) % 1.0
    phase = _gait_phase(boundary_poses[0]["pose_evidence"])
    return phase if phase is not None else float(fallback_phase_offset) % 1.0


def _gait_phase(pose_evidence: dict) -> float | None:
    keypoints = pose_evidence.get("keypoints", [])
    if not isinstance(keypoints, list) or len(keypoints) <= RIGHT_ANKLE_INDEX:
        return None
    left_ankle = keypoints[LEFT_ANKLE_INDEX]
    right_ankle = keypoints[RIGHT_ANKLE_INDEX]
    if not _valid_keypoint(left_ankle) or not _valid_keypoint(right_ankle):
        return None
    if min(float(left_ankle[2]), float(right_ankle[2])) < MINIMUM_POSE_CONFIDENCE:
        return None
    horizontal_delta = float(left_ankle[0]) - float(right_ankle[0])
    vertical_delta = float(left_ankle[1]) - float(right_ankle[1])
    return (math.atan2(vertical_delta, horizontal_delta) + math.pi) / math.tau


def _valid_keypoint(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(
        isinstance(channel, (int, float)) and not isinstance(channel, bool)
        for channel in value
    )
