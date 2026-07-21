from dataclasses import dataclass
from statistics import median


MINIMUM_HEIGHT_OBSERVATIONS = 5
HEIGHT_MAD_MULTIPLIER = 3.0
MAXIMUM_STABLE_HEIGHT_SPREAD = 0.15
SUPPORTED_CONFIDENCE_THRESHOLD = 0.75
REVIEW_CONFIDENCE_THRESHOLD = 0.50
CALIBRATION_WEIGHTS = {
    "static_feature_inlier_score": 0.25,
    "camera_motion_fit_score": 0.20,
    "ground_reprojection_score": 0.20,
    "horizon_stability_score": 0.15,
    "height_prior_stability_score": 0.15,
    "evidence_support_score": 0.05,
}


@dataclass(frozen=True)
class GroundCalibration:
    horizon_normalized_y: float = 0.39
    camera_height_meters: float = 1.8
    field_of_view_degrees: float = 58.0
    ground_near_y: float = 0.94
    ground_far_y: float = 0.42
    near_depth_meters: float = 3.0
    far_depth_meters: float = 20.0


def robust_height_prior(tracks: list[dict], frame_width: int, frame_height: int) -> dict:
    stable_track_medians: list[float] = []
    track_reports: list[dict] = []
    for track in tracks:
        if track.get("class_name") != "person":
            continue
        accepted_heights = _eligible_heights(track.get("detections", []), frame_width, frame_height)
        track_report = _track_height_report(str(track.get("id")), accepted_heights)
        track_reports.append(track_report)
        if track_report["stable"]:
            stable_track_medians.append(track_report["median_height_pixels"])
    combined_median = median(stable_track_medians) if stable_track_medians else 0.0
    combined_mad = _median_absolute_deviation(stable_track_medians, combined_median)
    return {
        "track_count": len(stable_track_medians),
        "median_height_pixels": round(combined_median, 3),
        "median_absolute_deviation": round(combined_mad, 3),
        "stable": bool(stable_track_medians),
        "tracks": track_reports,
    }


def calibration_confidence(components: dict[str, float]) -> dict:
    available = {name: _clamp(score) for name, score in components.items() if name in CALIBRATION_WEIGHTS}
    available_weight = sum(CALIBRATION_WEIGHTS[name] for name in available)
    weighted_score = 0.0
    if available_weight:
        weighted_score = sum(available[name] * CALIBRATION_WEIGHTS[name] for name in available) / available_weight
    required_available = {"camera_motion_fit_score", "ground_reprojection_score"}.issubset(available)
    if len(available) < 3 or not required_available:
        weighted_score = min(weighted_score, 0.49)
    final_score = round(_clamp(weighted_score), 4)
    return {
        "score": final_score,
        "tier": _confidence_tier(final_score),
        "components": available,
        "weights": {name: CALIBRATION_WEIGHTS[name] for name in available},
    }


def build_camera_contract(scene_report: dict, settings: GroundCalibration | None = None) -> dict:
    calibration = settings or GroundCalibration()
    video = scene_report["video"]
    height_prior = robust_height_prior(scene_report.get("tracks", []), video["width"], video["height"])
    support_score = min(1.0, height_prior["track_count"] / 5.0)
    height_score = _height_stability_score(height_prior)
    motion_report = scene_report.get("camera_motion_report", {})
    confidence = calibration_confidence({
        "static_feature_inlier_score": float(motion_report.get("static_feature_inlier_score", 0.0)),
        "camera_motion_fit_score": float(motion_report.get("camera_motion_fit_score", 0.0)),
        "height_prior_stability_score": height_score,
        "evidence_support_score": support_score,
    })
    motion_model = motion_report.get("classification", "unclassified")
    return {
        "mode": "generic_ground_prior",
        "motion_model": motion_model,
        "motion_applied_to_render": False,
        "compatibility": {
            "status": "supported" if motion_model == "static_camera" else "experimental",
            "reason": (
                "Static-camera evidence is compatible with the current renderer."
                if motion_model == "static_camera"
                else "Camera motion is measured but is not yet applied to the Blender camera."
            ),
        },
        "calibration_confidence": confidence["score"],
        "calibration_report": confidence,
        "focal_length_mm": 32.0,
        "position": [0.0, -2.0, calibration.camera_height_meters],
        "look_at": [0.0, 9.0, 1.0],
        "horizon_normalized_y": calibration.horizon_normalized_y,
        "field_of_view_degrees": calibration.field_of_view_degrees,
        "height_prior": height_prior,
        "camera_motion_report": motion_report,
        "ground_mapping": {
            "near_y": calibration.ground_near_y,
            "far_y": calibration.ground_far_y,
            "near_depth_meters": calibration.near_depth_meters,
            "far_depth_meters": calibration.far_depth_meters,
            "source": "generic_prior_not_measured_geometry",
        },
    }


def image_point_to_world(
    image_x: float,
    image_y: float,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> list[float]:
    mapping = camera_contract["ground_mapping"]
    normalized_x = (image_x / frame_width) - 0.5
    normalized_y = image_y / frame_height
    denominator = max(0.01, mapping["near_y"] - mapping["far_y"])
    depth_ratio = _clamp((mapping["near_y"] - normalized_y) / denominator)
    depth = mapping["near_depth_meters"] + depth_ratio * (
        mapping["far_depth_meters"] - mapping["near_depth_meters"]
    )
    horizontal_span = 5.5 + depth * 0.42
    return [round(normalized_x * horizontal_span, 4), round(depth, 4), 0.0]


def _eligible_heights(detections: list[dict], frame_width: int, frame_height: int) -> list[float]:
    heights: list[float] = []
    for detection in detections:
        bbox = detection.get("bbox", [])
        if len(bbox) != 4 or float(detection.get("confidence", 0.0)) < 0.35:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= 1 or y1 <= 1 or x2 >= frame_width - 1 or y2 >= frame_height - 1:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        heights.append(float(y2 - y1))
    return heights


def _track_height_report(track_id: str, heights: list[float]) -> dict:
    center = median(heights) if heights else 0.0
    deviation = _median_absolute_deviation(heights, center)
    filtered = [
        height for height in heights
        if (deviation == 0.0 and height == center)
        or (deviation > 0.0 and abs(height - center) <= HEIGHT_MAD_MULTIPLIER * deviation)
    ]
    filtered_center = median(filtered) if filtered else 0.0
    filtered_deviation = _median_absolute_deviation(filtered, filtered_center)
    spread = filtered_deviation / filtered_center if filtered_center else 1.0
    stable = len(filtered) >= MINIMUM_HEIGHT_OBSERVATIONS and spread <= MAXIMUM_STABLE_HEIGHT_SPREAD
    return {
        "track_id": track_id,
        "accepted_observations": len(filtered),
        "median_height_pixels": round(filtered_center, 3),
        "mad_pixels": round(filtered_deviation, 3),
        "relative_spread": round(spread, 4),
        "stable": stable,
    }


def _median_absolute_deviation(values: list[float], center: float) -> float:
    return median([abs(value - center) for value in values]) if values else 0.0


def _height_stability_score(height_prior: dict) -> float:
    center = float(height_prior["median_height_pixels"])
    deviation = float(height_prior["median_absolute_deviation"])
    if center <= 0.0:
        return 0.0
    return _clamp(1.0 - (deviation / center) / MAXIMUM_STABLE_HEIGHT_SPREAD)


def _confidence_tier(score: float) -> str:
    if score >= SUPPORTED_CONFIDENCE_THRESHOLD:
        return "supported"
    if score >= REVIEW_CONFIDENCE_THRESHOLD:
        return "review"
    return "unreliable"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
