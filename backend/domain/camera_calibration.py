from dataclasses import dataclass
from statistics import median

from domain.camera_projection import camera_pose, image_point_to_world


MINIMUM_HEIGHT_OBSERVATIONS = 5
HEIGHT_MAD_MULTIPLIER = 3.0
MAXIMUM_STABLE_HEIGHT_SPREAD = 0.15
MINIMUM_GROUND_FIT_OBSERVATIONS = 8
MINIMUM_NORMALIZED_HEIGHT_SPREAD = 0.04
MAXIMUM_GROUND_FIT_RESIDUAL = 0.06
ASSUMED_PERSON_HEIGHT_METERS = 1.72
MINIMUM_CAMERA_HEIGHT_METERS = 1.1
MAXIMUM_CAMERA_HEIGHT_METERS = 12.0
MINIMUM_CAMERA_HEIGHT_OBSERVATIONS = 5
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
    ground_fit = estimate_ground_geometry(
        scene_report.get("tracks", []), video["width"], video["height"],
    )
    support_score = min(1.0, height_prior["track_count"] / 5.0)
    height_score = _height_stability_score(height_prior)
    motion_report = scene_report.get("camera_motion_report", {})
    confidence_components = {
        "static_feature_inlier_score": float(motion_report.get("static_feature_inlier_score", 0.0)),
        "camera_motion_fit_score": float(motion_report.get("camera_motion_fit_score", 0.0)),
        "height_prior_stability_score": height_score,
        "evidence_support_score": support_score,
    }
    if ground_fit["supported"]:
        confidence_components["ground_reprojection_score"] = float(ground_fit["confidence"])
    confidence = calibration_confidence(confidence_components)
    motion_model = motion_report.get("classification", "unclassified")
    confidence = _motion_adjusted_confidence(confidence, motion_model)
    geometry_supported = ground_fit["supported"]
    horizon_normalized_y = (
        ground_fit["horizon_normalized_y"]
        if geometry_supported else calibration.horizon_normalized_y
    )
    camera_height_report = estimate_camera_height(
        scene_report.get("tracks", []),
        float(horizon_normalized_y),
        int(video["width"]),
        int(video["height"]),
    )
    camera_height_meters = (
        camera_height_report["height_meters"]
        if camera_height_report["supported"] else calibration.camera_height_meters
    )
    camera_pose_contract = camera_pose(
        camera_height_meters,
        float(horizon_normalized_y),
        calibration.field_of_view_degrees,
        int(video["width"]),
        int(video["height"]),
    )
    presentation_mode = (
        "source_camera_aligned"
        if motion_model == "static_camera"
        else "stabilized_forensic_view"
    )
    return {
        "mode": "person_height_ground_fit" if geometry_supported else "generic_ground_prior",
        "motion_model": motion_model,
        "motion_applied_to_render": False,
        "presentation_mode": presentation_mode,
        "compatibility": {
            "status": "supported" if motion_model == "static_camera" else "experimental",
            "reason": (
                "Static-camera evidence is compatible with the current renderer."
                if motion_model == "static_camera"
                else "Dynamic source footage is rendered as an explicitly labelled stabilized forensic view."
            ),
        },
        "calibration_confidence": confidence["score"],
        "calibration_report": confidence,
        "projection_model": "pinhole_ground_plane_v2",
        "focal_length_mm": camera_pose_contract["focal_length_mm"],
        "position": camera_pose_contract["position"],
        "look_at": camera_pose_contract["look_at"],
        "horizon_normalized_y": horizon_normalized_y,
        "field_of_view_degrees": calibration.field_of_view_degrees,
        "camera_height_report": camera_height_report,
        "height_prior": height_prior,
        "camera_motion_report": motion_report,
        "ground_fit_report": ground_fit,
        "ground_mapping": {
            "near_y": ground_fit["ground_near_y"] if geometry_supported else calibration.ground_near_y,
            "far_y": ground_fit["ground_far_y"] if geometry_supported else calibration.ground_far_y,
            "near_depth_meters": calibration.near_depth_meters,
            "far_depth_meters": calibration.far_depth_meters,
            "source": (
                "visible_person_ground_contact_fit"
                if geometry_supported else "generic_prior_not_measured_geometry"
            ),
        },
    }


def estimate_ground_geometry(tracks: list[dict], frame_width: int, frame_height: int) -> dict:
    observations = _ground_observations(tracks, frame_width, frame_height)
    if len(observations) < MINIMUM_GROUND_FIT_OBSERVATIONS:
        return _unsupported_ground_fit(len(observations), "insufficient_visible_person_contacts")
    heights = [item[0] for item in observations]
    bottoms = [item[1] for item in observations]
    if max(heights) - min(heights) < MINIMUM_NORMALIZED_HEIGHT_SPREAD:
        return _unsupported_ground_fit(len(observations), "insufficient_depth_variation")
    slope, intercept = _linear_fit(heights, bottoms)
    residual = median([
        abs(bottom - (intercept + slope * height))
        for height, bottom in observations
    ])
    valid_geometry = slope > 0.0 and 0.10 <= intercept <= 0.75
    confidence = _clamp(1.0 - residual / MAXIMUM_GROUND_FIT_RESIDUAL) if valid_geometry else 0.0
    supported = valid_geometry and confidence >= REVIEW_CONFIDENCE_THRESHOLD
    return {
        "supported": supported,
        "observation_count": len(observations),
        "horizon_normalized_y": round(intercept, 4),
        "ground_far_y": round(max(intercept + 0.03, _percentile(bottoms, 0.10)), 4),
        "ground_near_y": round(_percentile(bottoms, 0.90), 4),
        "median_reprojection_residual": round(residual, 5),
        "confidence": round(confidence, 4),
        "reason": "visible_person_ground_contact_fit" if supported else "unstable_ground_contact_fit",
    }


def estimate_camera_height(
    tracks: list[dict],
    horizon_normalized_y: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    estimates = _camera_height_estimates(
        tracks, horizon_normalized_y, frame_width, frame_height,
    )
    if len(estimates) < MINIMUM_CAMERA_HEIGHT_OBSERVATIONS:
        return _unsupported_camera_height(len(estimates))
    center = median(estimates)
    deviation = _median_absolute_deviation(estimates, center)
    filtered = _mad_filter(estimates, center, deviation)
    if len(filtered) < MINIMUM_CAMERA_HEIGHT_OBSERVATIONS:
        return _unsupported_camera_height(len(filtered))
    filtered_center = median(filtered)
    relative_spread = (
        _median_absolute_deviation(filtered, filtered_center) / filtered_center
        if filtered_center else 1.0
    )
    return {
        "supported": relative_spread <= MAXIMUM_STABLE_HEIGHT_SPREAD,
        "observation_count": len(filtered),
        "height_meters": round(filtered_center, 4),
        "relative_spread": round(relative_spread, 4),
        "source": "visible_person_vertical_geometry",
    }


def _camera_height_estimates(
    tracks: list[dict],
    horizon_normalized_y: float,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    estimates: list[float] = []
    for track in tracks:
        if track.get("class_name") != "person":
            continue
        for detection in track.get("detections", []):
            bbox = detection.get("bbox", [])
            if not _eligible_bbox(bbox, detection, frame_width, frame_height):
                continue
            _, top, _, bottom = [float(value) for value in bbox]
            horizon_pixels = horizon_normalized_y * frame_height
            ground_distance = bottom - horizon_pixels
            if ground_distance <= frame_height * 0.03:
                continue
            top_ratio = (top - horizon_pixels) / ground_distance
            denominator = 1.0 - top_ratio
            if denominator <= 0.0:
                continue
            estimate = ASSUMED_PERSON_HEIGHT_METERS / denominator
            if MINIMUM_CAMERA_HEIGHT_METERS <= estimate <= MAXIMUM_CAMERA_HEIGHT_METERS:
                estimates.append(estimate)
    return estimates


def _mad_filter(values: list[float], center: float, deviation: float) -> list[float]:
    if deviation == 0.0:
        return [value for value in values if value == center]
    return [
        value for value in values
        if abs(value - center) <= HEIGHT_MAD_MULTIPLIER * deviation
    ]


def _unsupported_camera_height(observation_count: int) -> dict:
    return {
        "supported": False,
        "observation_count": observation_count,
        "height_meters": None,
        "relative_spread": None,
        "source": "default_camera_height",
    }


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


def _ground_observations(
    tracks: list[dict], frame_width: int, frame_height: int,
) -> list[tuple[float, float]]:
    observations: list[tuple[float, float]] = []
    for track in tracks:
        if track.get("class_name") != "person":
            continue
        for detection in track.get("detections", []):
            bbox = detection.get("bbox", [])
            if not _eligible_bbox(bbox, detection, frame_width, frame_height):
                continue
            _, y1, _, y2 = [float(value) for value in bbox]
            observations.append((
                (y2 - y1) / frame_height,
                y2 / frame_height,
            ))
    return observations


def _eligible_bbox(
    bbox: list, detection: dict, frame_width: int, frame_height: int,
) -> bool:
    if len(bbox) != 4 or float(detection.get("confidence", 0.0)) < 0.35:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return all((
        x1 > 1,
        y1 > 1,
        x2 < frame_width - 1,
        y2 < frame_height - 1,
        x2 > x1,
        y2 > y1,
    ))


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


def _linear_fit(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    x_center = sum(x_values) / len(x_values)
    y_center = sum(y_values) / len(y_values)
    denominator = sum((value - x_center) ** 2 for value in x_values)
    if denominator <= 1e-9:
        return 0.0, y_center
    slope = sum(
        (x_value - x_center) * (y_value - y_center)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    return slope, y_center - slope * x_center


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * _clamp(fraction))
    return ordered[index]


def _unsupported_ground_fit(observation_count: int, reason: str) -> dict:
    return {
        "supported": False,
        "observation_count": observation_count,
        "horizon_normalized_y": None,
        "ground_far_y": None,
        "ground_near_y": None,
        "median_reprojection_residual": None,
        "confidence": 0.0,
        "reason": reason,
    }


def _motion_adjusted_confidence(confidence: dict, motion_model: str) -> dict:
    if motion_model == "static_camera":
        return confidence
    score = min(float(confidence["score"]), 0.49)
    return {
        **confidence,
        "score": round(score, 4),
        "tier": _confidence_tier(score),
        "motion_penalty": "dynamic_or_unclassified_camera",
    }


def _confidence_tier(score: float) -> str:
    if score >= SUPPORTED_CONFIDENCE_THRESHOLD:
        return "supported"
    if score >= REVIEW_CONFIDENCE_THRESHOLD:
        return "review"
    return "unreliable"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
