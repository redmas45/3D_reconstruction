import json
from pathlib import Path

from domain.camera_calibration import build_camera_contract
from domain.motion_profile import build_motion_profile
from domain.path_prediction import build_entity_prediction, fidelity_tier
from domain.render_resolution import adaptive_render_scale_percent


PLAN_SCHEMA_VERSION = 2
PLAN_STRATEGY = "ai_inferred_forensic_3d"
RENDERABLE_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
DEFAULT_MAX_RENDER_ENTITIES = 3
MINIMUM_PRESENTATION_CONFIDENCE = 0.45
MINIMUM_PRESENTATION_RELEVANCE = 0.12
MAXIMUM_WEAK_PRESENTATION_ENTITIES = 0
MAXIMUM_DUPLICATE_BOUNDARY_IOU = 0.55
MINIMUM_VISIBLE_ANCHOR_FRACTION = 0.05
MAXIMUM_VISIBLE_ANCHOR_FRACTION = 0.95
LIFECYCLE_FACTORS = {"continuous": 1.0, "enters": 0.75, "exits": 0.75, "uncertain": 0.50}
DEFAULT_RENDER_CONFIGURATION = {
    "engine": "BLENDER_EEVEE_NEXT",
    "preview_scale_percent": 75,
    "production_scale_percent": 100,
    "cycles_compute_device": "CUDA",
    "cycles_samples": 16,
    "cycles_use_denoising": True,
}
SUPPORTED_RENDER_ENGINES = frozenset({"BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "CYCLES"})
SUPPORTED_CYCLES_COMPUTE_DEVICES = frozenset({"CUDA", "OPTIX"})


class PlanValidationError(ValueError):
    pass


def build_reconstruction_plan_v2(
    scene_report: dict,
    identity_registry: dict,
    hidden_range: tuple[int, int],
    gap_index: int,
    maximum_entities: int = DEFAULT_MAX_RENDER_ENTITIES,
    context_frame_path: Path | None = None,
    post_context_frame_path: Path | None = None,
    render_configuration: dict | None = None,
) -> dict:
    video = scene_report["video"]
    camera = build_camera_contract(scene_report)
    candidates = [
        track for track in scene_report.get("tracks", [])
        if track.get("class_name") in RENDERABLE_CLASSES
    ]
    entities = _planned_entities(candidates, identity_registry, hidden_range, video, camera)
    selected_entities = _select_presentation_entities(entities, maximum_entities)
    plan = _plan_contract(
        hidden_range, gap_index, video, camera, identity_registry,
        entities, selected_entities, context_frame_path, post_context_frame_path,
        render_configuration,
    )
    validate_reconstruction_plan_v2(plan)
    return plan


def _plan_contract(
    hidden_range: tuple[int, int],
    gap_index: int,
    video: dict,
    camera: dict,
    identity_registry: dict,
    entities: list[dict],
    selected_entities: list[dict],
    context_frame_path: Path | None,
    post_context_frame_path: Path | None,
    render_configuration: dict | None,
) -> dict:
    frame_count = hidden_range[1] - hidden_range[0] + 1
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "strategy": PLAN_STRATEGY,
        "gap_index": gap_index,
        "hidden_range": {"start": hidden_range[0], "end": hidden_range[1]},
        "fps": float(video["fps"]),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / float(video["fps"]), 4),
        "overall_confidence": _overall_confidence(selected_entities, camera),
        "camera": camera,
        "environment": _environment_contract(
            hidden_range[0] - 1, context_frame_path, post_context_frame_path,
            camera, render_configuration,
        ),
        "render": _render_contract(render_configuration, video),
        "identity_registry": {
            "schema_version": identity_registry["schema_version"],
            "generator_version": identity_registry["generator_version"],
        },
        "selection_report": _selection_report(entities, selected_entities),
        "entities": selected_entities,
    }


def _environment_contract(
    backplate_frame: int,
    context_frame_path: Path | None,
    post_context_frame_path: Path | None,
    camera: dict,
    render_configuration: dict | None,
) -> dict:
    configured = render_configuration or {}
    hybrid_enabled = all((
        bool(configured.get("hybrid_static_backplate", True)),
        context_frame_path is not None,
    ))
    return {
        "style": "forensic_3d",
        "ground_color": [0.035, 0.047, 0.062],
        "grid_color": [0.04, 0.62, 0.68],
        "backplate_frame": backplate_frame,
        "context_frame_path": str(context_frame_path.resolve()) if context_frame_path else None,
        "post_context_frame_path": (
            str(post_context_frame_path.resolve()) if post_context_frame_path else None
        ),
        "context_treatment": "cleaned_blurred_boundary_transition",
        "presentation_mode": True,
        "show_debug_grid": False,
        "show_debug_paths": False,
        "proxy_profile": "neutral",
        "hybrid_backplate_enabled": hybrid_enabled,
        "hybrid_backplate_reason": (
            _backplate_reason(camera)
            if hybrid_enabled else "visible_boundary_frame_unavailable"
        ),
    }


def _selection_report(entities: list[dict], selected_entities: list[dict]) -> dict:
    selected_ids = {entity["id"] for entity in selected_entities}
    excluded_entities = [
        {
            "id": entity["id"],
            "relevance_score": entity["relevance_score"],
            "confidence": entity["confidence"],
            "reason": _exclusion_reason(entity),
        }
        for entity in entities
        if entity["id"] not in selected_ids
    ]
    return {
        "candidate_count": len(entities),
        "rendered_count": len(selected_entities),
        "rendered_ids": [entity["id"] for entity in selected_entities],
        "excluded_ids": [entity["id"] for entity in excluded_entities],
        "excluded_entities": excluded_entities,
        "presentation_cap_applied": any(item["reason"] == "presentation_cap" for item in excluded_entities),
    }


def _render_contract(render_configuration: dict | None, video: dict) -> dict:
    configured = render_configuration or {}
    scale_percent = adaptive_render_scale_percent(
        int(video["width"]),
        int(video["height"]),
        int(configured.get(
            "production_scale_percent",
            DEFAULT_RENDER_CONFIGURATION["production_scale_percent"],
        )),
        int(configured.get("minimum_render_long_edge", 960)),
        int(configured.get("maximum_render_long_edge", 1280)),
    )
    contract = {
        "engine": str(configured.get("engine", DEFAULT_RENDER_CONFIGURATION["engine"])),
        "preview_scale_percent": int(configured.get(
            "preview_scale_percent", DEFAULT_RENDER_CONFIGURATION["preview_scale_percent"]
        )),
        "production_scale_percent": scale_percent,
        "target_fps": int(configured.get("target_fps", 10)),
        "checkpoint_frame_batch": int(configured.get("checkpoint_frame_batch", 24)),
        "diagnostic_pose_count": int(configured.get("diagnostic_pose_count", 5)),
        "source_width": int(video["width"]),
        "source_height": int(video["height"]),
        "minimum_render_long_edge": int(configured.get("minimum_render_long_edge", 960)),
        "maximum_render_long_edge": int(configured.get("maximum_render_long_edge", 1280)),
        "production_hud_mode": str(configured.get("production_hud_mode", "minimal")),
        "hybrid_static_backplate": bool(configured.get("hybrid_static_backplate", True)),
    }
    if contract["engine"] == "CYCLES":
        contract.update({
            "cycles_compute_device": str(configured.get(
                "cycles_compute_device", DEFAULT_RENDER_CONFIGURATION["cycles_compute_device"]
            )),
            "cycles_samples": int(configured.get(
                "cycles_samples", DEFAULT_RENDER_CONFIGURATION["cycles_samples"]
            )),
            "cycles_use_denoising": bool(configured.get(
                "cycles_use_denoising", DEFAULT_RENDER_CONFIGURATION["cycles_use_denoising"]
            )),
        })
    return contract


def _exclusion_reason(entity: dict) -> str:
    if not _visual_anchor_is_fully_visible(entity):
        return "boundary_clipped"
    if entity["relevance_score"] < MINIMUM_PRESENTATION_RELEVANCE:
        return "below_relevance_threshold"
    if entity["confidence"] < MINIMUM_PRESENTATION_CONFIDENCE:
        return "below_confidence_threshold"
    return "presentation_cap"


def validate_reconstruction_plan_v2(plan: dict) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PlanValidationError("Plan schema_version must be 2")
    if plan.get("strategy") != PLAN_STRATEGY:
        raise PlanValidationError("Plan strategy is unsupported")
    if not isinstance(plan.get("camera"), dict) or not 0.0 <= plan["camera"].get("calibration_confidence", -1.0) <= 1.0:
        raise PlanValidationError("Camera calibration confidence is invalid")
    _validate_render_contract(plan.get("render"))
    entities = plan.get("entities")
    if not isinstance(entities, list):
        raise PlanValidationError("Plan entities must be a list")
    for entity in entities:
        _validate_entity(entity)


def write_reconstruction_plan_v2(plan: dict, output_path: Path) -> None:
    validate_reconstruction_plan_v2(plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as plan_file:
        json.dump(plan, plan_file, indent=2)


def _planned_entities(
    tracks: list[dict],
    identity_registry: dict,
    hidden_range: tuple[int, int],
    video: dict,
    camera: dict,
) -> list[dict]:
    entities: list[dict] = []
    identities = identity_registry["identities"]
    for track in tracks:
        prediction = build_entity_prediction(
            track, hidden_range, float(video["fps"]), (int(video["width"]), int(video["height"])), camera,
        )
        identity = identities.get(track["id"])
        if prediction is None or identity is None:
            continue
        entities.append(_entity_contract(track, identity, prediction, video, camera, hidden_range))
    return entities


def _entity_contract(
    track: dict,
    identity: dict,
    prediction: dict,
    video: dict,
    camera: dict,
    hidden_range: tuple[int, int],
) -> dict:
    confidence = prediction["confidence"]
    relevance_score = _relevance_score(
        track, prediction["lifecycle"], confidence, video, hidden_range,
    )
    speed = prediction["speed_meters_per_second"]
    visual_anchor = _visual_anchor(track, hidden_range, video)
    motion_profile = _motion_profile(track, hidden_range, speed, identity)
    entity = {
        "id": track["id"],
        "identity_registry_id": track["id"],
        "kind": track["class_name"],
        "confidence": confidence,
        "fidelity_tier": fidelity_tier(confidence, camera["calibration_confidence"]),
        "lifecycle": prediction["lifecycle"],
        "relevance_score": relevance_score,
        "appearance": identity["appearance"],
        "body_proportions": identity["body_proportions"],
        "associated_objects": identity["associated_objects"],
        "visual_anchor": visual_anchor,
        "animation": {
            "state": "idle" if speed < 0.15 else "walk",
            "speed_meters_per_second": speed,
            "phase_offset": (
                motion_profile["phase_offset"]
                if motion_profile else identity["animation_phase"]
            ),
        },
        "boundary_evidence": prediction["boundary_evidence"],
        "path_prediction": prediction["path_prediction"],
        "kinematics": prediction["kinematics"],
        "uncertainty": {
            "position_radius_meters": _uncertainty_radius(
                confidence, prediction["kinematics"]["duration_seconds"],
            ),
            "alternative_paths": 0 if confidence >= 0.75 else 2,
        },
    }
    if motion_profile is not None:
        entity["motion_profile"] = motion_profile
    return entity


def _motion_profile(
    track: dict,
    hidden_range: tuple[int, int],
    speed: float,
    identity: dict,
) -> dict | None:
    if track["class_name"] != "person":
        return None
    return build_motion_profile(
        track,
        hidden_range,
        speed,
        float(identity["animation_phase"]),
    )


def _uncertainty_radius(confidence: float, duration_seconds: float) -> float:
    duration_growth = max(0.0, duration_seconds - 3.0) * 0.12
    return round(min(2.5, 0.25 + (1.0 - confidence) * 1.2 + duration_growth), 4)


def _relevance_score(
    track: dict,
    lifecycle: str,
    confidence: float,
    video: dict,
    hidden_range: tuple[int, int],
) -> float:
    detection = _closest_boundary_detection(track, hidden_range)
    x1, y1, x2, y2 = detection["bbox"]
    frame_area = max(1.0, float(video["width"] * video["height"]))
    screen_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    screen_size_factor = max(0.25, min(1.0, (screen_area / frame_area) ** 0.5 * 4.0))
    proximity_factor = max(0.50, min(1.0, 0.50 + 0.50 * y2 / float(video["height"])))
    lifecycle_factor = LIFECYCLE_FACTORS.get(lifecycle, LIFECYCLE_FACTORS["uncertain"])
    return round(confidence * proximity_factor * screen_size_factor * lifecycle_factor, 6)


def _closest_boundary_detection(track: dict, hidden_range: tuple[int, int]) -> dict:
    hidden_start, hidden_end = hidden_range
    visible_detections = [
        item for item in track.get("detections", [])
        if int(item.get("frame", hidden_start)) < hidden_start
        or int(item.get("frame", hidden_end)) > hidden_end
    ]
    if not visible_detections:
        return {"bbox": track.get("last_bbox", [0, 0, 1, 1])}

    def distance_to_gap(detection: dict) -> int:
        frame_index = int(detection["frame"])
        boundary = hidden_start if frame_index < hidden_start else hidden_end
        return abs(frame_index - boundary)

    return min(visible_detections, key=distance_to_gap)


def _validate_render_contract(value: object) -> None:
    if not isinstance(value, dict):
        raise PlanValidationError("Plan render contract must be an object")
    if value.get("engine") not in SUPPORTED_RENDER_ENGINES:
        raise PlanValidationError("Render engine is unsupported")
    integer_fields = (
        "preview_scale_percent", "production_scale_percent", "source_width", "source_height",
    )
    for field_name in integer_fields:
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise PlanValidationError(f"Render {field_name} must be a positive integer")
    for field_name in ("target_fps", "checkpoint_frame_batch"):
        field_value = value.get(field_name)
        if field_value is not None and (
            isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0
        ):
            raise PlanValidationError(f"Render {field_name} must be a positive integer")
    if value["engine"] != "CYCLES":
        return
    if value.get("cycles_compute_device") not in SUPPORTED_CYCLES_COMPUTE_DEVICES:
        raise PlanValidationError("Cycles compute device must be CUDA or OPTIX")
    if not isinstance(value.get("cycles_use_denoising"), bool):
        raise PlanValidationError("Cycles denoising flag must be boolean")
    cycles_samples = value.get("cycles_samples")
    if isinstance(cycles_samples, bool) or not isinstance(cycles_samples, int) or cycles_samples <= 0:
        raise PlanValidationError("Cycles samples must be a positive integer")


def _overall_confidence(entities: list[dict], camera: dict) -> float:
    entity_score = sum(entity["confidence"] for entity in entities) / len(entities) if entities else 0.0
    return round(0.72 * entity_score + 0.28 * camera["calibration_confidence"], 4)


def _select_presentation_entities(entities: list[dict], maximum_entities: int) -> list[dict]:
    ranked_entities = sorted(entities, key=lambda entity: entity["relevance_score"], reverse=True)
    supported_entities = [
        entity for entity in ranked_entities
        if entity["confidence"] >= MINIMUM_PRESENTATION_CONFIDENCE
        and entity["relevance_score"] >= MINIMUM_PRESENTATION_RELEVANCE
        and _visual_anchor_is_fully_visible(entity)
    ]
    weak_entities = [] if MAXIMUM_WEAK_PRESENTATION_ENTITIES == 0 else [
        entity for entity in ranked_entities
        if entity not in supported_entities
        and entity["relevance_score"] >= MINIMUM_PRESENTATION_RELEVANCE
    ][:MAXIMUM_WEAK_PRESENTATION_ENTITIES]
    selected_ids = {entity["id"] for entity in supported_entities + weak_entities}
    selected = [entity for entity in ranked_entities if entity["id"] in selected_ids]
    return _without_boundary_duplicates(selected)[:maximum_entities]


def _visual_anchor(
    track: dict,
    hidden_range: tuple[int, int],
    video: dict,
) -> dict:
    detection = _closest_boundary_detection(track, hidden_range)
    x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
    width = max(1.0, float(video["width"]))
    height = max(1.0, float(video["height"]))
    return {
        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "center_x_fraction": round(((x1 + x2) * 0.5) / width, 5),
        "ground_y_fraction": round(y2 / height, 5),
        "width_fraction": round(max(1.0, x2 - x1) / width, 5),
        "height_fraction": round(max(1.0, y2 - y1) / height, 5),
        "source_frame": int(detection.get("frame", hidden_range[0] - 1)),
    }


def _without_boundary_duplicates(entities: list[dict]) -> list[dict]:
    retained: list[dict] = []
    for entity in entities:
        if any(_entities_overlap(entity, existing) for existing in retained):
            continue
        retained.append(entity)
    return retained


def _visual_anchor_is_fully_visible(entity: dict) -> bool:
    anchor = entity.get("visual_anchor", {})
    center_x = float(anchor.get("center_x_fraction", 0.0))
    ground_y = float(anchor.get("ground_y_fraction", 1.0))
    half_width = float(anchor.get("width_fraction", 1.0)) * 0.5
    height = float(anchor.get("height_fraction", 1.0))
    return all((
        center_x - half_width >= MINIMUM_VISIBLE_ANCHOR_FRACTION,
        center_x + half_width <= MAXIMUM_VISIBLE_ANCHOR_FRACTION,
        ground_y <= MAXIMUM_VISIBLE_ANCHOR_FRACTION,
        ground_y - height >= MINIMUM_VISIBLE_ANCHOR_FRACTION,
    ))


def _entities_overlap(first: dict, second: dict) -> bool:
    if first["kind"] != second["kind"] or first["lifecycle"] != second["lifecycle"]:
        return False
    first_bbox = first.get("visual_anchor", {}).get("bbox", [])
    second_bbox = second.get("visual_anchor", {}).get("bbox", [])
    return _intersection_over_union(first_bbox, second_bbox) > MAXIMUM_DUPLICATE_BOUNDARY_IOU


def _intersection_over_union(first: list, second: list) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _backplate_reason(camera: dict) -> str:
    if camera.get("motion_model") == "static_camera":
        return "static_camera_visible_evidence"
    return "stabilized_visible_boundary_for_dynamic_camera"


def _validate_entity(entity: dict) -> None:
    required_fields = {
        "id", "kind", "confidence", "fidelity_tier", "lifecycle",
        "path_prediction", "visual_anchor",
    }
    if not required_fields.issubset(entity):
        raise PlanValidationError(f"Entity is missing required fields: {sorted(required_fields - set(entity))}")
    if entity["kind"] not in RENDERABLE_CLASSES:
        raise PlanValidationError(f"Entity kind is unsupported: {entity['kind']}")
    if not 0.0 <= float(entity["confidence"]) <= 1.0:
        raise PlanValidationError("Entity confidence is invalid")
    _validate_visual_anchor(entity["visual_anchor"])
    path_prediction = entity["path_prediction"]
    if path_prediction.get("method") != "centripetal_catmull_rom":
        raise PlanValidationError("Entity path method must be centripetal_catmull_rom")
    waypoints = path_prediction.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 3:
        raise PlanValidationError("Entity path must contain at least three waypoints")
    _validate_motion_profile(entity.get("motion_profile"))
    kinematics = entity.get("kinematics")
    if kinematics is not None:
        _validate_kinematics(kinematics)


def _validate_motion_profile(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PlanValidationError("Entity motion profile is invalid")
    if value.get("clip") not in {"idle", "walk", "brisk_walk", "run"}:
        raise PlanValidationError("Entity motion clip is unsupported")
    if value.get("source") not in {
        "yolo_pose_visible_boundaries",
        "kinematic_fallback",
    }:
        raise PlanValidationError("Entity motion source is unsupported")
    for field_name in ("phase_offset", "cadence_scale", "blend_seconds", "pose_confidence"):
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise PlanValidationError(f"Entity motion profile {field_name} must be numeric")
    if not 0.0 <= float(value["phase_offset"]) <= 1.0:
        raise PlanValidationError("Entity motion phase offset is invalid")
    if not 0.0 <= float(value["pose_confidence"]) <= 1.0:
        raise PlanValidationError("Entity motion pose confidence is invalid")
    if float(value["cadence_scale"]) <= 0.0 or float(value["blend_seconds"]) < 0.0:
        raise PlanValidationError("Entity motion timing is invalid")


def _validate_visual_anchor(value: object) -> None:
    if not isinstance(value, dict):
        raise PlanValidationError("Entity visual anchor must be an object")
    fractions = ("center_x_fraction", "ground_y_fraction", "width_fraction", "height_fraction")
    if any(
        not isinstance(value.get(field), (int, float))
        or isinstance(value.get(field), bool)
        or not 0.0 <= float(value[field]) <= 1.0
        for field in fractions
    ):
        raise PlanValidationError("Entity visual anchor fractions are invalid")
    bbox = value.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise PlanValidationError("Entity visual anchor bounding box is invalid")


def _validate_kinematics(value: object) -> None:
    if not isinstance(value, dict):
        raise PlanValidationError("Entity kinematics must be an object")
    if value.get("model") != "ground_plane_kinematic":
        raise PlanValidationError("Entity kinematics model is unsupported")
    positive_fields = (
        "duration_seconds",
        "maximum_speed_meters_per_second",
        "maximum_acceleration_meters_per_second_squared",
        "maximum_turn_rate_degrees_per_second",
    )
    for field_name in positive_fields:
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise PlanValidationError(f"Entity kinematics {field_name} must be numeric")
        if float(field_value) <= 0.0:
            raise PlanValidationError(f"Entity kinematics {field_name} must be positive")
    if value.get("ground_contact_required") is not True:
        raise PlanValidationError("Entity kinematics must require ground contact")
