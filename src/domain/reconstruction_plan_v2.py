import json
from pathlib import Path

from domain.camera_calibration import build_camera_contract
from domain.path_prediction import build_entity_prediction, fidelity_tier


PLAN_SCHEMA_VERSION = 2
PLAN_STRATEGY = "ai_inferred_forensic_3d"
RENDERABLE_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
DEFAULT_MAX_RENDER_ENTITIES = 12
MINIMUM_PRESENTATION_CONFIDENCE = 0.40
MINIMUM_PRESENTATION_RELEVANCE = 0.10
MAXIMUM_WEAK_PRESENTATION_ENTITIES = 1
MINIMUM_WEAK_PRESENTATION_DEPTH_METERS = 4.5
LIFECYCLE_FACTORS = {"continuous": 1.0, "enters": 0.75, "exits": 0.75, "uncertain": 0.50}
DEFAULT_RENDER_CONFIGURATION = {
    "engine": "BLENDER_EEVEE_NEXT",
    "preview_scale_percent": 75,
    "production_scale_percent": 100,
}


class PlanValidationError(ValueError):
    pass


def build_reconstruction_plan_v2(
    scene_report: dict,
    identity_registry: dict,
    hidden_range: tuple[int, int],
    gap_index: int,
    maximum_entities: int = DEFAULT_MAX_RENDER_ENTITIES,
    context_frame_path: Path | None = None,
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
        entities, selected_entities, context_frame_path, render_configuration,
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
            hidden_range[0] - 1, context_frame_path, selected_entities,
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
    rendered_entities: list[dict],
) -> dict:
    proxy_profile = "street" if any(
        entity.get("kind") in VEHICLE_CLASSES for entity in rendered_entities
    ) else "neutral"
    return {
        "style": "forensic_3d",
        "ground_color": [0.035, 0.047, 0.062],
        "grid_color": [0.04, 0.62, 0.68],
        "backplate_frame": backplate_frame,
        "context_frame_path": str(context_frame_path.resolve()) if context_frame_path else None,
        "presentation_mode": True,
        "show_debug_paths": False,
        "proxy_profile": proxy_profile,
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
    return {
        "engine": str(configured.get("engine", DEFAULT_RENDER_CONFIGURATION["engine"])),
        "preview_scale_percent": int(configured.get(
            "preview_scale_percent", DEFAULT_RENDER_CONFIGURATION["preview_scale_percent"]
        )),
        "production_scale_percent": int(configured.get(
            "production_scale_percent", DEFAULT_RENDER_CONFIGURATION["production_scale_percent"]
        )),
        "source_width": int(video["width"]),
        "source_height": int(video["height"]),
    }


def _exclusion_reason(entity: dict) -> str:
    if entity["relevance_score"] < MINIMUM_PRESENTATION_RELEVANCE:
        return "below_relevance_threshold"
    midpoint_depth = entity["path_prediction"]["waypoints"][1]["world"][1]
    if entity["confidence"] < MINIMUM_PRESENTATION_CONFIDENCE:
        if midpoint_depth < MINIMUM_WEAK_PRESENTATION_DEPTH_METERS:
            return "weak_foreground_readability"
        return "weak_entity_budget"
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
    return {
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
        "animation": {
            "state": "idle" if speed < 0.15 else "walk",
            "speed_meters_per_second": speed,
            "phase_offset": identity["animation_phase"],
        },
        "boundary_evidence": prediction["boundary_evidence"],
        "path_prediction": prediction["path_prediction"],
        "uncertainty": {
            "position_radius_meters": round(0.25 + (1.0 - confidence) * 1.2, 4),
            "alternative_paths": 0 if confidence >= 0.75 else 2,
        },
    }


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
    if not isinstance(value.get("engine"), str) or not value["engine"].strip():
        raise PlanValidationError("Render engine must be a non-empty string")
    integer_fields = (
        "preview_scale_percent", "production_scale_percent", "source_width", "source_height",
    )
    for field_name in integer_fields:
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise PlanValidationError(f"Render {field_name} must be a positive integer")


def _overall_confidence(entities: list[dict], camera: dict) -> float:
    entity_score = sum(entity["confidence"] for entity in entities) / len(entities) if entities else 0.0
    return round(0.72 * entity_score + 0.28 * camera["calibration_confidence"], 4)


def _select_presentation_entities(entities: list[dict], maximum_entities: int) -> list[dict]:
    ranked_entities = sorted(entities, key=lambda entity: entity["relevance_score"], reverse=True)
    supported_entities = [
        entity for entity in ranked_entities
        if entity["confidence"] >= MINIMUM_PRESENTATION_CONFIDENCE
        and entity["relevance_score"] >= MINIMUM_PRESENTATION_RELEVANCE
    ]
    weak_entities = [
        entity for entity in ranked_entities
        if entity not in supported_entities
        and entity["relevance_score"] >= MINIMUM_PRESENTATION_RELEVANCE
        and entity["path_prediction"]["waypoints"][1]["world"][1]
        >= MINIMUM_WEAK_PRESENTATION_DEPTH_METERS
    ][:MAXIMUM_WEAK_PRESENTATION_ENTITIES]
    selected_ids = {entity["id"] for entity in supported_entities + weak_entities}
    return [entity for entity in ranked_entities if entity["id"] in selected_ids][:maximum_entities]


def _validate_entity(entity: dict) -> None:
    required_fields = {"id", "kind", "confidence", "fidelity_tier", "lifecycle", "path_prediction"}
    if not required_fields.issubset(entity):
        raise PlanValidationError(f"Entity is missing required fields: {sorted(required_fields - set(entity))}")
    if entity["kind"] not in RENDERABLE_CLASSES:
        raise PlanValidationError(f"Entity kind is unsupported: {entity['kind']}")
    if not 0.0 <= float(entity["confidence"]) <= 1.0:
        raise PlanValidationError("Entity confidence is invalid")
    path_prediction = entity["path_prediction"]
    if path_prediction.get("method") != "centripetal_catmull_rom":
        raise PlanValidationError("Entity path method must be centripetal_catmull_rom")
    waypoints = path_prediction.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 3:
        raise PlanValidationError("Entity path must contain at least three waypoints")
