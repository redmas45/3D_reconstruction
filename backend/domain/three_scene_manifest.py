"""Builds the Three.js scene manifest from validated reconstruction artifacts.

This manifest is the single contract the browser renderer consumes. The renderer
never invents entities, coordinates, or paths: everything it draws is assembled
here from artifacts that were themselves derived only from the visible 75% of the
source video (reconstruction plans, validated gap decisions, the public narrative,
and the clue catalog).

Coordinate convention carried through to the browser is the domain world frame:
X = right, Y = forward/depth, Z = up, in metres, with the ground plane at Z = 0.
Entity waypoints are world-metre points; the browser maps them into its own
Y-up frame and projects them through the calibrated camera. Nothing in this
manifest references a hidden (inside-gap) frame — see three_scene_validation.
"""

from math import atan2, degrees

from domain.actor_proxies import proxy_for
from domain.three_scene_validation import validate_three_scene_manifest


SCENE_MANIFEST_SCHEMA_VERSION = 1
RENDERER_NAME = "threejs"
WORLD_FRAME = "z_up_right_handed_metres"
GROUND_PLANE_Z = 0.0
EVIDENCE_DISCLOSURE = (
    "Every figure and path shown is an inference bounded by the visible 75% of the "
    "video. This is a transparent hypothesis, not recovered ground truth."
)
MAXIMUM_ENTITIES_PER_GAP = 24
MAXIMUM_CLUES_PER_GAP = 8
MAXIMUM_SCENE_CLUES = 10
MAXIMUM_REJECTED_PER_ENTITY = 4
MAXIMUM_EVIDENCE_REFERENCES = 8
# The renderer promotes fidelity into geometry detail and animation richness. These
# are the only three tiers the browser understands; anything else collapses to weak.
SUPPORTED_FIDELITY_TIERS = ("supported", "plausible", "weak")
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})


def build_three_scene_manifest(
    video_info: dict,
    gap_selection: dict,
    scene_report: dict,
    plans: list[dict],
    reasoning: dict,
    hypotheses: dict,
) -> dict:
    """Assemble and validate the browser scene manifest.

    All inputs are plain dicts already loaded from validated artifacts, so this
    function is pure and unit-testable without touching the filesystem.
    """
    fps = float(video_info["fps"])
    frame_count = int(video_info["frames"])
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    observed_fraction = round(1.0 - float(gap_selection["missing_fraction_actual"]), 4)
    plans_by_gap = {int(plan["gap_index"]): plan for plan in plans}
    decisions_by_gap = _index_by_gap_index(reasoning.get("decisions", []))
    summaries_by_gap = _index_by_gap_index(reasoning.get("gap_summaries", []))
    hypotheses_by_id = _index_hypotheses(hypotheses)
    manifest = {
        "schema_version": SCENE_MANIFEST_SCHEMA_VERSION,
        "renderer": RENDERER_NAME,
        "evidence_disclosure": EVIDENCE_DISCLOSURE,
        "world": {
            "frame": WORLD_FRAME,
            "ground_plane_z": GROUND_PLANE_Z,
            "units": "metres",
        },
        "source": {
            "video_name": str(scene_report.get("video", {}).get("name", video_info.get("name", "video"))),
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": round(duration_seconds, 3),
            "observed_fraction": observed_fraction,
            "reconstructed_fraction": round(float(gap_selection["missing_fraction_actual"]), 4),
        },
        "camera_default": _camera_contract(plans[0]["camera"]) if plans else _fallback_camera(),
        "narrative": _narrative_contract(reasoning),
        "clues": _scene_clues(reasoning),
        "gaps": [
            _gap_contract(
                index, hidden_range, fps, plans_by_gap.get(index, {}),
                decisions_by_gap.get(index, {}), summaries_by_gap.get(index, {}),
                hypotheses_by_id,
            )
            for index, hidden_range in enumerate(gap_selection["hidden_ranges"])
        ],
    }
    validate_three_scene_manifest(manifest)
    return manifest


def _gap_contract(
    gap_index: int,
    hidden_range: list[int],
    fps: float,
    plan: dict,
    decision: dict,
    summary: dict,
    hypotheses_by_id: dict[str, dict],
) -> dict:
    start_frame, end_frame = int(hidden_range[0]), int(hidden_range[1])
    duration_frames = end_frame - start_frame + 1
    camera = _camera_contract(plan.get("camera", {}))
    confidence = _first_number(
        summary.get("confidence"), decision.get("confidence"), plan.get("overall_confidence"),
    )
    return {
        "gap_id": f"gap_{gap_index:02d}",
        "gap_index": gap_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": round(start_frame / fps, 3),
        "end_seconds": round((end_frame + 1) / fps, 3),
        "duration_seconds": round(duration_frames / fps, 3),
        "confidence": round(confidence, 4),
        "calibration_confidence": camera["calibration_confidence"],
        "camera": camera,
        "environment": _environment_contract(plan, start_frame, end_frame),
        "narrative": {
            "before_observed": str(summary.get("before_observed", "Visible evidence before the interval.")),
            "inside_inferred": str(summary.get(
                "inside_inferred", decision.get("gap_summary", "Bounded motion inferred from the boundaries."),
            )),
            "after_observed": str(summary.get("after_observed", "Visible evidence after the interval.")),
            "unknowns": _string_list(summary.get("unknowns", decision.get("unknowns", []))),
        },
        "decision": {
            "gap_summary": str(decision.get("gap_summary", summary.get("inside_inferred", ""))),
            "confidence": round(_first_number(decision.get("confidence"), confidence), 4),
            "evidence_references": _string_list(
                decision.get("evidence_references", []),
            )[:MAXIMUM_EVIDENCE_REFERENCES],
            "unknowns": _string_list(decision.get("unknowns", [])),
            "event_beats": _event_beats(decision.get("event_beats", [])),
        },
        "clues": _gap_clues(decision),
        "entities": _gap_entities(plan, decision, hypotheses_by_id, start_frame, end_frame),
    }


def _gap_entities(
    plan: dict,
    decision: dict,
    hypotheses_by_id: dict[str, dict],
    start_frame: int,
    end_frame: int,
) -> list[dict]:
    decisions_by_entity = {
        str(item.get("entity_id")): item
        for item in decision.get("entities", [])
        if isinstance(item, dict)
    }
    entities = [
        _entity_contract(entity, decisions_by_entity.get(str(entity["id"]), {}), hypotheses_by_id, start_frame, end_frame)
        for entity in plan.get("entities", [])
        if isinstance(entity, dict)
    ]
    return entities[:MAXIMUM_ENTITIES_PER_GAP]


def _entity_contract(
    entity: dict,
    entity_decision: dict,
    hypotheses_by_id: dict[str, dict],
    start_frame: int,
    end_frame: int,
) -> dict:
    class_name = str(entity["kind"])
    selected_id = str(entity_decision.get("selected_hypothesis_id", ""))
    hypothesis = hypotheses_by_id.get(selected_id, {})
    raw_waypoints = _select_waypoints(entity, hypothesis)
    waypoints = _waypoint_contract(raw_waypoints)
    proxy = proxy_for(class_name)
    return {
        "track_id": str(entity["id"]),
        "class_name": class_name,
        "category": "vehicle" if class_name in VEHICLE_CLASSES else "person",
        "lifecycle": str(entity.get("lifecycle", "uncertain")),
        "confidence": round(float(entity.get("confidence", 0.0)), 4),
        "visual_fidelity_tier": _fidelity_tier(entity),
        "appearance_seed": _appearance_seed(str(entity["id"])),
        "appearance": _appearance_contract(entity.get("appearance", {})),
        "body_proportions": _body_proportions_contract(entity.get("body_proportions", {})),
        "visual_anchor": _visual_anchor_contract(entity.get("visual_anchor", {})),
        "proxy": {
            "type": proxy.proxy,
            "dimensions_metres": [proxy.length, proxy.width, proxy.height],
            "ground_offset_metres": proxy.ground_offset_meters,
        },
        "start_state": _boundary_state(waypoints, "start"),
        "end_state": _boundary_state(waypoints, "end"),
        "waypoints": waypoints,
        "motion_profile": _motion_profile_contract(entity),
        "kinematics": _kinematics_contract(entity.get("kinematics", {})),
        "occlusion_state": _occlusion_state(hypothesis),
        "selected_hypothesis": {
            "id": selected_id,
            "type": str(hypothesis.get("type", "continue_measured_motion")),
            "action": str(hypothesis.get("action", entity.get("animation", {}).get("state", "walk"))),
            "selection_score": round(float(hypothesis.get("selection_score", 0.0)), 4),
            "score_components": hypothesis.get("score_components", {}),
            "selection_source": str(entity_decision.get("selection_source", "azure_reasoner")),
        },
        "rejected_hypotheses": _rejected_hypotheses(entity_decision),
        "decision_summary": str(entity_decision.get("decision_summary", "Bounded hypothesis selected.")),
        "uncertainty": _uncertainty_contract(entity),
    }


def _select_waypoints(entity: dict, hypothesis: dict) -> list[dict]:
    """Prefer the selected hypothesis's validated path; fall back to the prediction.

    The hypothesis path is what the decision actually committed to (a held position
    collapses to a single anchor, a reduced-motion path is scaled). When no decision
    resolved, the raw forward prediction is the honest default.
    """
    hypothesis_path = hypothesis.get("path")
    if isinstance(hypothesis_path, list) and len(hypothesis_path) >= 3:
        return hypothesis_path
    prediction = entity.get("path_prediction", {})
    waypoints = prediction.get("waypoints")
    return waypoints if isinstance(waypoints, list) else []


def _waypoint_contract(raw_waypoints: list[dict]) -> list[dict]:
    ordered = [item for item in raw_waypoints if isinstance(item, dict) and _has_world(item)]
    if not ordered:
        return []
    first_frame = int(ordered[0].get("frame", 0))
    last_frame = int(ordered[-1].get("frame", first_frame))
    frame_span = max(1, last_frame - first_frame)
    waypoints = []
    for index, item in enumerate(ordered):
        world = [float(value) for value in item["world"][:3]]
        frame = int(item.get("frame", first_frame))
        waypoints.append({
            "role": str(item.get("role", _default_role(index, len(ordered)))),
            "frame": frame,
            "time_fraction": round(min(1.0, max(0.0, (frame - first_frame) / frame_span)), 5),
            "world": [round(world[0], 5), round(world[1], 5), round(world[2], 5)],
            "heading_degrees": _heading_at(ordered, index),
        })
    return waypoints


def _heading_at(waypoints: list[dict], index: int) -> float:
    """Facing derived from the ground-path tangent (world X-forward is +Y).

    Uses the outgoing segment except at the final point, which reuses the incoming
    one, so a figure never snaps to a default heading at the end of its path.
    """
    if len(waypoints) < 2:
        return 0.0
    if index >= len(waypoints) - 1:
        near, far = waypoints[index - 1]["world"], waypoints[index]["world"]
    else:
        near, far = waypoints[index]["world"], waypoints[index + 1]["world"]
    delta_x = float(far[0]) - float(near[0])
    delta_y = float(far[1]) - float(near[1])
    if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
        return 0.0
    return round(degrees(atan2(delta_x, delta_y)) % 360.0, 3)


def _boundary_state(waypoints: list[dict], which: str) -> dict:
    if not waypoints:
        return {"frame": 0, "world": [0.0, 0.0, GROUND_PLANE_Z], "heading_degrees": 0.0}
    waypoint = waypoints[0] if which == "start" else waypoints[-1]
    return {
        "frame": waypoint["frame"],
        "world": waypoint["world"],
        "heading_degrees": waypoint["heading_degrees"],
    }


def _camera_contract(camera: dict) -> dict:
    return {
        "projection_model": str(camera.get("projection_model", "pinhole_ground_plane_v2")),
        "position": _vector3(camera.get("position", [0.0, 0.0, 1.6])),
        "look_at": _vector3(camera.get("look_at", [0.0, 10.0, 1.6])),
        "field_of_view_degrees": round(float(camera.get("field_of_view_degrees", 58.0)), 4),
        "field_of_view_axis": "horizontal",
        "horizon_normalized_y": round(float(camera.get("horizon_normalized_y", 0.5)), 5),
        "focal_length_mm": round(float(camera.get("focal_length_mm", 35.0)), 4),
        "motion_model": str(camera.get("motion_model", "unclassified")),
        "presentation_mode": str(camera.get("presentation_mode", "stabilized_forensic_view")),
        "calibration_confidence": round(float(camera.get("calibration_confidence", 0.0)), 4),
        "motion_warning": _camera_motion_warning(camera),
    }


def _camera_motion_warning(camera: dict) -> str:
    if camera.get("motion_model") == "static_camera":
        return ""
    return (
        "Camera motion is unverified for this interval; entity placement is shown in a "
        "stabilized forensic view and its geometry is treated as lower confidence."
    )


def _environment_contract(plan: dict, start_frame: int, end_frame: int) -> dict:
    environment = plan.get("environment", {})
    backplate_frame = int(environment.get("backplate_frame", start_frame - 1))
    # The backplate is the last visible frame *before* the gap. If for any reason it
    # would land inside the hidden interval, drop it rather than sample hidden footage.
    has_backplate = bool(environment.get("hybrid_backplate_enabled")) and not (
        start_frame <= backplate_frame <= end_frame
    )
    return {
        "style": str(environment.get("style", "forensic_3d")),
        "ground_color": _color(environment.get("ground_color", [0.035, 0.047, 0.062])),
        "grid_color": _color(environment.get("grid_color", [0.04, 0.62, 0.68])),
        "backplate_frame": backplate_frame if has_backplate else None,
        "has_backplate": has_backplate,
        "backplate_reason": str(environment.get("hybrid_backplate_reason", "")),
    }


def _narrative_contract(reasoning: dict) -> dict:
    return {
        "headline": str(reasoning.get("headline", "Evidence-grounded reconstruction")),
        "whole_video_summary": str(reasoning.get("whole_video_summary", "")),
        "confidence": round(_first_number(reasoning.get("confidence")), 4),
        "causal_link_supported": bool(reasoning.get("causal_link_supported", False)),
        "planning_mode": str(reasoning.get("mode", "unknown")),
        "unknowns": _string_list(reasoning.get("unknowns", [])),
        "story_points": [
            str(item["statement"])
            for item in reasoning.get("story_points", [])
            if isinstance(item, dict) and isinstance(item.get("statement"), str)
        ],
    }


def _scene_clues(reasoning: dict) -> list[dict]:
    clues = [
        item for item in reasoning.get("clues", [])
        if isinstance(item, dict) and isinstance(item.get("statement"), str)
    ]
    ranked = sorted(clues, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return [
        {
            "id": str(item.get("id", "")),
            "category": str(item.get("category", "evidence")),
            "statement": str(item["statement"]),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
        }
        for item in ranked[:MAXIMUM_SCENE_CLUES]
    ]


def _gap_clues(decision: dict) -> list[dict]:
    return [
        {"id": str(clue_id)}
        for clue_id in _string_list(decision.get("clue_ids", []))[:MAXIMUM_CLUES_PER_GAP]
    ]


def _event_beats(beats: object) -> list[dict]:
    if not isinstance(beats, list):
        return []
    return [
        {
            "time_fraction": round(float(item.get("time_fraction", 0.0)), 4),
            "action": str(item.get("action", "continue")),
            "entity_ids": _string_list(item.get("entity_ids", [])),
        }
        for item in beats
        if isinstance(item, dict)
    ]


def _rejected_hypotheses(entity_decision: dict) -> list[dict]:
    rejected = entity_decision.get("rejected_hypotheses", [])
    if not isinstance(rejected, list):
        return []
    return [
        {
            "id": str(item.get("id", "alternative")),
            "reason": str(item.get("reason", "Less supported by visible evidence.")),
        }
        for item in rejected[:MAXIMUM_REJECTED_PER_ENTITY]
        if isinstance(item, dict)
    ]


def _appearance_contract(appearance: dict) -> dict:
    upper = _color(appearance.get("upper_color", [0.13, 0.48, 0.54]))
    lower = _color(appearance.get("lower_color", [0.16, 0.19, 0.24]))
    return {
        "upper_color": upper,
        "lower_color": lower,
        "vehicle_color": _color(appearance.get("vehicle_color", upper)),
        "source": str(appearance.get("source", "deterministic_fallback")),
    }


def _body_proportions_contract(proportions: dict) -> dict:
    return {
        "height_scale": round(float(proportions.get("height_scale", 1.0)), 4),
        "shoulder_scale": round(float(proportions.get("shoulder_scale", 1.0)), 4),
        "limb_scale": round(float(proportions.get("limb_scale", 1.0)), 4),
    }


def _visual_anchor_contract(anchor: object) -> dict:
    """Carry only visible boundary size metadata to the browser renderer.

    The anchor is a measured bbox from an evidence frame, not a hidden position. It
    lets the browser match an actor's apparent height before the inferred path begins
    instead of assuming every person is the same pixel size.
    """
    if not isinstance(anchor, dict):
        return {}
    contract: dict = {}
    for field_name in (
        "center_x_fraction", "ground_y_fraction", "width_fraction", "height_fraction",
    ):
        value = anchor.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            contract[field_name] = round(max(0.0, min(1.0, float(value))), 5)
    source_frame = anchor.get("source_frame")
    if isinstance(source_frame, int) and not isinstance(source_frame, bool) and source_frame >= 0:
        contract["source_frame"] = source_frame
    return contract


def _motion_profile_contract(entity: dict) -> dict:
    profile = entity.get("motion_profile")
    animation = entity.get("animation", {})
    speed = float(animation.get("speed_meters_per_second", 0.0))
    if not isinstance(profile, dict):
        return {
            "clip": "idle" if speed < 0.2 else "walk",
            "phase_offset": round(float(animation.get("phase_offset", 0.0)), 4),
            "cadence_scale": 1.0,
            "blend_seconds": 0.18,
            "pose_confidence": 0.0,
            "speed_meters_per_second": round(speed, 4),
        }
    return {
        "clip": str(profile.get("clip", "walk")),
        "phase_offset": round(float(profile.get("phase_offset", 0.0)), 4),
        "cadence_scale": round(float(profile.get("cadence_scale", 1.0)), 4),
        "blend_seconds": round(float(profile.get("blend_seconds", 0.18)), 4),
        "pose_confidence": round(float(profile.get("pose_confidence", 0.0)), 4),
        "speed_meters_per_second": round(speed, 4),
    }


def _kinematics_contract(kinematics: dict) -> dict:
    return {
        "duration_seconds": round(float(kinematics.get("duration_seconds", 0.0)), 4),
        "maximum_speed_meters_per_second": round(
            float(kinematics.get("maximum_speed_meters_per_second", 8.0)), 4),
        "maximum_turn_rate_degrees_per_second": round(
            float(kinematics.get("maximum_turn_rate_degrees_per_second", 45.0)), 4),
    }


def _uncertainty_contract(entity: dict) -> dict:
    uncertainty = entity.get("uncertainty", {})
    boundary = entity.get("boundary_evidence", {})
    return {
        "position_radius_metres": round(float(uncertainty.get("position_radius_meters", 0.5)), 4),
        "alternative_paths": int(uncertainty.get("alternative_paths", 0)),
        "heading_disagreement_degrees": round(
            float(boundary.get("heading_disagreement_degrees", 0.0)), 3),
    }


def _occlusion_state(hypothesis: dict) -> str:
    visibility = str(hypothesis.get("visibility", ""))
    if visibility:
        return visibility
    if hypothesis.get("type") == "remain_occluded":
        return "occluded"
    return "uncertain_proxy" if hypothesis.get("action") == "proxy" else "visible_throughout"


def _fidelity_tier(entity: dict) -> str:
    tier = str(entity.get("fidelity_tier", "weak"))
    return tier if tier in SUPPORTED_FIDELITY_TIERS else "weak"


def _appearance_seed(track_id: str) -> int:
    """Stable per-track integer seed so the browser can vary silhouette detail.

    Derived from the track id alone, so the same tracked entity seeds identically in
    every gap it appears in — a hard requirement for consistent appearance.
    """
    seed = 0
    for character in track_id:
        seed = (seed * 131 + ord(character)) & 0xFFFFFFFF
    return seed


def _index_by_gap_index(items: object) -> dict[int, dict]:
    if not isinstance(items, list):
        return {}
    return {
        int(item["gap_index"]): item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("gap_index"), int)
    }


def _index_hypotheses(hypotheses: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for gap in hypotheses.get("gaps", []) if isinstance(hypotheses, dict) else []:
        if not isinstance(gap, dict):
            continue
        for entity in gap.get("entities", []):
            if not isinstance(entity, dict):
                continue
            for hypothesis in entity.get("hypotheses", []):
                if isinstance(hypothesis, dict) and isinstance(hypothesis.get("id"), str):
                    index[hypothesis["id"]] = hypothesis
    return index


def _has_world(waypoint: dict) -> bool:
    world = waypoint.get("world")
    return isinstance(world, list) and len(world) >= 3


def _default_role(index: int, count: int) -> str:
    if index == 0:
        return "start"
    if index == count - 1:
        return "predicted_end"
    return f"inferred_{index:02d}"


def _color(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        return [0.5, 0.5, 0.5]
    return [round(min(1.0, max(0.0, float(channel))), 4) for channel in value[:3]]


def _vector3(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        return [0.0, 0.0, 0.0]
    return [round(float(component), 5) for component in value[:3]]


def _fallback_camera() -> dict:
    return _camera_contract({})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _first_number(*values: object) -> float:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0
