"""Builds bounded reconstruction hypotheses and validates reasoning decisions."""

import copy
import hashlib
import json

from domain.motion_profile import synchronize_motion_profile
from domain.reconstruction_plan_v2 import validate_reconstruction_plan_v2


EVIDENCE_LEDGER_SCHEMA_VERSION = 1
DECISION_TRACE_SCHEMA_VERSION = 1
BASELINE_HYPOTHESIS = "measured_continuation"
MOTION_SCALES = {
    BASELINE_HYPOTHESIS: 1.0,
    "reduced_motion": 0.6,
    "stationary_hold": 0.0,
}
HYPOTHESIS_DESCRIPTIONS = {
    BASELINE_HYPOTHESIS: "Continue motion measured immediately before the gap.",
    "reduced_motion": "Continue in the measured direction at a conservative reduced displacement.",
    "stationary_hold": "Keep the entity near its visible boundary position during the gap.",
}
MAXIMUM_TRACE_TEXT_LENGTH = 500
MAXIMUM_TRACE_LIST_ITEMS = 12


class DecisionTraceValidationError(ValueError):
    pass


def build_evidence_ledger(scene_report: dict, plans: list[dict]) -> dict:
    gaps = [_gap_evidence(plan) for plan in sorted(plans, key=lambda item: item["gap_index"])]
    ledger = {
        "schema_version": EVIDENCE_LEDGER_SCHEMA_VERSION,
        "evidence_policy": "visible_frames_and_gap_boundaries_only",
        "scene_clues": _scene_clues(scene_report, gaps),
        "gaps": gaps,
    }
    return {**ledger, "evidence_digest": _canonical_digest(ledger)}


def build_deterministic_decision_trace(ledger: dict, reason: str) -> dict:
    decisions = []
    for gap in ledger["gaps"]:
        decisions.append({
            "gap_index": gap["gap_index"],
            "selected_hypothesis_id": BASELINE_HYPOTHESIS,
            "evidence_references": list(gap["allowed_evidence_references"]),
            "decision_summary": "Used measured visible-boundary motion without an external reasoning decision.",
            "rejected_hypotheses": [
                {"id": item["id"], "reason": "The deterministic fallback preserves the measured baseline."}
                for item in gap["hypotheses"] if item["id"] != BASELINE_HYPOTHESIS
            ],
            "confidence": gap["baseline_confidence"],
            "unknowns": [reason],
        })
    return {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "evidence_digest": ledger["evidence_digest"],
        "decisions": decisions,
    }


def validate_decision_trace(trace: object, ledger: dict) -> dict:
    if not isinstance(trace, dict):
        raise DecisionTraceValidationError("Decision trace must be an object")
    if trace.get("schema_version") != DECISION_TRACE_SCHEMA_VERSION:
        raise DecisionTraceValidationError("Decision trace schema version is invalid")
    if trace.get("evidence_digest") != ledger["evidence_digest"]:
        raise DecisionTraceValidationError("Decision trace does not match the evidence ledger")
    decisions = trace.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(ledger["gaps"]):
        raise DecisionTraceValidationError("Decision trace must contain one decision per gap")
    gaps_by_index = {gap["gap_index"]: gap for gap in ledger["gaps"]}
    validated = [_validate_gap_decision(item, gaps_by_index) for item in decisions]
    if {item["gap_index"] for item in validated} != set(gaps_by_index):
        raise DecisionTraceValidationError("Decision trace contains duplicate or missing gap indexes")
    return {**trace, "decisions": sorted(validated, key=lambda item: item["gap_index"])}


def apply_decision_trace(plans: list[dict], ledger: dict, trace: dict) -> list[dict]:
    gap_evidence = {gap["gap_index"]: gap for gap in ledger["gaps"]}
    decisions = {item["gap_index"]: item for item in trace["decisions"]}
    updated_plans: list[dict] = []
    for original_plan in plans:
        plan = copy.deepcopy(original_plan)
        gap_index = int(plan["gap_index"])
        decision = decisions[gap_index]
        hypothesis = _hypothesis(gap_evidence[gap_index], decision["selected_hypothesis_id"])
        _apply_hypothesis_paths(plan, hypothesis)
        plan["reasoning_decision"] = _plan_decision_contract(decision, ledger["evidence_digest"])
        validate_reconstruction_plan_v2(plan)
        updated_plans.append(plan)
    return updated_plans


def decision_trace_json_schema() -> dict:
    text = {"type": "string", "minLength": 1, "maxLength": MAXIMUM_TRACE_TEXT_LENGTH}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "evidence_digest", "decisions"],
        "properties": {
            "schema_version": {"type": "integer", "const": DECISION_TRACE_SCHEMA_VERSION},
            "evidence_digest": {"type": "string"},
            "decisions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["gap_index", "selected_hypothesis_id", "evidence_references", "decision_summary", "rejected_hypotheses", "confidence", "unknowns"],
                "properties": {
                    "gap_index": {"type": "integer", "minimum": 0},
                    "selected_hypothesis_id": text,
                    "evidence_references": {"type": "array", "maxItems": MAXIMUM_TRACE_LIST_ITEMS, "items": text},
                    "decision_summary": text,
                    "rejected_hypotheses": {"type": "array", "maxItems": MAXIMUM_TRACE_LIST_ITEMS, "items": {
                        "type": "object", "additionalProperties": False, "required": ["id", "reason"],
                        "properties": {"id": text, "reason": text},
                    }},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "unknowns": {"type": "array", "maxItems": MAXIMUM_TRACE_LIST_ITEMS, "items": text},
                },
            }},
        },
    }


def _gap_evidence(plan: dict) -> dict:
    entities = [_entity_evidence(entity) for entity in plan["entities"]]
    references = [f"gap:{plan['gap_index']}:camera_calibration"]
    references.extend(reference for entity in entities for reference in entity["evidence_references"])
    return {
        "gap_index": int(plan["gap_index"]),
        "hidden_range": copy.deepcopy(plan["hidden_range"]),
        "duration_seconds": float(plan["duration_seconds"]),
        "baseline_confidence": float(plan["overall_confidence"]),
        "calibration_confidence": float(plan["camera"]["calibration_confidence"]),
        "entities": entities,
        "allowed_evidence_references": references,
        "hypotheses": [_build_hypothesis(plan, identifier, scale) for identifier, scale in MOTION_SCALES.items()],
    }


def _entity_evidence(entity: dict) -> dict:
    boundary = entity["boundary_evidence"]
    identifier = str(entity["id"])
    references = []
    if boundary.get("before_frame") is not None:
        references.append(f"track:{identifier}:pre_boundary")
    if boundary.get("after_frame") is not None:
        references.append(f"track:{identifier}:post_boundary")
    return {
        "id": identifier,
        "kind": str(entity["kind"]),
        "lifecycle": str(entity["lifecycle"]),
        "confidence": float(entity["confidence"]),
        "speed_meters_per_second": float(entity["animation"]["speed_meters_per_second"]),
        "before_frame": boundary.get("before_frame"),
        "after_frame": boundary.get("after_frame"),
        "pre_gap_heading_degrees": boundary.get("pre_gap_heading_degrees"),
        "post_gap_heading_degrees": boundary.get("post_gap_heading_degrees"),
        "heading_disagreement_degrees": float(boundary["heading_disagreement_degrees"]),
        "post_gap_position_residual_meters": boundary.get("post_gap_position_residual_meters"),
        "evidence_references": references,
    }


def _build_hypothesis(plan: dict, identifier: str, displacement_scale: float) -> dict:
    paths = {
        str(entity["id"]): _scaled_waypoints(entity["path_prediction"]["waypoints"], displacement_scale)
        for entity in plan["entities"]
    }
    return {
        "id": identifier,
        "description": HYPOTHESIS_DESCRIPTIONS[identifier],
        "displacement_scale": displacement_scale,
        "entity_paths": paths,
    }


def _scaled_waypoints(waypoints: list[dict], scale: float) -> list[dict]:
    anchor = waypoints[0]["world"]
    return [
        {**waypoint, "world": [round(anchor[index] + (coordinate - anchor[index]) * scale, 5) for index, coordinate in enumerate(waypoint["world"])]}
        for waypoint in waypoints
    ]


def _scene_clues(scene_report: dict, gaps: list[dict]) -> list[str]:
    track_count = len(scene_report.get("tracks", []))
    entity_count = sum(len(gap["entities"]) for gap in gaps)
    camera = scene_report.get("camera_motion_report", {})
    motion_mode = str(camera.get("mode", camera.get("classification", "unknown")))
    return [
        f"{track_count} visible-evidence tracks were analyzed.",
        f"{entity_count} gap-entity appearances have boundary support.",
        f"Camera motion assessment: {motion_mode}.",
        "Hidden frames were excluded from reasoning evidence.",
    ]


def _validate_gap_decision(value: object, gaps_by_index: dict[int, dict]) -> dict:
    if not isinstance(value, dict) or isinstance(value.get("gap_index"), bool):
        raise DecisionTraceValidationError("Each gap decision must be an object with an integer index")
    gap_index = value.get("gap_index")
    if not isinstance(gap_index, int) or gap_index not in gaps_by_index:
        raise DecisionTraceValidationError("Decision references an unknown gap")
    gap = gaps_by_index[gap_index]
    selected = value.get("selected_hypothesis_id")
    hypothesis_ids = {item["id"] for item in gap["hypotheses"]}
    if selected not in hypothesis_ids:
        raise DecisionTraceValidationError("Decision references an unknown hypothesis")
    references = _validated_string_list(value.get("evidence_references"), "evidence references")
    if not set(references).issubset(set(gap["allowed_evidence_references"])):
        raise DecisionTraceValidationError("Decision references evidence that was not supplied")
    rejected = _validated_rejections(value.get("rejected_hypotheses"), hypothesis_ids, str(selected))
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise DecisionTraceValidationError("Decision confidence must be between zero and one")
    return {
        "gap_index": gap_index,
        "selected_hypothesis_id": selected,
        "evidence_references": references,
        "decision_summary": _validated_text(value.get("decision_summary"), "decision summary"),
        "rejected_hypotheses": rejected,
        "confidence": round(float(confidence), 4),
        "unknowns": _validated_string_list(value.get("unknowns"), "unknowns"),
    }


def _validated_rejections(value: object, hypotheses: set[str], selected: str) -> list[dict]:
    if not isinstance(value, list) or len(value) > MAXIMUM_TRACE_LIST_ITEMS:
        raise DecisionTraceValidationError("Rejected hypotheses must be a bounded list")
    validated: list[dict] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise DecisionTraceValidationError("Rejected hypothesis entries must be objects")
        identifier = _validated_text(item.get("id"), "rejected hypothesis id")
        if identifier not in hypotheses or identifier == selected or identifier in seen:
            raise DecisionTraceValidationError("Rejected hypothesis id is invalid")
        seen.add(identifier)
        validated.append({"id": identifier, "reason": _validated_text(item.get("reason"), "rejection reason")})
    return validated


def _validated_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAXIMUM_TRACE_LIST_ITEMS:
        raise DecisionTraceValidationError(f"{field_name.title()} must be a bounded list")
    return [_validated_text(item, field_name) for item in value]


def _validated_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAXIMUM_TRACE_TEXT_LENGTH:
        raise DecisionTraceValidationError(f"{field_name.title()} is invalid")
    return value.strip()


def _hypothesis(gap: dict, identifier: str) -> dict:
    return next(item for item in gap["hypotheses"] if item["id"] == identifier)


def _apply_hypothesis_paths(plan: dict, hypothesis: dict) -> None:
    for entity in plan["entities"]:
        entity["path_prediction"]["waypoints"] = copy.deepcopy(hypothesis["entity_paths"][str(entity["id"])])
        entity["animation"]["speed_meters_per_second"] = round(
            float(entity["animation"]["speed_meters_per_second"]) * hypothesis["displacement_scale"], 4,
        )
        if hypothesis["displacement_scale"] == 0:
            entity["animation"]["state"] = "idle"
        synchronize_motion_profile(entity)


def _plan_decision_contract(decision: dict, evidence_digest: str) -> dict:
    return {
        "evidence_digest": evidence_digest,
        "selected_hypothesis_id": decision["selected_hypothesis_id"],
        "confidence": decision["confidence"],
        "decision_summary": decision["decision_summary"],
        "evidence_references": decision["evidence_references"],
        "unknowns": decision["unknowns"],
    }


def _canonical_digest(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
