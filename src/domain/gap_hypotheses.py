"""Builds and validates per-entity reconstruction hypotheses."""

import copy
import hashlib
import json


GAP_HYPOTHESES_SCHEMA_VERSION = 2
SUPPORTED_ACTIONS = frozenset({"walk", "drive", "idle", "proxy"})
SUPPORTED_VISIBILITY = frozenset({
    "visible_throughout", "enters", "exits", "occluded", "uncertain_proxy",
})
HYPOTHESIS_TYPES = (
    "continue_measured_motion",
    "continue_reduced_motion",
    "hold_position",
    "exit_visible_region",
    "enter_visible_region",
    "follow_supported_turn",
    "remain_occluded",
    "identity_unresolved_proxy",
)


class GapHypothesisValidationError(ValueError):
    pass


def build_gap_hypotheses(plans: list[dict], clue_catalog: dict) -> dict:
    clues_by_scope = _clues_by_scope(clue_catalog)
    gaps = []
    for plan in sorted(plans, key=lambda item: int(item["gap_index"])):
        gap_index = int(plan["gap_index"])
        gaps.append({
            "gap_index": gap_index,
            "entities": [
                _entity_hypotheses(gap_index, entity, clues_by_scope)
                for entity in plan.get("entities", [])
            ],
        })
    payload = {
        "schema_version": GAP_HYPOTHESES_SCHEMA_VERSION,
        "clue_digest": clue_catalog["clue_digest"],
        "gaps": gaps,
    }
    payload["hypothesis_digest"] = _canonical_digest(payload)
    return validate_gap_hypotheses(payload)


def validate_gap_hypotheses(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != GAP_HYPOTHESES_SCHEMA_VERSION:
        raise GapHypothesisValidationError("Gap hypothesis schema version is invalid")
    gaps = value.get("gaps")
    if not isinstance(gaps, list):
        raise GapHypothesisValidationError("Gap hypotheses must contain a gap list")
    validated_gaps = [_validate_gap(item) for item in gaps]
    gap_indexes = [item["gap_index"] for item in validated_gaps]
    if len(set(gap_indexes)) != len(gap_indexes):
        raise GapHypothesisValidationError("Gap hypothesis indexes must be unique")
    return {**value, "gaps": validated_gaps}


def _entity_hypotheses(gap_index: int, entity: dict, clues_by_scope: dict[str, list[str]]) -> dict:
    identifier = str(entity["id"])
    scope = f"gap:{gap_index}:entity:{identifier}"
    supported_types = _supported_types(entity)
    hypotheses = [
        _hypothesis(gap_index, entity, hypothesis_type, clues_by_scope.get(scope, []))
        for hypothesis_type in supported_types
    ]
    return {"entity_id": identifier, "kind": str(entity["kind"]), "hypotheses": hypotheses}


def _supported_types(entity: dict) -> list[str]:
    lifecycle = str(entity["lifecycle"])
    types = ["continue_measured_motion", "continue_reduced_motion", "hold_position"]
    if lifecycle == "exits":
        types.append("exit_visible_region")
    if lifecycle == "enters":
        types.append("enter_visible_region")
    if _turn_is_supported(entity):
        types.append("follow_supported_turn")
    if lifecycle == "uncertain":
        types.extend(["remain_occluded", "identity_unresolved_proxy"])
    if float(entity["confidence"]) < 0.5 and "identity_unresolved_proxy" not in types:
        types.append("identity_unresolved_proxy")
    return types


def _hypothesis(gap_index: int, entity: dict, hypothesis_type: str, clue_ids: list[str]) -> dict:
    scale = _motion_scale(hypothesis_type)
    lifecycle = str(entity["lifecycle"])
    return {
        "id": f"gap_{gap_index:02d}_{entity['id']}_{hypothesis_type}",
        "type": hypothesis_type,
        "path": _path_for_type(entity["path_prediction"]["waypoints"], hypothesis_type, scale),
        "action": _action_for_type(entity, hypothesis_type, scale),
        "visibility": _visibility_for_type(lifecycle, hypothesis_type),
        "speed_meters_per_second": round(
            float(entity["animation"]["speed_meters_per_second"]) * scale, 4,
        ),
        "constraints": {
            "uses_post_gap_as_soft_check": True,
            "hard_arrival_constraint": False,
            "source_method": "visible_boundary_motion",
        },
        "prior": _prior(entity, hypothesis_type),
        "supporting_clue_ids": clue_ids,
    }


def _path_for_type(waypoints: list[dict], hypothesis_type: str, scale: float) -> list[dict]:
    if hypothesis_type == "follow_supported_turn":
        return copy.deepcopy(waypoints)
    anchor = waypoints[0]["world"]
    return [
        {
            **waypoint,
            "world": [
                round(anchor[index] + (coordinate - anchor[index]) * scale, 5)
                for index, coordinate in enumerate(waypoint["world"])
            ],
        }
        for waypoint in waypoints
    ]


def _action_for_type(entity: dict, hypothesis_type: str, scale: float) -> str:
    if hypothesis_type == "identity_unresolved_proxy":
        return "proxy"
    if scale == 0.0:
        return "idle"
    return "drive" if str(entity["kind"]) != "person" else "walk"


def _visibility_for_type(lifecycle: str, hypothesis_type: str) -> str:
    if hypothesis_type == "identity_unresolved_proxy":
        return "uncertain_proxy"
    if hypothesis_type == "remain_occluded":
        return "occluded"
    if hypothesis_type == "exit_visible_region" or lifecycle == "exits":
        return "exits"
    if hypothesis_type == "enter_visible_region" or lifecycle == "enters":
        return "enters"
    return "visible_throughout"


def _motion_scale(hypothesis_type: str) -> float:
    if hypothesis_type in {"hold_position", "remain_occluded"}:
        return 0.0
    if hypothesis_type == "continue_reduced_motion":
        return 0.6
    return 1.0


def _prior(entity: dict, hypothesis_type: str) -> float:
    confidence = float(entity["confidence"])
    lifecycle = str(entity["lifecycle"])
    if hypothesis_type == "continue_measured_motion":
        value = confidence
    elif hypothesis_type in {"exit_visible_region", "enter_visible_region"}:
        value = confidence if lifecycle in {"exits", "enters"} else confidence * 0.35
    elif hypothesis_type == "identity_unresolved_proxy":
        value = 1.0 - confidence
    else:
        value = 0.55 * confidence
    return round(max(0.05, min(0.95, value)), 4)


def _turn_is_supported(entity: dict) -> bool:
    disagreement = float(entity["boundary_evidence"].get("heading_disagreement_degrees", 180.0))
    return 8.0 <= disagreement <= 35.0


def _clues_by_scope(catalog: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for clue in catalog["clues"]:
        grouped.setdefault(str(clue["scope"]), []).append(str(clue["id"]))
    return grouped


def _validate_gap(value: object) -> dict:
    if not isinstance(value, dict) or isinstance(value.get("gap_index"), bool):
        raise GapHypothesisValidationError("Each hypothesis gap must have an integer index")
    if not isinstance(value.get("gap_index"), int) or not isinstance(value.get("entities"), list):
        raise GapHypothesisValidationError("Hypothesis gap contract is invalid")
    entities = [_validate_entity(item) for item in value["entities"]]
    identifiers = [item["entity_id"] for item in entities]
    if len(set(identifiers)) != len(identifiers):
        raise GapHypothesisValidationError("Hypothesis entity identifiers must be unique")
    return {**value, "entities": entities}


def _validate_entity(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("entity_id"), str):
        raise GapHypothesisValidationError("Hypothesis entity contract is invalid")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise GapHypothesisValidationError("Each entity requires hypotheses")
    validated = [_validate_hypothesis(item) for item in hypotheses]
    identifiers = [item["id"] for item in validated]
    if len(set(identifiers)) != len(identifiers):
        raise GapHypothesisValidationError("Hypothesis identifiers must be unique")
    return {**value, "hypotheses": validated}


def _validate_hypothesis(value: object) -> dict:
    if not isinstance(value, dict) or value.get("type") not in HYPOTHESIS_TYPES:
        raise GapHypothesisValidationError("Hypothesis type is invalid")
    if value.get("action") not in SUPPORTED_ACTIONS or value.get("visibility") not in SUPPORTED_VISIBILITY:
        raise GapHypothesisValidationError("Hypothesis action or visibility is invalid")
    path = value.get("path")
    if not isinstance(path, list) or len(path) < 3:
        raise GapHypothesisValidationError("Hypothesis path requires at least three waypoints")
    if any(not isinstance(item, dict) or not _valid_world(item.get("world")) for item in path):
        raise GapHypothesisValidationError("Hypothesis waypoint is invalid")
    prior = value.get("prior")
    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not 0 <= prior <= 1:
        raise GapHypothesisValidationError("Hypothesis prior is invalid")
    return value


def _valid_world(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _canonical_digest(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
