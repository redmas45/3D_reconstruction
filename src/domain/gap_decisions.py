"""Validates per-entity gap decisions and applies approved tokens to plans."""

import copy

from domain.reconstruction_plan_v2 import validate_reconstruction_plan_v2


GAP_DECISIONS_SCHEMA_VERSION = 2
MAXIMUM_TEXT_LENGTH = 500
MAXIMUM_LIST_ITEMS = 32
SUPPORTED_BEAT_ACTIONS = frozenset({"continue", "slow", "hold", "enter", "exit", "turn", "occlude"})


class GapDecisionValidationError(ValueError):
    pass


def build_deterministic_gap_decisions(
    evidence_digest: str,
    clue_catalog: dict,
    hypothesis_library: dict,
    reason: str,
) -> dict:
    decisions = [
        _fallback_gap_decision(gap, clue_catalog, reason)
        for gap in hypothesis_library["gaps"]
    ]
    return {
        "schema_version": GAP_DECISIONS_SCHEMA_VERSION,
        "evidence_digest": evidence_digest,
        "clue_digest": clue_catalog["clue_digest"],
        "hypothesis_digest": hypothesis_library["hypothesis_digest"],
        "decisions": decisions,
    }


def validate_gap_decisions(
    value: object,
    evidence_digest: str,
    clue_catalog: dict,
    hypothesis_library: dict,
) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != GAP_DECISIONS_SCHEMA_VERSION:
        raise GapDecisionValidationError("Gap decision schema version is invalid")
    _validate_digest(value, "evidence_digest", evidence_digest)
    _validate_digest(value, "clue_digest", clue_catalog["clue_digest"])
    _validate_digest(value, "hypothesis_digest", hypothesis_library["hypothesis_digest"])
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise GapDecisionValidationError("Gap decisions must be a list")
    gaps = {int(item["gap_index"]): item for item in hypothesis_library["gaps"]}
    clues = {str(item["id"]): item for item in clue_catalog["clues"]}
    validated = [_validate_gap_decision(item, gaps, clues) for item in decisions]
    if {item["gap_index"] for item in validated} != set(gaps) or len(validated) != len(gaps):
        raise GapDecisionValidationError("Gap decisions contain duplicate or missing indexes")
    return {**value, "decisions": sorted(validated, key=lambda item: item["gap_index"])}


def apply_gap_decisions(
    plans: list[dict],
    hypothesis_library: dict,
    gap_decisions: dict,
) -> list[dict]:
    hypotheses = _hypotheses_by_gap_and_entity(hypothesis_library)
    decisions = {int(item["gap_index"]): item for item in gap_decisions["decisions"]}
    updated_plans = []
    for source_plan in plans:
        plan = copy.deepcopy(source_plan)
        gap_index = int(plan["gap_index"])
        entity_decisions = {
            item["entity_id"]: item for item in decisions[gap_index]["entities"]
        }
        for entity in plan.get("entities", []):
            _apply_entity_decision(entity, entity_decisions[str(entity["id"])], hypotheses[gap_index])
        plan["reasoning_decision_v2"] = _plan_decision(decisions[gap_index], gap_decisions)
        plan["reasoning_decision"] = {
            "evidence_digest": gap_decisions["evidence_digest"],
            "selected_hypothesis_id": "per_entity_v2",
            "confidence": decisions[gap_index]["confidence"],
            "decision_summary": decisions[gap_index]["gap_summary"],
            "evidence_references": decisions[gap_index]["evidence_references"],
            "unknowns": decisions[gap_index]["unknowns"],
        }
        validate_reconstruction_plan_v2(plan)
        updated_plans.append(plan)
    return updated_plans


def gap_decisions_json_schema() -> dict:
    text = {"type": "string", "minLength": 1, "maxLength": MAXIMUM_TEXT_LENGTH}
    text_list = {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": text}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "evidence_digest", "clue_digest",
            "hypothesis_digest", "decisions",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": GAP_DECISIONS_SCHEMA_VERSION},
            "evidence_digest": text,
            "clue_digest": text,
            "hypothesis_digest": text,
            "decisions": {"type": "array", "items": _gap_schema(text, text_list)},
        },
    }


def _gap_schema(text: dict, text_list: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "gap_index", "gap_summary", "evidence_references", "clue_ids",
            "confidence", "unknowns", "entities", "event_beats",
        ],
        "properties": {
            "gap_index": {"type": "integer", "minimum": 0},
            "gap_summary": text,
            "evidence_references": text_list,
            "clue_ids": text_list,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "unknowns": text_list,
            "entities": {"type": "array", "items": _entity_schema(text)},
            "event_beats": {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": _beat_schema(text)},
        },
    }


def _entity_schema(text: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "entity_id", "selected_hypothesis_id", "decision_summary",
            "rejected_hypotheses", "confidence",
        ],
        "properties": {
            "entity_id": text,
            "selected_hypothesis_id": text,
            "decision_summary": text,
            "rejected_hypotheses": {
                "type": "array",
                "maxItems": MAXIMUM_LIST_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "reason"],
                    "properties": {"id": text, "reason": text},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def _beat_schema(text: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["time_fraction", "action", "entity_ids"],
        "properties": {
            "time_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "action": {"type": "string", "enum": sorted(SUPPORTED_BEAT_ACTIONS)},
            "entity_ids": {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": text},
        },
    }


def _fallback_gap_decision(gap: dict, clue_catalog: dict, reason: str) -> dict:
    clue_ids = _gap_clue_ids(int(gap["gap_index"]), clue_catalog)
    entities = [_fallback_entity_decision(item) for item in gap["entities"]]
    confidence = sum(item["confidence"] for item in entities) / len(entities) if entities else 0.2
    entity_ids = [item["entity_id"] for item in entities]
    return {
        "gap_index": int(gap["gap_index"]),
        "gap_summary": _fallback_gap_summary(entities),
        "evidence_references": _evidence_references(clue_ids, clue_catalog),
        "clue_ids": clue_ids,
        "confidence": round(confidence, 4),
        "unknowns": [reason],
        "entities": entities,
        "event_beats": [
            {"time_fraction": 0.0, "action": "continue", "entity_ids": entity_ids},
            {"time_fraction": 1.0, "action": "continue", "entity_ids": entity_ids},
        ],
    }


def _fallback_entity_decision(entity: dict) -> dict:
    selected = _fallback_hypothesis(entity["hypotheses"])
    rejected = [
        {"id": item["id"], "reason": "The deterministic fallback selected the strongest bounded prior."}
        for item in entity["hypotheses"] if item["id"] != selected["id"]
    ]
    return {
        "entity_id": str(entity["entity_id"]),
        "selected_hypothesis_id": str(selected["id"]),
        "decision_summary": f"Selected {selected['type']} from visible boundary evidence.",
        "rejected_hypotheses": rejected,
        "confidence": float(selected["prior"]),
    }


def _fallback_hypothesis(hypotheses: list[dict]) -> dict:
    preferred = next(
        (item for item in hypotheses if item["type"] == "identity_unresolved_proxy" and item["prior"] >= 0.5),
        None,
    )
    return preferred or max(hypotheses, key=lambda item: float(item["prior"]))


def _fallback_gap_summary(entities: list[dict]) -> str:
    if not entities:
        return "No supported entity action could be reconstructed for this interval."
    names = ", ".join(item["entity_id"] for item in entities)
    return f"Visible boundary motion supports a conservative continuation for {names}."


def _validate_gap_decision(value: object, gaps: dict[int, dict], clues: dict[str, dict]) -> dict:
    if not isinstance(value, dict) or isinstance(value.get("gap_index"), bool):
        raise GapDecisionValidationError("Each gap decision must be an object")
    gap_index = value.get("gap_index")
    if not isinstance(gap_index, int) or gap_index not in gaps:
        raise GapDecisionValidationError("Decision references an unknown gap")
    clue_ids = _validated_identifier_list(value.get("clue_ids"), "clue IDs")
    if not set(clue_ids).issubset(clues):
        raise GapDecisionValidationError("Decision references an unknown clue")
    references = _validated_identifier_list(value.get("evidence_references"), "evidence references")
    allowed_references = {
        reference for clue_id in clue_ids for reference in clues[clue_id]["evidence_references"]
    }
    if not set(references).issubset(allowed_references):
        raise GapDecisionValidationError("Decision references evidence outside the clue catalog")
    entities = _validate_entities(value.get("entities"), gaps[gap_index])
    beats = _validate_beats(value.get("event_beats"), {item["entity_id"] for item in entities})
    return {
        "gap_index": gap_index,
        "gap_summary": _validated_text(value.get("gap_summary"), "gap summary"),
        "evidence_references": references,
        "clue_ids": clue_ids,
        "confidence": _validated_confidence(value.get("confidence")),
        "unknowns": _validated_identifier_list(value.get("unknowns"), "unknowns"),
        "entities": entities,
        "event_beats": beats,
    }


def _validate_entities(value: object, gap: dict) -> list[dict]:
    if not isinstance(value, list):
        raise GapDecisionValidationError("Gap entity decisions must be a list")
    hypotheses = {
        item["entity_id"]: {hypothesis["id"] for hypothesis in item["hypotheses"]}
        for item in gap["entities"]
    }
    validated = [_validate_entity_decision(item, hypotheses) for item in value]
    identifiers = [item["entity_id"] for item in validated]
    if set(identifiers) != set(hypotheses) or len(identifiers) != len(hypotheses):
        raise GapDecisionValidationError("Gap decision must contain every entity exactly once")
    return validated


def _validate_entity_decision(value: object, hypotheses: dict[str, set[str]]) -> dict:
    if not isinstance(value, dict) or value.get("entity_id") not in hypotheses:
        raise GapDecisionValidationError("Entity decision references an unknown entity")
    entity_id = str(value["entity_id"])
    selected = value.get("selected_hypothesis_id")
    if selected not in hypotheses[entity_id]:
        raise GapDecisionValidationError("Entity decision references an unknown hypothesis")
    rejected = _validate_rejections(value.get("rejected_hypotheses"), hypotheses[entity_id], str(selected))
    return {
        "entity_id": entity_id,
        "selected_hypothesis_id": str(selected),
        "decision_summary": _validated_text(value.get("decision_summary"), "decision summary"),
        "rejected_hypotheses": rejected,
        "confidence": _validated_confidence(value.get("confidence")),
    }


def _validate_rejections(value: object, allowed: set[str], selected: str) -> list[dict]:
    if not isinstance(value, list) or len(value) > MAXIMUM_LIST_ITEMS:
        raise GapDecisionValidationError("Rejected hypotheses must be a bounded list")
    validated = []
    for item in value:
        if not isinstance(item, dict) or item.get("id") not in allowed or item.get("id") == selected:
            raise GapDecisionValidationError("Rejected hypothesis is invalid")
        validated.append({
            "id": str(item["id"]),
            "reason": _validated_text(item.get("reason"), "rejection reason"),
        })
    return validated


def _validate_beats(value: object, entity_ids: set[str]) -> list[dict]:
    if not isinstance(value, list) or len(value) > MAXIMUM_LIST_ITEMS:
        raise GapDecisionValidationError("Event beats must be a bounded list")
    validated = []
    for item in value:
        if not isinstance(item, dict) or item.get("action") not in SUPPORTED_BEAT_ACTIONS:
            raise GapDecisionValidationError("Event beat action is invalid")
        time_fraction = item.get("time_fraction")
        if isinstance(time_fraction, bool) or not isinstance(time_fraction, (int, float)):
            raise GapDecisionValidationError("Event beat time is invalid")
        identifiers = _validated_identifier_list(item.get("entity_ids"), "event entity IDs")
        if not set(identifiers).issubset(entity_ids) or not 0 <= float(time_fraction) <= 1:
            raise GapDecisionValidationError("Event beat references invalid entities or time")
        validated.append({
            "time_fraction": round(float(time_fraction), 4),
            "action": str(item["action"]),
            "entity_ids": identifiers,
        })
    return sorted(validated, key=lambda item: item["time_fraction"])


def _apply_entity_decision(entity: dict, decision: dict, hypotheses: dict[str, dict]) -> None:
    hypothesis = hypotheses[str(entity["id"])][decision["selected_hypothesis_id"]]
    entity["path_prediction"]["waypoints"] = copy.deepcopy(hypothesis["path"])
    entity["animation"]["state"] = "idle" if hypothesis["action"] == "proxy" else hypothesis["action"]
    entity["animation"]["speed_meters_per_second"] = hypothesis["speed_meters_per_second"]
    if hypothesis["action"] == "proxy":
        entity["fidelity_tier"] = "weak"
    entity["reasoning_decision_v2"] = copy.deepcopy(decision)


def _plan_decision(decision: dict, gap_decisions: dict) -> dict:
    return {
        "evidence_digest": gap_decisions["evidence_digest"],
        "clue_digest": gap_decisions["clue_digest"],
        "hypothesis_digest": gap_decisions["hypothesis_digest"],
        "gap_summary": decision["gap_summary"],
        "confidence": decision["confidence"],
        "clue_ids": decision["clue_ids"],
        "unknowns": decision["unknowns"],
        "event_beats": decision["event_beats"],
    }


def _hypotheses_by_gap_and_entity(library: dict) -> dict[int, dict[str, dict[str, dict]]]:
    return {
        int(gap["gap_index"]): {
            entity["entity_id"]: {item["id"]: item for item in entity["hypotheses"]}
            for entity in gap["entities"]
        }
        for gap in library["gaps"]
    }


def _gap_clue_ids(gap_index: int, catalog: dict) -> list[str]:
    prefix = f"gap:{gap_index}"
    scene_ids = [item["id"] for item in catalog["clues"] if item["scope"] == "scene"]
    gap_ids = [item["id"] for item in catalog["clues"] if str(item["scope"]).startswith(prefix)]
    return scene_ids + gap_ids


def _evidence_references(clue_ids: list[str], catalog: dict) -> list[str]:
    selected = {item["id"]: item for item in catalog["clues"]}
    return list(dict.fromkeys(
        reference for clue_id in clue_ids for reference in selected[clue_id]["evidence_references"]
    ))


def _validated_identifier_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAXIMUM_LIST_ITEMS:
        raise GapDecisionValidationError(f"{field_name.title()} must be a bounded list")
    return [_validated_text(item, field_name) for item in value]


def _validated_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAXIMUM_TEXT_LENGTH:
        raise GapDecisionValidationError(f"{field_name.title()} is invalid")
    return value.strip()


def _validated_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise GapDecisionValidationError("Decision confidence must be between zero and one")
    return round(float(value), 4)


def _validate_digest(value: dict, field_name: str, expected: str) -> None:
    if value.get(field_name) != expected:
        raise GapDecisionValidationError(f"Gap decision {field_name} is invalid")
