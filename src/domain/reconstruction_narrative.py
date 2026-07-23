"""Builds and validates the judge-facing reconstruction narrative."""


NARRATIVE_SCHEMA_VERSION = 1
MAXIMUM_TEXT_LENGTH = 700
MAXIMUM_LIST_ITEMS = 50


class NarrativeValidationError(ValueError):
    pass


def build_deterministic_narrative(
    clue_catalog: dict,
    gap_decisions: dict,
    mode: str,
    warning: str | None,
) -> dict:
    decisions = gap_decisions["decisions"]
    summaries = [item["gap_summary"] for item in decisions]
    average_confidence = (
        sum(float(item["confidence"]) for item in decisions) / len(decisions)
        if decisions else 0.0
    )
    return {
        "schema_version": NARRATIVE_SCHEMA_VERSION,
        "headline": "Evidence-grounded reconstruction of missing intervals",
        "whole_video_summary": _whole_video_summary(summaries),
        "story_points": _story_points(decisions),
        "gap_summaries": [_gap_summary(item) for item in decisions],
        "confidence": round(average_confidence, 4),
        "unknowns": _combined_unknowns(decisions, warning),
        "causal_link_supported": False,
        "mode": mode,
        "clue_digest": clue_catalog["clue_digest"],
    }


def validate_narrative(
    value: object,
    clue_catalog: dict,
    gap_decisions: dict,
    mode: str,
) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != NARRATIVE_SCHEMA_VERSION:
        raise NarrativeValidationError("Narrative schema version is invalid")
    if value.get("clue_digest") != clue_catalog["clue_digest"]:
        raise NarrativeValidationError("Narrative does not match the clue catalog")
    clue_ids = {item["id"] for item in clue_catalog["clues"]}
    gap_indexes = {item["gap_index"] for item in gap_decisions["decisions"]}
    validated = {
        **value,
        "headline": _text(value.get("headline"), "headline"),
        "whole_video_summary": _text(value.get("whole_video_summary"), "whole-video summary"),
        "story_points": _validate_story_points(value.get("story_points"), clue_ids, gap_indexes),
        "gap_summaries": _validate_gap_summaries(value.get("gap_summaries"), gap_indexes),
        "confidence": _confidence(value.get("confidence")),
        "unknowns": _text_list(value.get("unknowns"), "unknowns"),
        "mode": mode,
    }
    if not isinstance(value.get("causal_link_supported"), bool):
        raise NarrativeValidationError("Narrative causal support flag must be boolean")
    if value["causal_link_supported"] and any(
        not item["clue_ids"] for item in validated["story_points"]
    ):
        raise NarrativeValidationError("Causal narrative points require explicit clue references")
    return validated


def narrative_json_schema() -> dict:
    text = {"type": "string", "minLength": 1, "maxLength": MAXIMUM_TEXT_LENGTH}
    text_list = {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": text}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "headline", "whole_video_summary", "story_points",
            "gap_summaries", "confidence", "unknowns", "causal_link_supported",
            "mode", "clue_digest",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": NARRATIVE_SCHEMA_VERSION},
            "headline": text,
            "whole_video_summary": text,
            "story_points": {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "clue_ids", "gap_indexes"],
                "properties": {
                    "statement": text,
                    "clue_ids": text_list,
                    "gap_indexes": {
                        "type": "array",
                        "maxItems": MAXIMUM_LIST_ITEMS,
                        "items": {"type": "integer", "minimum": 0},
                    },
                },
            }},
            "gap_summaries": {"type": "array", "maxItems": MAXIMUM_LIST_ITEMS, "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "gap_index", "before_observed", "inside_inferred",
                    "after_observed", "confidence", "unknowns",
                ],
                "properties": {
                    "gap_index": {"type": "integer", "minimum": 0},
                    "before_observed": text,
                    "inside_inferred": text,
                    "after_observed": text,
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "unknowns": text_list,
                },
            }},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "unknowns": text_list,
            "causal_link_supported": {"type": "boolean"},
            "mode": text,
            "clue_digest": text,
        },
    }


def narrative_request_payload(clue_catalog: dict, gap_decisions: dict) -> dict:
    return {
        "clue_digest": clue_catalog["clue_digest"],
        "clues": clue_catalog["clues"],
        "gap_decisions": gap_decisions["decisions"],
        "presentation_rules": {
            "describe_inside_gap_as_inferred": True,
            "do_not_claim_ground_truth": True,
            "causal_links_require_explicit_clue_support": True,
        },
    }


def _whole_video_summary(summaries: list[str]) -> str:
    if not summaries:
        return "The visible evidence did not support a specific missing-interval reconstruction."
    return " ".join(summaries)


def _story_points(decisions: list[dict]) -> list[dict]:
    return [
        {
            "statement": item["gap_summary"],
            "clue_ids": item["clue_ids"],
            "gap_indexes": [item["gap_index"]],
        }
        for item in decisions
    ]


def _gap_summary(decision: dict) -> dict:
    entity_names = ", ".join(item["entity_id"] for item in decision["entities"]) or "no supported entity"
    return {
        "gap_index": decision["gap_index"],
        "before_observed": f"Visible pre-gap evidence identifies {entity_names}.",
        "inside_inferred": decision["gap_summary"],
        "after_observed": "Visible post-gap evidence was used as a consistency check, not a hard path target.",
        "confidence": decision["confidence"],
        "unknowns": decision["unknowns"],
    }


def _combined_unknowns(decisions: list[dict], warning: str | None) -> list[str]:
    unknowns = list(dict.fromkeys(item for decision in decisions for item in decision["unknowns"]))
    if warning and warning not in unknowns:
        unknowns.append(warning)
    return unknowns[:MAXIMUM_LIST_ITEMS]


def _validate_story_points(value: object, clue_ids: set[str], gap_indexes: set[int]) -> list[dict]:
    if not isinstance(value, list) or len(value) > MAXIMUM_LIST_ITEMS:
        raise NarrativeValidationError("Narrative story points must be a bounded list")
    validated = []
    for item in value:
        if not isinstance(item, dict):
            raise NarrativeValidationError("Narrative story point must be an object")
        references = _text_list(item.get("clue_ids"), "story point clue IDs")
        indexes = item.get("gap_indexes")
        if not isinstance(indexes, list) or not set(indexes).issubset(gap_indexes):
            raise NarrativeValidationError("Narrative story point references an unknown gap")
        if not set(references).issubset(clue_ids):
            raise NarrativeValidationError("Narrative story point references an unknown clue")
        validated.append({
            "statement": _text(item.get("statement"), "story point"),
            "clue_ids": references,
            "gap_indexes": indexes,
        })
    return validated


def _validate_gap_summaries(value: object, gap_indexes: set[int]) -> list[dict]:
    if not isinstance(value, list) or len(value) != len(gap_indexes):
        raise NarrativeValidationError("Narrative must summarize every gap")
    validated = [_validate_gap_summary(item, gap_indexes) for item in value]
    indexes = [item["gap_index"] for item in validated]
    if set(indexes) != gap_indexes or len(indexes) != len(gap_indexes):
        raise NarrativeValidationError("Narrative gap summaries contain duplicate or missing gaps")
    return sorted(validated, key=lambda item: item["gap_index"])


def _validate_gap_summary(value: object, gap_indexes: set[int]) -> dict:
    if not isinstance(value, dict) or value.get("gap_index") not in gap_indexes:
        raise NarrativeValidationError("Narrative gap summary references an unknown gap")
    return {
        "gap_index": int(value["gap_index"]),
        "before_observed": _text(value.get("before_observed"), "before observation"),
        "inside_inferred": _text(value.get("inside_inferred"), "inside inference"),
        "after_observed": _text(value.get("after_observed"), "after observation"),
        "confidence": _confidence(value.get("confidence")),
        "unknowns": _text_list(value.get("unknowns"), "gap unknowns"),
    }


def _text_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAXIMUM_LIST_ITEMS:
        raise NarrativeValidationError(f"{field_name.title()} must be a bounded list")
    return [_text(item, field_name) for item in value]


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAXIMUM_TEXT_LENGTH:
        raise NarrativeValidationError(f"Narrative {field_name} is invalid")
    return value.strip()


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise NarrativeValidationError("Narrative confidence must be between zero and one")
    return round(float(value), 4)
