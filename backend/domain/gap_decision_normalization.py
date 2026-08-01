"""Normalizes unambiguous model references before strict gap-decision validation."""

import copy

from domain.gap_decisions import MAXIMUM_LIST_ITEMS, MAXIMUM_TEXT_LENGTH


def normalize_gap_decision_references(
    value: object,
    hypothesis_library: dict,
) -> tuple[object, dict[str, int]]:
    normalized = copy.deepcopy(value)
    report = _empty_report()
    if not isinstance(normalized, dict):
        return normalized, report
    decisions = normalized.get("decisions")
    if not isinstance(decisions, list):
        return normalized, report
    catalogs = _hypothesis_catalogs(hypothesis_library)
    for decision in decisions:
        _normalize_gap_decision(decision, catalogs, report)
    return normalized, report


def _normalize_gap_decision(
    decision: object,
    catalogs: dict[int, dict[str, dict[str, str]]],
    report: dict[str, int],
) -> None:
    if not isinstance(decision, dict) or isinstance(decision.get("gap_index"), bool):
        return
    gap_index = decision.get("gap_index")
    if not isinstance(gap_index, int) or gap_index not in catalogs:
        return
    entities = decision.get("entities")
    if not isinstance(entities, list):
        return
    for entity in entities:
        _normalize_entity_decision(entity, catalogs[gap_index], report)


def _normalize_entity_decision(
    entity: object,
    catalogs: dict[str, dict[str, str]],
    report: dict[str, int],
) -> None:
    if not isinstance(entity, dict):
        return
    entity_id = entity.get("entity_id")
    if not isinstance(entity_id, str) or entity_id not in catalogs:
        return
    aliases = catalogs[entity_id]
    selected = _resolve_identifier(entity.get("selected_hypothesis_id"), aliases)
    if selected is not None and selected != entity.get("selected_hypothesis_id"):
        entity["selected_hypothesis_id"] = selected
        report["repaired_selected_hypotheses"] += 1
    entity["rejected_hypotheses"] = _normalize_rejections(
        entity.get("rejected_hypotheses"), aliases, selected, report,
    )


def _normalize_rejections(
    value: object,
    aliases: dict[str, str],
    selected: str | None,
    report: dict[str, int],
) -> list[dict]:
    if not isinstance(value, list):
        report["dropped_rejected_hypotheses"] += 1
        return []
    normalized = []
    seen: set[str] = set()
    for item_index, item in enumerate(value):
        canonical = _canonical_rejection(item, aliases, selected)
        if canonical is None or canonical["id"] in seen:
            report["dropped_rejected_hypotheses"] += 1
            continue
        if isinstance(item, dict) and canonical["id"] != item.get("id"):
            report["repaired_rejected_hypotheses"] += 1
        seen.add(canonical["id"])
        normalized.append(canonical)
        if len(normalized) == MAXIMUM_LIST_ITEMS:
            report["dropped_rejected_hypotheses"] += len(value) - item_index - 1
            break
    return normalized


def _canonical_rejection(
    value: object,
    aliases: dict[str, str],
    selected: str | None,
) -> dict | None:
    if not isinstance(value, dict) or not _valid_reason(value.get("reason")):
        return None
    identifier = _resolve_identifier(value.get("id"), aliases)
    if identifier is None or identifier == selected:
        return None
    return {"id": identifier, "reason": str(value["reason"]).strip()}


def _resolve_identifier(value: object, aliases: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    return aliases.get(value.strip())


def _hypothesis_catalogs(
    hypothesis_library: dict,
) -> dict[int, dict[str, dict[str, str]]]:
    catalogs: dict[int, dict[str, dict[str, str]]] = {}
    for gap in hypothesis_library.get("gaps", []):
        if not isinstance(gap, dict) or not isinstance(gap.get("gap_index"), int):
            continue
        catalogs[int(gap["gap_index"])] = {
            str(entity["entity_id"]): _entity_aliases(entity)
            for entity in gap.get("entities", [])
            if isinstance(entity, dict) and isinstance(entity.get("entity_id"), str)
        }
    return catalogs


def _entity_aliases(entity: dict) -> dict[str, str]:
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for hypothesis in entity.get("hypotheses", []):
        if not isinstance(hypothesis, dict) or not isinstance(hypothesis.get("id"), str):
            continue
        canonical = str(hypothesis["id"])
        for alias in (canonical, hypothesis.get("type")):
            if not isinstance(alias, str):
                continue
            if alias in aliases and aliases[alias] != canonical:
                ambiguous.add(alias)
            else:
                aliases[alias] = canonical
    return {alias: canonical for alias, canonical in aliases.items() if alias not in ambiguous}


def _valid_reason(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= MAXIMUM_TEXT_LENGTH


def _empty_report() -> dict[str, int]:
    return {
        "repaired_selected_hypotheses": 0,
        "repaired_rejected_hypotheses": 0,
        "dropped_rejected_hypotheses": 0,
    }
