"""Coordinates visible evidence, Azure decisions, narrative, and storyboard artifacts."""

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from application.visible_evidence import (
    export_visible_evidence,
    validate_visual_evidence_manifest,
)
from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.clue_catalog import build_clue_catalog, public_clues
from domain.evidence_reasoning import build_evidence_ledger
from domain.gap_decisions import (
    GapDecisionValidationError,
    apply_gap_decisions,
    build_deterministic_gap_decisions,
    gap_decisions_json_schema,
    validate_gap_decisions,
)
from domain.gap_hypotheses import build_gap_hypotheses
from domain.reconstruction_narrative import (
    NarrativeValidationError,
    build_deterministic_narrative,
    narrative_json_schema,
    narrative_request_payload,
    validate_narrative,
)
from domain.reconstruction_plan_v2 import validate_reconstruction_plan_v2
from domain.render_storyboard import compile_render_storyboard
from infrastructure.azure_openai_reasoner import (
    AzureReasoningRequestError,
    AzureReasoningResponseError,
    AzureReasoningSettings,
    request_gap_decisions,
    request_reconstruction_narrative,
)
from infrastructure.json_files import read_json_file, write_json_file


LOGGER = logging.getLogger(__name__)
REASONING_CACHE_SCHEMA_VERSION = 2
REASONING_PROMPT_VERSION = "visible-story-planner-v2"
PUBLIC_REASONING_FILENAME = "reasoning_public.json"
DEFAULT_MAXIMUM_GAPS_PER_BATCH = 4
DEFAULT_MAXIMUM_IMAGES_PER_BATCH = 12
DIGEST_FIELDS = ("evidence_digest", "clue_digest", "hypothesis_digest")


@dataclass(frozen=True)
class ReasoningResult:
    mode: str
    plans: list[dict]
    trace: dict


def reason_about_reconstruction(
    scene_report: dict,
    plan_paths: list[Path],
    work_directory: Path,
    configuration: dict,
    reuse_work: bool,
    cancellation_check: CancellationCheck | None = None,
) -> ReasoningResult:
    plans = [_read_plan(path) for path in plan_paths]
    artifacts = _build_evidence_artifacts(
        scene_report, plans, work_directory, configuration, cancellation_check,
    )
    settings = _reasoning_settings(configuration)
    mode, decisions, decision_metadata = _select_decisions(
        artifacts, settings, work_directory, configuration, reuse_work, cancellation_check,
    )
    updated_plans = apply_gap_decisions(plans, artifacts["hypotheses"], decisions)
    updated_plans = _apply_fidelity_budget(
        updated_plans,
        int(configuration.get("renderer", {}).get("maximum_detailed_entities", 8)),
    )
    _write_plans(plan_paths, updated_plans)
    narrative, narrative_metadata = _select_narrative(
        artifacts["clues"], decisions, settings, mode, decision_metadata, cancellation_check,
    )
    storyboard = _write_storyboard(
        scene_report, updated_plans, artifacts["hypotheses"], decisions,
        work_directory, configuration,
    )
    _write_reasoning_artifacts(
        work_directory, artifacts, decisions, narrative, mode,
        decision_metadata, narrative_metadata, storyboard,
    )
    return ReasoningResult(mode, updated_plans, decisions)


def _build_evidence_artifacts(
    scene_report: dict,
    plans: list[dict],
    work_directory: Path,
    configuration: dict,
    cancellation_check: CancellationCheck | None,
) -> dict:
    evidence_directory = work_directory / "evidence"
    ledger = _evidence_ledger_v2(build_evidence_ledger(scene_report, plans))
    clues = build_clue_catalog(scene_report, plans)
    video_path = Path(str(scene_report.get("video", {}).get("path", "")))
    images = export_visible_evidence(
        video_path, scene_report, plans, evidence_directory,
        configuration.get("images", {}), cancellation_check,
    )
    validate_visual_evidence_manifest(images, scene_report)
    hypotheses = build_gap_hypotheses(plans, clues)
    write_json_file(evidence_directory / "evidence_ledger_v2.json", ledger)
    write_json_file(evidence_directory / "clue_catalog.json", clues)
    write_json_file(work_directory / "reasoning" / "gap_hypotheses_v2.json", hypotheses)
    return {"ledger": ledger, "clues": clues, "images": images, "hypotheses": hypotheses}


def _select_decisions(
    artifacts: dict,
    settings: AzureReasoningSettings | None,
    work_directory: Path,
    configuration: dict,
    reuse_work: bool,
    cancellation_check: CancellationCheck | None,
) -> tuple[str, dict, dict]:
    signature = _cache_signature(artifacts, settings)
    cached = _read_cached_decisions(work_directory, artifacts, signature) if reuse_work else None
    if cached is not None:
        return "azure_cache", cached[0], cached[1]
    if settings is None:
        return _fallback_decisions(artifacts, "Azure OpenAI is not configured.")
    raise_if_cancelled(cancellation_check)
    try:
        decisions, metadata = _request_decision_batches(
            settings, artifacts, configuration, cancellation_check,
        )
        validated = validate_gap_decisions(
            decisions, artifacts["ledger"]["evidence_digest"],
            artifacts["clues"], artifacts["hypotheses"],
        )
    except (AzureReasoningRequestError, AzureReasoningResponseError, GapDecisionValidationError):
        LOGGER.exception("Azure gap planning failed for %s", artifacts["ledger"]["evidence_digest"])
        return _fallback_decisions(artifacts, "Azure gap planning was unavailable or invalid.")
    cache_metadata = {**metadata, "signature": signature}
    cache = {"schema_version": REASONING_CACHE_SCHEMA_VERSION, "decisions": validated, "metadata": cache_metadata}
    write_json_file(work_directory / "reasoning" / "reasoning_cache.json", cache)
    return "azure", validated, cache_metadata


def _request_decision_batches(
    settings: AzureReasoningSettings,
    artifacts: dict,
    configuration: dict,
    cancellation_check: CancellationCheck | None,
) -> tuple[dict, dict]:
    maximum_gaps = max(1, int(configuration.get(
        "maximum_gaps_per_batch", DEFAULT_MAXIMUM_GAPS_PER_BATCH,
    )))
    gap_indexes = [int(item["gap_index"]) for item in artifacts["hypotheses"]["gaps"]]
    batches = [gap_indexes[index:index + maximum_gaps] for index in range(0, len(gap_indexes), maximum_gaps)]
    combined: list[dict] = []
    metadata_items = []
    for indexes in batches:
        raise_if_cancelled(cancellation_check)
        payload, image_paths = _batch_payload(artifacts, indexes, configuration)
        response, metadata = request_gap_decisions(
            settings, payload, _decision_batch_schema(payload), image_paths,
            str(configuration.get("image_detail", "low")),
        )
        _validate_batch_digests(response, artifacts)
        batch_decisions = response.get("decisions")
        if not isinstance(batch_decisions, list):
            raise GapDecisionValidationError("Azure batch decisions must be a list")
        combined.extend(batch_decisions)
        metadata_items.append(metadata)
    return _combined_decisions(artifacts, combined), _combined_metadata(metadata_items, settings)


def _batch_payload(
    artifacts: dict,
    gap_indexes: list[int],
    configuration: dict,
) -> tuple[dict, list[str]]:
    gaps = set(gap_indexes)
    clues = [
        clue for clue in artifacts["clues"]["clues"]
        if clue["scope"] == "scene" or _scope_gap_index(str(clue["scope"])) in gaps
    ]
    hypothesis_gaps = [
        item for item in artifacts["hypotheses"]["gaps"] if int(item["gap_index"]) in gaps
    ]
    ledger_gaps = [item for item in artifacts["ledger"]["gaps"] if int(item["gap_index"]) in gaps]
    images = _batch_images(artifacts["images"], gaps, configuration)
    return {
        "evidence_digest": artifacts["ledger"]["evidence_digest"],
        "clue_digest": artifacts["clues"]["clue_digest"],
        "hypothesis_digest": artifacts["hypotheses"]["hypothesis_digest"],
        "evidence_policy": "visible_frames_and_gap_boundaries_only",
        "vision_policy": "images may support semantics only; numeric geometry comes from supplied measurements",
        "gaps": ledger_gaps,
        "clues": clues,
        "hypothesis_gaps": hypothesis_gaps,
        "visual_evidence": [_public_image(item) for item in images],
    }, [str(item["path"]) for item in images]


def _batch_images(manifest: dict, gap_indexes: set[int], configuration: dict) -> list[dict]:
    limit = max(1, int(configuration.get(
        "maximum_images_per_batch", DEFAULT_MAXIMUM_IMAGES_PER_BATCH,
    )))
    global_images = [item for item in manifest["images"] if item["kind"] == "global_keyframe"]
    gap_images = [
        item for item in manifest["images"]
        if item.get("gap_index") in gap_indexes or item["kind"] == "entity_crop"
    ]
    return (global_images + gap_images)[:limit]


def _select_narrative(
    clues: dict,
    decisions: dict,
    settings: AzureReasoningSettings | None,
    mode: str,
    decision_metadata: dict,
    cancellation_check: CancellationCheck | None,
) -> tuple[dict, dict]:
    warning = decision_metadata.get("warning")
    fallback = build_deterministic_narrative(clues, decisions, mode, warning)
    if settings is None or mode == "deterministic_fallback":
        return validate_narrative(fallback, clues, decisions, mode), {"mode": "deterministic_fallback"}
    raise_if_cancelled(cancellation_check)
    try:
        response, metadata = request_reconstruction_narrative(
            settings,
            narrative_request_payload(clues, decisions),
            _narrative_schema(clues, mode),
        )
        return validate_narrative(response, clues, decisions, mode), metadata
    except (AzureReasoningRequestError, AzureReasoningResponseError, NarrativeValidationError):
        LOGGER.exception("Azure narrative synthesis failed for %s", clues["clue_digest"])
        return validate_narrative(fallback, clues, decisions, mode), {
            "mode": "deterministic_fallback",
            "warning": "Azure narrative synthesis was unavailable or invalid.",
        }


def _write_storyboard(
    scene_report: dict,
    plans: list[dict],
    hypotheses: dict,
    decisions: dict,
    work_directory: Path,
    configuration: dict,
) -> dict:
    renderer_configuration = configuration.get("renderer", {})
    storyboard, shell_manifest, render_budget = compile_render_storyboard(
        scene_report, plans, hypotheses, decisions, renderer_configuration,
    )
    storyboard_directory = work_directory / "storyboard"
    write_json_file(storyboard_directory / "render_storyboard.json", storyboard)
    write_json_file(storyboard_directory / "scene_shell_manifest.json", shell_manifest)
    write_json_file(storyboard_directory / "render_budget.json", render_budget)
    return {"storyboard": storyboard, "scene_shell": shell_manifest, "render_budget": render_budget}


def _write_reasoning_artifacts(
    work_directory: Path,
    artifacts: dict,
    decisions: dict,
    narrative: dict,
    mode: str,
    decision_metadata: dict,
    narrative_metadata: dict,
    storyboard: dict,
) -> None:
    reasoning_directory = work_directory / "reasoning"
    write_json_file(reasoning_directory / "gap_decisions_v2.json", decisions)
    write_json_file(reasoning_directory / "reconstruction_narrative.json", narrative)
    report = {
        "mode": mode,
        "decision_metadata": decision_metadata,
        "narrative_metadata": narrative_metadata,
        "storyboard_digest": storyboard["storyboard"]["storyboard_digest"],
    }
    write_json_file(reasoning_directory / "reasoning_report.json", report)
    write_json_file(work_directory / "evidence_ledger.json", artifacts["ledger"])
    write_json_file(work_directory / "decision_trace.json", {**decisions, "metadata": decision_metadata})
    write_json_file(
        work_directory / PUBLIC_REASONING_FILENAME,
        _public_summary(artifacts["clues"], decisions, narrative, mode, decision_metadata),
    )


def _public_summary(
    clues: dict,
    decisions: dict,
    narrative: dict,
    mode: str,
    metadata: dict,
) -> dict:
    bounded_clues = public_clues(clues)[:50]
    return {
        "status": "completed",
        "schema_version": 2,
        "mode": mode,
        "deployment": metadata.get("deployment"),
        "warning": metadata.get("warning"),
        "scene_clues": [item["statement"] for item in bounded_clues],
        "clues": bounded_clues,
        "headline": narrative["headline"],
        "whole_video_summary": narrative["whole_video_summary"],
        "story_points": narrative["story_points"],
        "gap_summaries": narrative["gap_summaries"],
        "causal_link_supported": narrative["causal_link_supported"],
        "confidence": narrative["confidence"],
        "unknowns": narrative["unknowns"],
        "decisions": decisions["decisions"],
    }


def _fallback_decisions(artifacts: dict, reason: str) -> tuple[str, dict, dict]:
    decisions = build_deterministic_gap_decisions(
        artifacts["ledger"]["evidence_digest"], artifacts["clues"], artifacts["hypotheses"], reason,
    )
    validated = validate_gap_decisions(
        decisions, artifacts["ledger"]["evidence_digest"], artifacts["clues"], artifacts["hypotheses"],
    )
    return "deterministic_fallback", validated, {"warning": reason}


def _read_cached_decisions(
    work_directory: Path,
    artifacts: dict,
    signature: str,
) -> tuple[dict, dict] | None:
    cached = read_json_file(work_directory / "reasoning" / "reasoning_cache.json")
    if not isinstance(cached, dict) or not isinstance(cached.get("metadata"), dict):
        return None
    if cached["metadata"].get("signature") != signature:
        return None
    try:
        decisions = validate_gap_decisions(
            cached.get("decisions"), artifacts["ledger"]["evidence_digest"],
            artifacts["clues"], artifacts["hypotheses"],
        )
    except GapDecisionValidationError:
        return None
    return decisions, cached["metadata"]


def _cache_signature(artifacts: dict, settings: AzureReasoningSettings | None) -> str:
    contract = {
        "schema_version": REASONING_CACHE_SCHEMA_VERSION,
        "prompt_version": REASONING_PROMPT_VERSION,
        "evidence_digest": artifacts["ledger"]["evidence_digest"],
        "clue_digest": artifacts["clues"]["clue_digest"],
        "hypothesis_digest": artifacts["hypotheses"]["hypothesis_digest"],
        "visual_digest": artifacts["images"]["manifest_digest"],
        "deployment": settings.deployment if settings else None,
        "reasoning_effort": settings.reasoning_effort if settings else None,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_plan(plan_path: Path) -> dict:
    payload = read_json_file(plan_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Reconstruction plan is missing or invalid: {plan_path.name}")
    validate_reconstruction_plan_v2(payload)
    return payload


def _write_plans(plan_paths: list[Path], plans: list[dict]) -> None:
    for plan_path, plan in zip(plan_paths, plans):
        write_json_file(plan_path, plan)


def _apply_fidelity_budget(plans: list[dict], maximum_detailed_entities: int) -> list[dict]:
    updated = copy.deepcopy(plans)
    for plan in updated:
        ranked = sorted(
            plan.get("entities", []),
            key=lambda item: float(item.get("relevance_score", 0.0)),
            reverse=True,
        )
        detailed_ids = set([
            str(item["id"]) for item in ranked
            if item.get("fidelity_tier") != "weak"
        ][:max(1, maximum_detailed_entities)])
        for entity in plan.get("entities", []):
            if str(entity["id"]) not in detailed_ids:
                entity["fidelity_tier"] = "weak"
    return updated


def _reasoning_settings(configuration: dict) -> AzureReasoningSettings | None:
    if not configuration.get("enabled", True):
        return None
    return AzureReasoningSettings.from_environment(configuration)


def _scope_gap_index(scope: str) -> int | None:
    parts = scope.split(":")
    if len(parts) < 2 or parts[0] != "gap":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _public_image(image: dict) -> dict:
    return {
        "id": image["id"],
        "kind": image["kind"],
        "frame": image["frame"],
        "gap_index": image.get("gap_index"),
        "track_id": image.get("track_id"),
    }


def _validate_batch_digests(response: dict, artifacts: dict) -> None:
    expected = {
        "evidence_digest": artifacts["ledger"]["evidence_digest"],
        "clue_digest": artifacts["clues"]["clue_digest"],
        "hypothesis_digest": artifacts["hypotheses"]["hypothesis_digest"],
    }
    if any(response.get(field) != value for field, value in expected.items()):
        raise GapDecisionValidationError("Azure batch response does not match supplied evidence")


def _decision_batch_schema(payload: dict) -> dict:
    schema = gap_decisions_json_schema()
    for field in DIGEST_FIELDS:
        schema["properties"][field] = {
            "type": "string",
            "enum": [str(payload[field])],
        }
    return schema


def _narrative_schema(clues: dict, mode: str) -> dict:
    schema = narrative_json_schema()
    schema["properties"]["clue_digest"] = {
        "type": "string",
        "enum": [str(clues["clue_digest"])],
    }
    schema["properties"]["mode"] = {
        "type": "string",
        "enum": [mode],
    }
    return schema


def _combined_decisions(artifacts: dict, decisions: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "evidence_digest": artifacts["ledger"]["evidence_digest"],
        "clue_digest": artifacts["clues"]["clue_digest"],
        "hypothesis_digest": artifacts["hypotheses"]["hypothesis_digest"],
        "decisions": decisions,
    }


def _combined_metadata(items: list[dict], settings: AzureReasoningSettings) -> dict:
    return {
        "provider": "azure_openai",
        "deployment": settings.deployment,
        "batch_count": len(items),
        "response_ids": [item.get("response_id") for item in items],
        "usage": [item.get("usage", {}) for item in items],
    }


def _evidence_ledger_v2(legacy_ledger: dict) -> dict:
    gaps = [
        {key: value for key, value in gap.items() if key != "hypotheses"}
        for gap in legacy_ledger["gaps"]
    ]
    contract = {
        "schema_version": 2,
        "evidence_policy": legacy_ledger["evidence_policy"],
        "scene_clues": legacy_ledger["scene_clues"],
        "gaps": gaps,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**contract, "evidence_digest": hashlib.sha256(serialized).hexdigest()}
