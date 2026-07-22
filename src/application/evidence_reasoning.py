"""Coordinates evidence-ledger reasoning and reconstruction-plan updates."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.evidence_reasoning import (
    DecisionTraceValidationError,
    apply_decision_trace,
    build_deterministic_decision_trace,
    build_evidence_ledger,
    decision_trace_json_schema,
    validate_decision_trace,
)
from domain.reconstruction_plan_v2 import validate_reconstruction_plan_v2
from infrastructure.azure_openai_reasoner import (
    AzureReasoningRequestError,
    AzureReasoningResponseError,
    AzureReasoningSettings,
    request_decision_trace,
)
from infrastructure.json_files import read_json_file, write_json_file


LOGGER = logging.getLogger(__name__)
REASONING_CACHE_SCHEMA_VERSION = 1
REASONING_PROMPT_VERSION = "bounded-forensic-hypotheses-v1"
PUBLIC_REASONING_FILENAME = "reasoning_public.json"


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
    ledger = build_evidence_ledger(scene_report, plans)
    write_json_file(work_directory / "evidence_ledger.json", ledger)
    settings = AzureReasoningSettings.from_environment(configuration) if configuration.get("enabled", True) else None
    mode, trace, metadata = _select_trace(
        ledger, settings, work_directory, reuse_work, cancellation_check,
    )
    updated_plans = apply_decision_trace(plans, ledger, trace)
    for plan_path, plan in zip(plan_paths, updated_plans):
        write_json_file(plan_path, plan)
    write_json_file(work_directory / "decision_trace.json", {**trace, "metadata": metadata})
    write_json_file(work_directory / PUBLIC_REASONING_FILENAME, _public_summary(ledger, trace, mode, metadata))
    return ReasoningResult(mode, updated_plans, trace)


def _select_trace(
    ledger: dict,
    settings: AzureReasoningSettings | None,
    work_directory: Path,
    reuse_work: bool,
    cancellation_check: CancellationCheck | None,
) -> tuple[str, dict, dict]:
    if settings is None:
        reason = "Azure OpenAI is not configured; deterministic visible-evidence planning was used."
        trace = validate_decision_trace(build_deterministic_decision_trace(ledger, reason), ledger)
        return "deterministic_fallback", trace, {"warning": reason}
    signature = _cache_signature(ledger, settings, configuration_version=REASONING_PROMPT_VERSION)
    cached = _read_cached_trace(work_directory, ledger, signature) if reuse_work else None
    if cached is not None:
        return "azure_cache", cached[0], cached[1]
    raise_if_cancelled(cancellation_check)
    try:
        raw_trace, metadata = request_decision_trace(settings, ledger, decision_trace_json_schema())
        trace = validate_decision_trace(raw_trace, ledger)
    except (AzureReasoningRequestError, AzureReasoningResponseError, DecisionTraceValidationError):
        LOGGER.exception("Azure reasoning failed for evidence digest %s", ledger["evidence_digest"])
        warning = "Azure reasoning was unavailable or invalid; deterministic visible-evidence planning was used."
        trace = validate_decision_trace(build_deterministic_decision_trace(ledger, warning), ledger)
        return "deterministic_fallback", trace, {"warning": warning, "deployment": settings.deployment}
    raise_if_cancelled(cancellation_check)
    cache_metadata = {**metadata, "signature": signature}
    write_json_file(work_directory / "reasoning_cache.json", {"trace": trace, "metadata": cache_metadata})
    return "azure", trace, cache_metadata


def _read_cached_trace(work_directory: Path, ledger: dict, signature: str) -> tuple[dict, dict] | None:
    cached = read_json_file(work_directory / "reasoning_cache.json")
    if not isinstance(cached, dict) or not isinstance(cached.get("metadata"), dict):
        return None
    if cached["metadata"].get("signature") != signature:
        return None
    try:
        trace = validate_decision_trace(cached.get("trace"), ledger)
    except DecisionTraceValidationError:
        return None
    return trace, cached["metadata"]


def _read_plan(plan_path: Path) -> dict:
    payload = read_json_file(plan_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Reconstruction plan is missing or invalid: {plan_path.name}")
    validate_reconstruction_plan_v2(payload)
    return payload


def _cache_signature(
    ledger: dict,
    settings: AzureReasoningSettings,
    configuration_version: str,
) -> str:
    cache_contract = {
        "schema_version": REASONING_CACHE_SCHEMA_VERSION,
        "evidence_digest": ledger["evidence_digest"],
        "deployment": settings.deployment,
        "reasoning_effort": settings.reasoning_effort,
        "configuration_version": configuration_version,
    }
    serialized = json.dumps(cache_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _public_summary(ledger: dict, trace: dict, mode: str, metadata: dict) -> dict:
    decisions = []
    for decision in trace["decisions"]:
        decisions.append({
            "gap_index": decision["gap_index"],
            "selected_hypothesis_id": decision["selected_hypothesis_id"],
            "evidence_references": decision["evidence_references"],
            "decision_summary": decision["decision_summary"],
            "rejected_hypotheses": decision["rejected_hypotheses"],
            "confidence": decision["confidence"],
            "unknowns": decision["unknowns"],
        })
    return {
        "status": "completed",
        "mode": mode,
        "deployment": metadata.get("deployment"),
        "warning": metadata.get("warning"),
        "scene_clues": ledger["scene_clues"],
        "decisions": decisions,
    }
