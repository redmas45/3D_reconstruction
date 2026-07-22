"""Calculates public processing-job status without mutating job state."""

import stat
from datetime import datetime, timezone
from pathlib import Path

from domain.processing_job import JobStatus, ProcessingJob
from infrastructure.json_files import read_json_file


MINIMUM_ETA_PROGRESS = 0.02
PUBLIC_PROGRESS_DECIMAL_PLACES = 4
TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})
PUBLIC_REASONING_FILENAME = "reasoning_public.json"
MAXIMUM_PUBLIC_REASONING_BYTES = 512_000
MAXIMUM_PUBLIC_REASONING_ITEMS = 50


def estimate_eta(record: ProcessingJob) -> int | None:
    if record.started_at is None or record.progress < MINIMUM_ETA_PROGRESS:
        return None
    started_at = datetime.fromisoformat(record.started_at)
    elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
    remaining_seconds = elapsed_seconds * (1.0 - record.progress) / record.progress
    return max(0, int(round(remaining_seconds)))


def build_public_job_record(record: ProcessingJob) -> dict[str, object]:
    output_exists, output_size_bytes = _output_snapshot(record)
    live_eta_seconds, eta_status = live_eta(record)
    return {
        "id": record.job_id,
        "source_name": record.source_name,
        "status": record.status.value,
        "stage": record.stage.value,
        "progress": round(record.progress, PUBLIC_PROGRESS_DECIMAL_PLACES),
        "detail": record.detail,
        "created_at": record.created_at,
        "completed_at": record.completed_at,
        "elapsed_seconds": elapsed_seconds(record),
        "eta_seconds": live_eta_seconds,
        "eta_status": eta_status,
        "activity_log": [dict(item) for item in record.activity_log],
        "error": record.error,
        "output_url": f"/api/outputs/{record.job_id}" if output_exists else None,
        "download_url": f"/api/outputs/{record.job_id}?download=1" if output_exists else None,
        "size_bytes": output_size_bytes,
        "is_legacy_output": record.is_legacy_output,
        "renderer_mode": record.renderer_mode,
        "reasoning": _reasoning_snapshot(record.output_dir),
    }


def _reasoning_snapshot(output_directory: Path) -> dict | None:
    work_directory = output_directory / "_work"
    if not work_directory.is_dir():
        return None
    try:
        summary_paths = list(work_directory.glob(f"*/{PUBLIC_REASONING_FILENAME}"))
    except OSError:
        return None
    if len(summary_paths) != 1 or not _is_bounded_file(summary_paths[0]):
        return None
    payload = read_json_file(summary_paths[0])
    return payload if _valid_reasoning_summary(payload) else None


def _is_bounded_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size <= MAXIMUM_PUBLIC_REASONING_BYTES
    except OSError:
        return False


def _valid_reasoning_summary(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "completed" or not isinstance(payload.get("mode"), str):
        return False
    scene_clues = payload.get("scene_clues")
    decisions = payload.get("decisions")
    if not isinstance(scene_clues, list) or not isinstance(decisions, list):
        return False
    if len(scene_clues) > MAXIMUM_PUBLIC_REASONING_ITEMS or len(decisions) > MAXIMUM_PUBLIC_REASONING_ITEMS:
        return False
    return all(isinstance(item, str) for item in scene_clues) and all(
        _valid_public_decision(item) for item in decisions
    )


def _valid_public_decision(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    confidence = value.get("confidence")
    required_text = ("selected_hypothesis_id", "decision_summary")
    evidence_references = value.get("evidence_references")
    rejected_hypotheses = value.get("rejected_hypotheses")
    unknowns = value.get("unknowns")
    return all((
        isinstance(value.get("gap_index"), int) and not isinstance(value.get("gap_index"), bool),
        all(isinstance(value.get(field_name), str) for field_name in required_text),
        _valid_public_text_list(evidence_references),
        _valid_public_text_list(unknowns),
        isinstance(rejected_hypotheses, list),
        len(rejected_hypotheses) <= MAXIMUM_PUBLIC_REASONING_ITEMS if isinstance(rejected_hypotheses, list) else False,
        all(_valid_public_rejection(item) for item in rejected_hypotheses) if isinstance(rejected_hypotheses, list) else False,
        isinstance(confidence, (int, float)) and not isinstance(confidence, bool),
        0.0 <= float(confidence) <= 1.0,
    ))


def _valid_public_text_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= MAXIMUM_PUBLIC_REASONING_ITEMS
        and all(isinstance(item, str) for item in value)
    )


def _valid_public_rejection(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and isinstance(value.get("reason"), str)
    )


def _output_snapshot(record: ProcessingJob) -> tuple[bool, int | None]:
    if record.output_path is None:
        return False, None
    try:
        output_metadata = record.output_path.stat()
    except OSError:
        return False, None
    if not stat.S_ISREG(output_metadata.st_mode):
        return False, None
    return True, output_metadata.st_size


def live_eta(record: ProcessingJob) -> tuple[int | None, str]:
    if record.status is JobStatus.QUEUED:
        return None, "waiting"
    if record.status in TERMINAL_JOB_STATUSES:
        return 0, "finished"
    if record.eta_seconds is None or record.progress_updated_at is None:
        return None, "estimating"
    try:
        updated_at = datetime.fromisoformat(record.progress_updated_at)
    except ValueError:
        return None, "estimating"
    estimate_age = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    remaining_seconds = record.eta_seconds - estimate_age
    if remaining_seconds <= 0:
        return None, "recalibrating"
    return int(round(remaining_seconds)), "counting_down"


def elapsed_seconds(record: ProcessingJob) -> int:
    if record.started_at is None:
        return 0
    start = datetime.fromisoformat(record.started_at)
    end = datetime.fromisoformat(record.completed_at) if record.completed_at else datetime.now(timezone.utc)
    return max(0, int((end - start).total_seconds()))
