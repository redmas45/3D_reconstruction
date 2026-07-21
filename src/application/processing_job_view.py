"""Calculates public processing-job status without mutating job state."""

import stat
from datetime import datetime, timezone

from domain.processing_job import JobStatus, ProcessingJob


MINIMUM_ETA_PROGRESS = 0.02
PUBLIC_PROGRESS_DECIMAL_PLACES = 4
TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


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
    }


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
