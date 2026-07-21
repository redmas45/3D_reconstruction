import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypedDict


JOB_IDENTIFIER_PATTERN = re.compile(r"[a-f0-9]{32}")
MINIMUM_PROGRESS = 0.0
MAXIMUM_PROGRESS = 1.0
SUPPORTED_RENDERER_MODES = frozenset({"blender", "2d"})
MAXIMUM_STORED_ACTIVITY_ITEMS = 80


class JobActivity(TypedDict):
    timestamp: str
    stage: str
    detail: str
    progress: float


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    SELECTING_GAPS = "selecting_gaps"
    PREPARING = "preparing"
    DETECTING = "detecting"
    PLANNING = "planning"
    RENDERING = "rendering"
    EVALUATING = "evaluating"
    STITCHING = "stitching"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_job_identifier(job_id: str) -> str:
    if JOB_IDENTIFIER_PATTERN.fullmatch(job_id) is None:
        raise ValueError("Job identifier is invalid")
    return job_id


def validate_renderer_mode(renderer_mode: str) -> str:
    if renderer_mode not in SUPPORTED_RENDERER_MODES:
        raise ValueError("Renderer mode must be 'blender' or '2d'")
    return renderer_mode


@dataclass
class ProcessingJob:
    job_id: str
    source_name: str
    input_path: Path
    output_dir: Path
    status: JobStatus = JobStatus.QUEUED
    stage: ProcessingStage = ProcessingStage.QUEUED
    progress: float = MINIMUM_PROGRESS
    detail: str = "Waiting to start"
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    output_path: Path | None = None
    error: str | None = None
    eta_seconds: int | None = None
    progress_updated_at: str | None = None
    activity_log: list[JobActivity] = field(default_factory=list)
    is_legacy_output: bool = False
    renderer_mode: str = "blender"

    def to_storage_payload(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "source_name": self.source_name,
            "input_file": self.input_path.name,
            "status": self.status.value,
            "stage": self.stage.value,
            "progress": self.progress,
            "detail": self.detail,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_file": self.output_path.name if self.output_path else None,
            "error": self.error,
            "eta_seconds": self.eta_seconds,
            "progress_updated_at": self.progress_updated_at,
            "activity_log": [dict(item) for item in self.activity_log],
            "is_legacy_output": self.is_legacy_output,
            "renderer_mode": self.renderer_mode,
        }

    @classmethod
    def from_storage_payload(
        cls,
        upload_root: Path,
        output_dir: Path,
        payload: dict[str, object],
    ) -> "ProcessingJob":
        job_id = validate_job_identifier(str(payload["job_id"]))
        input_file = _validate_stored_filename(payload["input_file"])
        output_value = payload.get("output_file")
        output_file = _validate_stored_filename(output_value) if output_value else None
        progress = float(payload["progress"])
        if not MINIMUM_PROGRESS <= progress <= MAXIMUM_PROGRESS:
            raise ValueError("Persisted job progress is outside the supported range")
        return cls(
            job_id=job_id,
            source_name=_validate_stored_filename(payload["source_name"]),
            input_path=upload_root / job_id / input_file,
            output_dir=output_dir.resolve(),
            status=JobStatus(str(payload["status"])),
            stage=ProcessingStage(str(payload["stage"])),
            progress=progress,
            detail=str(payload["detail"]),
            created_at=_aware_timestamp(payload["created_at"], "created_at"),
            started_at=_optional_aware_timestamp(payload.get("started_at"), "started_at"),
            completed_at=_optional_aware_timestamp(payload.get("completed_at"), "completed_at"),
            output_path=(output_dir / output_file).resolve() if output_file else None,
            error=_optional_string(payload.get("error")),
            eta_seconds=_optional_integer(payload.get("eta_seconds")),
            progress_updated_at=_optional_aware_timestamp(
                payload.get("progress_updated_at"), "progress_updated_at"
            ),
            activity_log=_validated_activity_log(payload.get("activity_log", [])),
            is_legacy_output=bool(payload.get("is_legacy_output", False)),
            renderer_mode=validate_renderer_mode(str(payload.get("renderer_mode", "2d"))),
        )


def _validate_stored_filename(value: object) -> str:
    filename = str(value)
    if not filename or Path(filename).name != filename:
        raise ValueError("Persisted filename is invalid")
    return filename


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_integer(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_aware_timestamp(value: object, field_name: str) -> str | None:
    return None if value is None else _aware_timestamp(value, field_name)


def _aware_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Persisted {field_name} timestamp is invalid")
    try:
        parsed_timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Persisted {field_name} timestamp is invalid") from error
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise ValueError(f"Persisted {field_name} timestamp must include a timezone")
    return parsed_timestamp.astimezone(timezone.utc).isoformat()


def _validated_activity_log(value: object) -> list[JobActivity]:
    if not isinstance(value, list):
        return []
    validated_items: list[JobActivity] = []
    for raw_item in value[-MAXIMUM_STORED_ACTIVITY_ITEMS:]:
        validated_item = _validated_activity_item(raw_item)
        if validated_item is not None:
            validated_items.append(validated_item)
    return validated_items


def _validated_activity_item(value: object) -> JobActivity | None:
    if not isinstance(value, dict):
        return None
    try:
        stage = ProcessingStage(str(value["stage"])).value
        progress = float(value["progress"])
        timestamp = _aware_timestamp(value["timestamp"], "activity")
        detail = str(value["detail"])
    except (KeyError, TypeError, ValueError):
        return None
    if not MINIMUM_PROGRESS <= progress <= MAXIMUM_PROGRESS or not timestamp or not detail:
        return None
    return {"timestamp": timestamp, "stage": stage, "detail": detail, "progress": progress}
