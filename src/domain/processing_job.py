import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


JOB_IDENTIFIER_PATTERN = re.compile(r"[a-f0-9]{32}")
MINIMUM_PROGRESS = 0.0
MAXIMUM_PROGRESS = 1.0
SUPPORTED_RENDERER_MODES = frozenset({"blender", "2d"})


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
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
            created_at=str(payload["created_at"]),
            started_at=_optional_string(payload.get("started_at")),
            completed_at=_optional_string(payload.get("completed_at")),
            output_path=(output_dir / output_file).resolve() if output_file else None,
            error=_optional_string(payload.get("error")),
            eta_seconds=_optional_integer(payload.get("eta_seconds")),
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
