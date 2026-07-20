"""Owns the persisted lifecycle of locally processed reconstruction jobs."""

import hashlib
import json
import logging
import random
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from application.reconstruction_pipeline import PipelineOptions, ProgressCallback, process_video, video_info
from domain.processing_job import (
    JobStatus,
    ProcessingJob,
    ProcessingStage,
    utc_now,
    validate_job_identifier,
    validate_renderer_mode,
)
from domain.video_upload import (
    DEFAULT_MAX_UPLOAD_BYTES,
    UPLOAD_CHUNK_BYTES,
    UploadValidationError,
    validate_upload_metadata,
)


LOGGER = logging.getLogger(__name__)
Processor = Callable[[Path, PipelineOptions, random.Random, ProgressCallback | None], Path]


class JobNotFoundError(LookupError):
    pass


class JobConflictError(RuntimeError):
    pass


class JobManager:
    def __init__(
        self,
        upload_root: Path,
        output_root: Path,
        config_data: dict,
        processor: Processor = process_video,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        legacy_output_root: Path | None = None,
    ) -> None:
        self.upload_root = upload_root.resolve()
        self.output_root = output_root.resolve()
        self.config_data = config_data
        self.processor = processor
        self.max_upload_bytes = max_upload_bytes
        self.legacy_output_root = legacy_output_root.resolve() if legacy_output_root else None
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ProcessingJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reconstruction")
        self._load_existing_jobs()
        self._load_legacy_outputs()

    def create_job(
        self,
        source_name: str,
        reader: BinaryIO,
        content_length: int,
        renderer_mode: str = "blender",
    ) -> dict:
        safe_name = validate_upload_metadata(source_name, content_length, self.max_upload_bytes)
        renderer_mode = validate_renderer_mode(renderer_mode)
        job_id = uuid.uuid4().hex
        input_dir = self.upload_root / job_id
        output_dir = self.output_root / job_id
        input_path = input_dir / safe_name
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._write_upload(reader, input_path, content_length)
            video_info(input_path)
        except (OSError, ValueError) as error:
            self._remove_job_directories(input_dir, output_dir)
            raise UploadValidationError(str(error)) from error
        record = ProcessingJob(
            job_id=job_id,
            source_name=safe_name,
            input_path=input_path,
            output_dir=output_dir,
            created_at=utc_now(),
            renderer_mode=renderer_mode,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._persist(record)
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
            return self._public_record(record)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._public_record(record) for record in records]

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            return self._public_record(self._require_job(job_id))

    def output_path(self, job_id: str) -> Path:
        with self._lock:
            record = self._require_job(job_id)
            if record.status is not JobStatus.COMPLETED or record.output_path is None or not record.output_path.is_file():
                raise JobConflictError("Output is not available yet")
            return record.output_path

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            if record.status in {JobStatus.QUEUED, JobStatus.PROCESSING}:
                raise JobConflictError("A queued or processing job cannot be deleted")
            input_dir = record.input_path.parent
            output_dir = record.output_dir
        if record.is_legacy_output:
            self._remove_legacy_output(record)
        else:
            self._remove_job_directories(input_dir, output_dir)
        with self._lock:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _write_upload(self, reader: BinaryIO, input_path: Path, content_length: int) -> None:
        remaining = content_length
        with input_path.open("wb") as output_file:
            while remaining:
                chunk = reader.read(min(remaining, UPLOAD_CHUNK_BYTES))
                if not chunk:
                    raise UploadValidationError("Upload ended before the declared file size")
                output_file.write(chunk)
                remaining -= len(chunk)

    def _execute(self, job_id: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            record.status = JobStatus.PROCESSING
            record.stage = ProcessingStage.VALIDATING
            record.detail = "Starting reconstruction"
            record.started_at = utc_now()
            self._persist(record)

        def update(stage: str, progress: float, detail: str) -> None:
            self._update_progress(job_id, stage, progress, detail)

        try:
            options = PipelineOptions(
                self.config_data,
                record.output_dir,
                reuse_work=False,
                renderer_mode=record.renderer_mode,
            )
            rng = random.Random(int(job_id[:16], 16))
            output_path = self.processor(record.input_path, options, rng, update)
        except Exception as error:
            LOGGER.exception("Reconstruction job %s failed for %s", job_id, record.source_name)
            self._fail_job(job_id, str(error) or error.__class__.__name__)
            raise
        with self._lock:
            record = self._require_job(job_id)
            record.status = JobStatus.COMPLETED
            record.stage = ProcessingStage.COMPLETED
            record.progress = 1.0
            record.detail = "Reconstruction complete"
            record.completed_at = utc_now()
            record.output_path = output_path.resolve()
            record.eta_seconds = 0
            self._persist(record)

    def _update_progress(self, job_id: str, stage: str, progress: float, detail: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            record.stage = ProcessingStage(stage)
            record.progress = max(record.progress, min(0.99, progress))
            record.detail = detail
            record.eta_seconds = self._estimate_eta(record)
            self._persist(record)

    def _fail_job(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            record.status = JobStatus.FAILED
            record.stage = ProcessingStage.FAILED
            record.detail = "Processing failed"
            record.error = message
            record.completed_at = utc_now()
            record.eta_seconds = None
            self._persist(record)

    def _estimate_eta(self, record: ProcessingJob) -> int | None:
        if record.started_at is None or record.progress < 0.02:
            return None
        started_at = datetime.fromisoformat(record.started_at)
        elapsed_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
        remaining = elapsed_seconds * (1.0 - record.progress) / record.progress
        return max(0, int(round(remaining)))

    def _public_record(self, record: ProcessingJob) -> dict:
        output_exists = record.output_path is not None and record.output_path.is_file()
        elapsed_seconds = self._elapsed_seconds(record)
        return {
            "id": record.job_id,
            "source_name": record.source_name,
            "status": record.status.value,
            "stage": record.stage.value,
            "progress": round(record.progress, 4),
            "detail": record.detail,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": record.eta_seconds,
            "error": record.error,
            "output_url": f"/api/outputs/{record.job_id}" if output_exists else None,
            "download_url": f"/api/outputs/{record.job_id}?download=1" if output_exists else None,
            "size_bytes": record.output_path.stat().st_size if output_exists else None,
            "is_legacy_output": record.is_legacy_output,
            "renderer_mode": record.renderer_mode,
        }

    def _elapsed_seconds(self, record: ProcessingJob) -> int:
        if record.started_at is None:
            return 0
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.completed_at) if record.completed_at else datetime.now(timezone.utc)
        return max(0, int((end - start).total_seconds()))

    def _persist(self, record: ProcessingJob) -> None:
        metadata_path = record.output_dir / "job.json"
        temporary_path = record.output_dir / "job.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(record.to_storage_payload(), metadata_file, indent=2)
        temporary_path.replace(metadata_path)

    def _load_existing_jobs(self) -> None:
        for metadata_path in self.output_root.glob("*/job.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                record = self._record_from_storage(metadata_path.parent, payload)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                LOGGER.exception("Ignoring invalid job metadata at %s", metadata_path)
                continue
            if record.status in {JobStatus.QUEUED, JobStatus.PROCESSING}:
                record.status = JobStatus.FAILED
                record.stage = ProcessingStage.FAILED
                record.error = "Processing was interrupted when the local server stopped"
                record.completed_at = utc_now()
                self._persist(record)
            self._jobs[record.job_id] = record

    def _load_legacy_outputs(self) -> None:
        if self.legacy_output_root is None or not self.legacy_output_root.exists():
            return
        for output_path in self.legacy_output_root.rglob("*_reconstructed.mp4"):
            resolved_output = output_path.resolve()
            if self.output_root in resolved_output.parents or "_work" in resolved_output.parts:
                continue
            relative_path = resolved_output.relative_to(self.legacy_output_root)
            job_id = hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:32]
            completed_at = datetime.fromtimestamp(resolved_output.stat().st_mtime, timezone.utc).isoformat()
            self._jobs[job_id] = ProcessingJob(
                job_id=job_id,
                source_name=resolved_output.name.replace("_reconstructed", ""),
                input_path=resolved_output,
                output_dir=resolved_output.parent,
                status=JobStatus.COMPLETED,
                stage=ProcessingStage.COMPLETED,
                progress=1.0,
                detail="Existing reconstruction",
                created_at=completed_at,
                completed_at=completed_at,
                output_path=resolved_output,
                eta_seconds=0,
                is_legacy_output=True,
                renderer_mode="2d",
            )

    def _record_from_storage(self, output_dir: Path, payload: dict[str, object]) -> ProcessingJob:
        return ProcessingJob.from_storage_payload(self.upload_root, output_dir, payload)

    def _require_job(self, job_id: str) -> ProcessingJob:
        try:
            validate_job_identifier(job_id)
        except ValueError as error:
            raise JobNotFoundError("Job was not found") from error
        record = self._jobs.get(job_id)
        if record is None:
            raise JobNotFoundError("Job was not found")
        return record

    def _remove_job_directories(self, input_dir: Path, output_dir: Path) -> None:
        self._remove_managed_tree(input_dir, self.upload_root)
        self._remove_managed_tree(output_dir, self.output_root)

    def _remove_legacy_output(self, record: ProcessingJob) -> None:
        if self.legacy_output_root is None or record.output_path is None:
            raise ValueError("Legacy output root is unavailable")
        output_path = record.output_path.resolve()
        if self.legacy_output_root not in output_path.parents or not output_path.is_file():
            raise ValueError(f"Refusing to delete unmanaged legacy output: {output_path}")
        output_path.unlink()

    @staticmethod
    def _remove_managed_tree(target: Path, root: Path) -> None:
        resolved_target = target.resolve()
        resolved_root = root.resolve()
        if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
            raise ValueError(f"Refusing to delete unmanaged path: {resolved_target}")
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
