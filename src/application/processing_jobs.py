"""Owns the persisted lifecycle of locally processed reconstruction jobs."""

import hashlib
import json
import logging
import random
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from application.reconstruction_pipeline import PipelineOptions, ProgressCallback, process_video, video_info
from domain.cancellation import CancellationRequestedError, raise_if_cancelled
from domain.processing_job import (
    JobActivity,
    JobStatus,
    MAXIMUM_STORED_ACTIVITY_ITEMS,
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
METADATA_REPLACE_ATTEMPTS = 8
METADATA_RETRY_BASE_SECONDS = 0.05


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
        self._cancellation_events: dict[str, threading.Event] = {}
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
            self._cancellation_events[job_id] = threading.Event()
            self._record_activity(record)
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
            if record.status in {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.CANCELLING}:
                raise JobConflictError("Cancel the active job before deleting it")
            input_dir = record.input_path.parent
            output_dir = record.output_dir
        if record.is_legacy_output:
            self._remove_legacy_output(record)
        else:
            self._remove_job_directories(input_dir, output_dir)
        with self._lock:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
            self._cancellation_events.pop(job_id, None)

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            record = self._require_job(job_id)
            if record.status not in {JobStatus.QUEUED, JobStatus.PROCESSING}:
                raise JobConflictError("Only a queued or processing job can be cancelled")
            cancellation_event = self._cancellation_events[job_id]
            cancellation_event.set()
            future = self._futures.get(job_id)
            if record.status is JobStatus.QUEUED and future is not None and future.cancel():
                self._mark_cancelled(record)
                return self._public_record(record)
            record.status = JobStatus.CANCELLING
            record.stage = ProcessingStage.CANCELLING
            record.detail = "Stopping active reconstruction processes"
            record.eta_seconds = None
            self._record_activity(record)
            self._persist(record)
            return self._public_record(record)

    def shutdown(self) -> None:
        with self._lock:
            for cancellation_event in self._cancellation_events.values():
                cancellation_event.set()
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
        started_job = self._begin_job(job_id)
        if started_job is None:
            return
        record, cancellation_event = started_job
        try:
            output_path = self._run_processor(job_id, record, cancellation_event)
        except CancellationRequestedError:
            LOGGER.info("Reconstruction job %s was cancelled", job_id)
            self._cancel_running_job(job_id)
            return
        except Exception as error:
            LOGGER.exception("Reconstruction job %s failed for %s", job_id, record.source_name)
            self._fail_job(job_id, str(error) or error.__class__.__name__)
            raise
        self._complete_job(job_id, output_path)

    def _begin_job(self, job_id: str) -> tuple[ProcessingJob, threading.Event] | None:
        with self._lock:
            record = self._require_job(job_id)
            cancellation_event = self._cancellation_events[job_id]
            if cancellation_event.is_set():
                self._mark_cancelled(record)
                return None
            record.status = JobStatus.PROCESSING
            record.stage = ProcessingStage.VALIDATING
            record.detail = "Starting reconstruction"
            record.started_at = utc_now()
            record.progress_updated_at = record.started_at
            self._record_activity(record)
            self._persist(record)
            return record, cancellation_event

    def _run_processor(
        self,
        job_id: str,
        record: ProcessingJob,
        cancellation_event: threading.Event,
    ) -> Path:
        def update(stage: str, progress: float, detail: str) -> None:
            raise_if_cancelled(cancellation_event.is_set)
            self._update_progress(job_id, stage, progress, detail)

        options = PipelineOptions(
            self.config_data,
            record.output_dir,
            reuse_work=False,
            renderer_mode=record.renderer_mode,
            cancellation_check=cancellation_event.is_set,
        )
        rng = random.Random(int(job_id[:16], 16))
        output_path = self.processor(record.input_path, options, rng, update)
        raise_if_cancelled(cancellation_event.is_set)
        return output_path

    def _complete_job(self, job_id: str, output_path: Path) -> None:
        with self._lock:
            record = self._require_job(job_id)
            if record.status is JobStatus.CANCELLING:
                self._mark_cancelled(record)
                return
            record.status = JobStatus.COMPLETED
            record.stage = ProcessingStage.COMPLETED
            record.progress = 1.0
            record.detail = "Reconstruction complete"
            record.completed_at = utc_now()
            record.output_path = output_path.resolve()
            record.eta_seconds = 0
            record.progress_updated_at = record.completed_at
            self._record_activity(record)
            self._persist(record)

    def _cancel_running_job(self, job_id: str) -> None:
        with self._lock:
            self._mark_cancelled(self._require_job(job_id))

    def _mark_cancelled(self, record: ProcessingJob) -> None:
        record.status = JobStatus.CANCELLED
        record.stage = ProcessingStage.CANCELLED
        record.detail = "Reconstruction cancelled"
        record.completed_at = utc_now()
        record.error = None
        record.eta_seconds = None
        record.progress_updated_at = record.completed_at
        self._record_activity(record)
        self._persist(record)

    def _update_progress(self, job_id: str, stage: str, progress: float, detail: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            if record.status is JobStatus.CANCELLING:
                raise CancellationRequestedError("Reconstruction was cancelled by the operator")
            next_progress = max(record.progress, min(0.99, progress))
            record.stage = ProcessingStage(stage)
            if next_progress > record.progress:
                record.progress_updated_at = utc_now()
            record.progress = next_progress
            record.detail = detail
            record.eta_seconds = self._estimate_eta(record)
            self._record_activity(record)
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
            record.progress_updated_at = record.completed_at
            self._record_activity(record)
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
        live_eta_seconds, eta_status = self._live_eta(record)
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
            "eta_seconds": live_eta_seconds,
            "eta_status": eta_status,
            "activity_log": [dict(item) for item in record.activity_log],
            "error": record.error,
            "output_url": f"/api/outputs/{record.job_id}" if output_exists else None,
            "download_url": f"/api/outputs/{record.job_id}?download=1" if output_exists else None,
            "size_bytes": record.output_path.stat().st_size if output_exists else None,
            "is_legacy_output": record.is_legacy_output,
            "renderer_mode": record.renderer_mode,
        }

    @staticmethod
    def _live_eta(record: ProcessingJob) -> tuple[int | None, str]:
        if record.status is JobStatus.QUEUED:
            return None, "waiting"
        if record.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
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

    def _record_activity(self, record: ProcessingJob) -> None:
        activity: JobActivity = {
            "timestamp": utc_now(),
            "stage": record.stage.value,
            "detail": record.detail,
            "progress": round(record.progress, 4),
        }
        if record.activity_log and _same_activity(record.activity_log[-1], activity):
            return
        record.activity_log.append(activity)
        del record.activity_log[:-MAXIMUM_STORED_ACTIVITY_ITEMS]

    def _elapsed_seconds(self, record: ProcessingJob) -> int:
        if record.started_at is None:
            return 0
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.completed_at) if record.completed_at else datetime.now(timezone.utc)
        return max(0, int((end - start).total_seconds()))

    def _persist(self, record: ProcessingJob) -> None:
        metadata_path = record.output_dir / "job.json"
        temporary_path = record.output_dir / f"job.{uuid.uuid4().hex}.tmp"
        try:
            with temporary_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(record.to_storage_payload(), metadata_file, indent=2)
            _replace_metadata_file(temporary_path, metadata_path)
        finally:
            temporary_path.unlink(missing_ok=True)

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
                record.progress_updated_at = record.completed_at
                self._record_activity(record)
                self._persist(record)
            elif record.status is JobStatus.CANCELLING:
                self._mark_cancelled(record)
            self._jobs[record.job_id] = record
            self._cancellation_events[record.job_id] = threading.Event()

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


def _replace_metadata_file(temporary_path: Path, metadata_path: Path) -> None:
    for attempt_index in range(METADATA_REPLACE_ATTEMPTS):
        try:
            temporary_path.replace(metadata_path)
            return
        except PermissionError:
            if attempt_index == METADATA_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(METADATA_RETRY_BASE_SECONDS * (attempt_index + 1))


def _same_activity(first: JobActivity, second: JobActivity) -> bool:
    return all(first[field_name] == second[field_name] for field_name in ("stage", "detail", "progress"))
