"""Owns the persisted lifecycle of locally processed reconstruction jobs."""

import hashlib
import json
import logging
import random
import shutil
import stat
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from application.processing_job_view import build_public_job_record, estimate_eta
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
from gap_selector import choose_hidden_gaps
from infrastructure.job_metadata_store import JobMetadataStore, replace_metadata_file as _replace_metadata_file
from infrastructure.blender_runner import BlenderRenderError, BlenderUnavailableError
from infrastructure.media_tools import (
    MediaProcessingError,
    MediaToolUnavailableError,
    UnsupportedVideoTimingError,
)


LOGGER = logging.getLogger(__name__)
Processor = Callable[[Path, PipelineOptions, random.Random, ProgressCallback | None], Path]
PUBLIC_FAILURE_MESSAGE_LIMIT = 240
GENERIC_FAILURE_MESSAGE = "Reconstruction failed. Check the server log for details."
INCOMPLETE_UPLOAD_MARKER = ".upload_incomplete"
MAXIMUM_UPLOAD_TRANSFER_SECONDS = 1_800


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
        self._metadata_store = JobMetadataStore()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reconstruction")
        self._load_existing_jobs()
        self._remove_incomplete_uploads()
        self._load_legacy_outputs()

    def create_job(
        self,
        source_name: str,
        reader: BinaryIO,
        content_length: int,
        renderer_mode: str = "blender",
    ) -> dict[str, object]:
        safe_name = validate_upload_metadata(source_name, content_length, self.max_upload_bytes)
        renderer_mode = validate_renderer_mode(renderer_mode)
        job_id = uuid.uuid4().hex
        input_dir = self.upload_root / job_id
        output_dir = self.output_root / job_id
        input_path = input_dir / safe_name
        try:
            input_dir.mkdir(parents=True, exist_ok=False)
            (input_dir / INCOMPLETE_UPLOAD_MARKER).touch()
            output_dir.mkdir(parents=True, exist_ok=False)
            self._write_upload(reader, input_path, content_length)
            input_video_info = video_info(input_path)
            _validate_gap_policy(input_video_info, self.config_data)
        except UploadValidationError:
            self._remove_job_directories(input_dir, output_dir)
            raise
        except ValueError as error:
            self._remove_job_directories(input_dir, output_dir)
            raise UploadValidationError(str(error)) from error
        except OSError as error:
            self._remove_job_directories(input_dir, output_dir)
            LOGGER.exception("Could not save or validate uploaded video %s", safe_name)
            raise UploadValidationError("The uploaded video could not be saved or decoded") from error
        record = ProcessingJob(
            job_id=job_id,
            source_name=safe_name,
            input_path=input_path,
            output_dir=output_dir,
            created_at=utc_now(),
            renderer_mode=renderer_mode,
        )
        try:
            with self._lock:
                self._jobs[job_id] = record
                self._cancellation_events[job_id] = threading.Event()
                self._record_activity(record)
                self._persist(record, force=True)
                (input_dir / INCOMPLETE_UPLOAD_MARKER).unlink(missing_ok=True)
                future = self._executor.submit(self._execute, job_id)
                self._futures[job_id] = future
                future.add_done_callback(lambda completed: self._observe_future(job_id, completed))
                return self._public_record(record)
        except Exception:
            LOGGER.exception("Failed to queue reconstruction job %s for %s", job_id, safe_name)
            self._rollback_job_creation(job_id, input_dir, output_dir)
            raise

    def list_jobs(self) -> list[dict[str, object]]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._public_record(record) for record in records]

    def get_job(self, job_id: str) -> dict[str, object]:
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
            self._metadata_store.forget(job_id)

    def cancel_job(self, job_id: str) -> dict[str, object]:
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
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _write_upload(self, reader: BinaryIO, input_path: Path, content_length: int) -> None:
        remaining = content_length
        deadline = time.monotonic() + MAXIMUM_UPLOAD_TRANSFER_SECONDS
        with input_path.open("wb") as output_file:
            while remaining:
                if time.monotonic() >= deadline:
                    raise UploadValidationError("The upload exceeded the allowed transfer time")
                chunk = reader.read(min(remaining, UPLOAD_CHUNK_BYTES))
                if not chunk:
                    raise UploadValidationError("Upload ended before the declared file size")
                output_file.write(chunk)
                remaining -= len(chunk)

    def _execute(self, job_id: str) -> None:
        try:
            started_job = self._begin_job(job_id)
            if started_job is None:
                return
            record, cancellation_event = started_job
            output_path = self._run_processor(job_id, record, cancellation_event)
            self._complete_job(job_id, output_path)
        except CancellationRequestedError:
            LOGGER.info("Reconstruction job %s was cancelled", job_id)
            self._settle_cancelled(job_id)
        except Exception as error:
            if self._cancellation_requested(job_id):
                LOGGER.exception("Reconstruction job %s encountered an error while cancelling", job_id)
                self._settle_cancelled(job_id)
                return
            LOGGER.exception("Reconstruction job %s failed", job_id)
            self._settle_failed(job_id, _public_failure_message(error))

    def _settle_cancelled(self, job_id: str) -> None:
        try:
            self._cancel_running_job(job_id)
        except Exception:
            LOGGER.exception("Failed to persist cancellation for reconstruction job %s", job_id)

    def _settle_failed(self, job_id: str, message: str) -> None:
        try:
            self._fail_job(job_id, message)
        except Exception:
            LOGGER.exception("Failed to persist failure for reconstruction job %s", job_id)

    def _cancellation_requested(self, job_id: str) -> bool:
        with self._lock:
            cancellation_event = self._cancellation_events.get(job_id)
            record = self._jobs.get(job_id)
            event_is_set = cancellation_event is not None and cancellation_event.is_set()
            status_is_cancelling = record is not None and record.status is JobStatus.CANCELLING
            return event_is_set or status_is_cancelling

    def _observe_future(self, job_id: str, future: Future[None]) -> None:
        try:
            escaped_error = future.exception()
        except CancelledError:
            escaped_error = None
        if escaped_error is not None:
            LOGGER.error(
                "Unhandled worker error escaped reconstruction job %s",
                job_id,
                exc_info=(type(escaped_error), escaped_error, escaped_error.__traceback__),
            )
        with self._lock:
            if self._futures.get(job_id) is future:
                self._futures.pop(job_id, None)

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
        resolved_output_path = output_path.resolve()
        with self._lock:
            record = self._require_job(job_id)
            output_root = record.output_dir.resolve()
        if output_root not in resolved_output_path.parents:
            raise ValueError("The reconstruction worker returned an unmanaged output path")
        try:
            output_status = resolved_output_path.stat()
        except OSError as error:
            raise FileNotFoundError("The reconstruction worker did not produce an output video") from error
        if not stat.S_ISREG(output_status.st_mode) or output_status.st_size < 1:
            raise FileNotFoundError("The reconstruction worker did not produce an output video")
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
            record.output_path = resolved_output_path
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
            progress_advanced = next_progress > record.progress
            record.stage = ProcessingStage(stage)
            record.progress = next_progress
            if progress_advanced:
                record.progress_updated_at = utc_now()
                record.eta_seconds = self._estimate_eta(record)
            record.detail = detail
            self._record_activity(record)
            self._persist(record)

    def _fail_job(self, job_id: str, message: str) -> None:
        with self._lock:
            record = self._require_job(job_id)
            if self._cancellation_requested(job_id):
                self._mark_cancelled(record)
                return
            record.status = JobStatus.FAILED
            record.stage = ProcessingStage.FAILED
            record.detail = "Processing failed"
            record.error = message
            record.completed_at = utc_now()
            record.output_path = None
            record.eta_seconds = None
            record.progress_updated_at = record.completed_at
            self._record_activity(record)
            self._persist(record)

    def _estimate_eta(self, record: ProcessingJob) -> int | None:
        return estimate_eta(record)

    def _public_record(self, record: ProcessingJob) -> dict[str, object]:
        return build_public_job_record(record)

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

    def _persist(self, record: ProcessingJob, force: bool = False) -> None:
        self._metadata_store.persist(record, force=force)

    def _load_existing_jobs(self) -> None:
        for metadata_path in self.output_root.glob("*/job.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                record = self._record_from_storage(metadata_path.parent, payload)
            except (AttributeError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
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
        legacy_root = self.legacy_output_root.resolve()
        for output_path in self.legacy_output_root.rglob("*_reconstructed.mp4"):
            try:
                resolved_output = output_path.resolve()
                if legacy_root not in resolved_output.parents:
                    continue
                output_status = resolved_output.stat()
            except OSError:
                LOGGER.warning("Skipped a legacy output that disappeared during discovery: %s", output_path)
                continue
            if self.output_root in resolved_output.parents or "_work" in resolved_output.parts:
                continue
            if not stat.S_ISREG(output_status.st_mode) or output_status.st_size < 1:
                continue
            relative_path = resolved_output.relative_to(legacy_root)
            job_id = hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:32]
            completed_at = datetime.fromtimestamp(output_status.st_mtime, timezone.utc).isoformat()
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

    def _remove_incomplete_uploads(self) -> None:
        for input_directory in self.upload_root.iterdir():
            if not input_directory.is_dir():
                continue
            job_id = input_directory.name
            try:
                validate_job_identifier(job_id)
            except ValueError:
                continue
            marker_path = input_directory / INCOMPLETE_UPLOAD_MARKER
            if job_id in self._jobs:
                try:
                    marker_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception("Could not clear recovered upload marker for job %s", job_id)
                continue
            output_directory = self.output_root / job_id
            metadata_exists = (output_directory / "job.json").exists()
            if metadata_exists or (not marker_path.exists() and output_directory.exists()):
                LOGGER.error(
                    "Preserved untracked job directories because output state may be recoverable: %s",
                    job_id,
                )
                continue
            try:
                self._remove_job_directories(input_directory, output_directory)
            except (OSError, ValueError):
                LOGGER.exception("Could not remove incomplete upload for job %s", job_id)

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

    def _rollback_job_creation(self, job_id: str, input_dir: Path, output_dir: Path) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
            self._cancellation_events.pop(job_id, None)
            self._metadata_store.forget(job_id)
        try:
            self._remove_job_directories(input_dir, output_dir)
        except (OSError, ValueError):
            LOGGER.exception("Failed to remove directories for unqueued reconstruction job %s", job_id)

    def _remove_legacy_output(self, record: ProcessingJob) -> None:
        if self.legacy_output_root is None or record.output_path is None:
            raise ValueError("Legacy output root is unavailable")
        output_path = record.output_path.resolve()
        if self.legacy_output_root not in output_path.parents:
            raise ValueError(f"Refusing to delete unmanaged legacy output: {output_path}")
        if output_path.exists() and not output_path.is_file():
            raise ValueError(f"Refusing to delete non-file legacy output: {output_path}")
        output_path.unlink(missing_ok=True)

    @staticmethod
    def _remove_managed_tree(target: Path, root: Path) -> None:
        resolved_target = target.resolve()
        resolved_root = root.resolve()
        if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
            raise ValueError(f"Refusing to delete unmanaged path: {resolved_target}")
        if resolved_target.exists():
            shutil.rmtree(resolved_target)


def _same_activity(first: JobActivity, second: JobActivity) -> bool:
    return all(first[field_name] == second[field_name] for field_name in ("stage", "detail", "progress"))


def _validate_gap_policy(video: dict, configuration: dict) -> None:
    gap_configuration = configuration.get("gap")
    if not isinstance(gap_configuration, dict):
        return
    choose_hidden_gaps(
        total_frames=int(video["frames"]),
        fps=float(video["fps"]),
        rng=random.Random(0),
        missing_fraction=float(gap_configuration.get("missing_fraction", 0.25)),
        min_gap_seconds=float(gap_configuration.get("min_seconds", 1.0)),
        max_gap_seconds=float(gap_configuration.get("max_seconds", 3.0)),
        context_seconds=float(gap_configuration.get("context_seconds", 2.0)),
    )


def _public_failure_message(error: Exception) -> str:
    if isinstance(error, (BlenderUnavailableError, MediaToolUnavailableError, UnsupportedVideoTimingError)):
        first_line = next((line.strip() for line in str(error).splitlines() if line.strip()), "")
        return first_line[:PUBLIC_FAILURE_MESSAGE_LIMIT] or GENERIC_FAILURE_MESSAGE
    if isinstance(error, BlenderRenderError):
        return "Blender rendering failed. Check the job render log for details."
    if isinstance(error, MediaProcessingError):
        return "Final video encoding or validation failed. Check the job log for details."
    return GENERIC_FAILURE_MESSAGE
