"""Atomically persists processing-job metadata without frame-rate disk writes."""

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from domain.processing_job import ProcessingJob


METADATA_REPLACE_ATTEMPTS = 8
METADATA_RETRY_BASE_SECONDS = 0.05
NONTERMINAL_PERSIST_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class PersistedJobState:
    status: str
    stage: str
    written_at_seconds: float


class JobMetadataStore:
    def __init__(self, persist_interval_seconds: float = NONTERMINAL_PERSIST_INTERVAL_SECONDS) -> None:
        if persist_interval_seconds < 0:
            raise ValueError("Metadata persistence interval cannot be negative")
        self._persist_interval_seconds = persist_interval_seconds
        self._persisted_states: dict[str, PersistedJobState] = {}
        self._lock = threading.Lock()

    def persist(self, record: ProcessingJob, force: bool = False) -> bool:
        with self._lock:
            written_at_seconds = time.monotonic()
            previous_state = self._persisted_states.get(record.job_id)
            if not self._should_persist(record, previous_state, written_at_seconds, force):
                return False
            _write_metadata(record)
            self._persisted_states[record.job_id] = PersistedJobState(
                status=record.status.value,
                stage=record.stage.value,
                written_at_seconds=written_at_seconds,
            )
            return True

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._persisted_states.pop(job_id, None)

    def _should_persist(
        self,
        record: ProcessingJob,
        previous_state: PersistedJobState | None,
        written_at_seconds: float,
        force: bool,
    ) -> bool:
        if force or previous_state is None:
            return True
        if previous_state.status != record.status.value or previous_state.stage != record.stage.value:
            return True
        elapsed_seconds = written_at_seconds - previous_state.written_at_seconds
        return elapsed_seconds >= self._persist_interval_seconds


def _write_metadata(record: ProcessingJob) -> None:
    metadata_path = record.output_dir / "job.json"
    temporary_path = record.output_dir / f"job.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(record.to_storage_payload(), metadata_file, indent=2)
        replace_metadata_file(temporary_path, metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def replace_metadata_file(temporary_path: Path, metadata_path: Path) -> None:
    for attempt_index in range(METADATA_REPLACE_ATTEMPTS):
        try:
            temporary_path.replace(metadata_path)
            return
        except PermissionError:
            if attempt_index == METADATA_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(METADATA_RETRY_BASE_SECONDS * (attempt_index + 1))
