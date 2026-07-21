import io
import json
import random
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.processing_jobs import (
    INCOMPLETE_UPLOAD_MARKER,
    JobManager,
    _public_failure_message,
    _replace_metadata_file,
)
from application.processing_job_view import live_eta
from application.reconstruction_pipeline import PipelineOptions, ProgressCallback
from domain.cancellation import raise_if_cancelled
from domain.processing_job import JobStatus, ProcessingJob
from domain.video_upload import UploadValidationError
from infrastructure.blender_runner import BlenderRenderError


JOB_COMPLETION_TIMEOUT_SECONDS = 5.0


def create_test_video(video_path: Path) -> bytes:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    for frame_index in range(8):
        frame = np.full((48, 64, 3), frame_index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path.read_bytes()


def copy_video_processor(
    video_path: Path,
    options: PipelineOptions,
    random_generator: random.Random,
    progress_callback: ProgressCallback | None,
) -> Path:
    del random_generator
    if progress_callback is not None:
        progress_callback("rendering", 0.75, "Creating test output")
    output_path = options.output_dir / f"{video_path.stem}_reconstructed.mp4"
    shutil.copyfile(video_path, output_path)
    return output_path


def cancellable_video_processor(
    video_path: Path,
    options: PipelineOptions,
    random_generator: random.Random,
    progress_callback: ProgressCallback | None,
) -> Path:
    del random_generator
    if progress_callback is not None:
        progress_callback("rendering", 0.5, "Waiting for cancellation")
    for _ in range(200):
        raise_if_cancelled(options.cancellation_check)
        time.sleep(0.01)
    return options.output_dir / f"{video_path.stem}_reconstructed.mp4"


def secondary_error_after_cancel_processor(
    video_path: Path,
    options: PipelineOptions,
    random_generator: random.Random,
    progress_callback: ProgressCallback | None,
) -> Path:
    del random_generator
    if progress_callback is not None:
        progress_callback("rendering", 0.5, "Waiting for cancellation")
    while options.cancellation_check is not None and not options.cancellation_check():
        time.sleep(0.01)
    raise OSError("secondary cleanup error with a private filesystem path")


class ProcessingJobManagerTests(unittest.TestCase):
    def test_blender_failure_message_does_not_expose_local_paths(self) -> None:
        error = BlenderRenderError(
            r"Blender render failed with exit code 1. See C:\private\job\blender.log"
        )

        message = _public_failure_message(error)

        self.assertEqual("Blender rendering failed. Check the job render log for details.", message)
        self.assertNotIn(r"C:\private", message)

    def test_too_short_video_is_rejected_before_it_is_queued(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={
                    "gap": {
                        "missing_fraction": 0.25,
                        "min_seconds": 1.0,
                        "max_seconds": 3.0,
                        "context_seconds": 2.0,
                    },
                },
                processor=copy_video_processor,
            )
            try:
                with self.assertRaisesRegex(UploadValidationError, "too short"):
                    manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                self.assertEqual([], manager.list_jobs())
                self.assertEqual([], list(manager.upload_root.iterdir()))
                self.assertEqual([], list(manager.output_root.iterdir()))
            finally:
                manager.shutdown()

    def test_invalid_job_metadata_cannot_abort_manager_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            invalid_job_directory = temporary_root / "outputs" / ("a" * 32)
            invalid_job_directory.mkdir(parents=True)
            (invalid_job_directory / "job.json").write_text("[]", encoding="utf-8")
            retained_upload_directory = temporary_root / "uploads" / ("a" * 32)
            retained_upload_directory.mkdir(parents=True)
            (retained_upload_directory / "source.mp4").write_bytes(b"retain")

            with self.assertLogs("application.processing_jobs", level="ERROR"):
                manager = JobManager(
                    temporary_root / "uploads",
                    temporary_root / "outputs",
                    config_data={},
                    processor=copy_video_processor,
                )
            try:
                self.assertEqual([], manager.list_jobs())
                self.assertTrue(invalid_job_directory.exists())
                self.assertTrue(retained_upload_directory.exists())
            finally:
                manager.shutdown()

    def test_startup_removes_untracked_upload_job_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            job_id = "b" * 32
            input_directory = temporary_root / "uploads" / job_id
            output_directory = temporary_root / "outputs" / job_id
            input_directory.mkdir(parents=True)
            output_directory.mkdir(parents=True)
            (input_directory / INCOMPLETE_UPLOAD_MARKER).touch()
            (input_directory / "partial.mp4").write_bytes(b"partial")
            orphan_job_id = "c" * 32
            orphan_input_directory = temporary_root / "uploads" / orphan_job_id
            orphan_output_directory = temporary_root / "outputs" / orphan_job_id
            orphan_input_directory.mkdir(parents=True)
            unmanaged_directory = temporary_root / "uploads" / "operator-notes"
            unmanaged_directory.mkdir(parents=True)

            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            try:
                self.assertFalse(input_directory.exists())
                self.assertFalse(output_directory.exists())
                self.assertFalse(orphan_input_directory.exists())
                self.assertFalse(orphan_output_directory.exists())
                self.assertTrue(unmanaged_directory.exists())
            finally:
                manager.shutdown()

    def test_upload_os_error_is_sanitized_and_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            try:
                with patch.object(
                    manager,
                    "_write_upload",
                    side_effect=OSError(r"Access denied: C:\private\upload\fixture.mp4"),
                ):
                    with self.assertLogs("application.processing_jobs", level="ERROR"):
                        with self.assertRaisesRegex(
                            UploadValidationError,
                            "could not be saved or decoded",
                        ) as context:
                            manager.create_job("fixture.mp4", io.BytesIO(b"video"), 5)
                self.assertNotIn(r"C:\private", str(context.exception))
                self.assertEqual([], list(manager.upload_root.iterdir()))
                self.assertEqual([], list(manager.output_root.iterdir()))
            finally:
                manager.shutdown()

    def test_job_completes_persists_and_deletes_owned_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            try:
                created_job = manager.create_job("Judge Clip.MP4", io.BytesIO(video_bytes), len(video_bytes))
                completed_job = self._wait_for_completion(manager, created_job["id"])
                self.assertEqual("completed", completed_job["status"])
                self.assertEqual("blender", completed_job["renderer_mode"])
                self.assertEqual(1.0, completed_job["progress"])
                self.assertEqual("finished", completed_job["eta_status"])
                activity_stages = [item["stage"] for item in completed_job["activity_log"]]
                self.assertIn("queued", activity_stages)
                self.assertIn("rendering", activity_stages)
                self.assertEqual("completed", activity_stages[-1])
                output_path = manager.output_path(created_job["id"])
                self.assertTrue(output_path.is_file())
                self.assertTrue((output_path.parent / "job.json").is_file())
                self.assertEqual([], list(output_path.parent.glob("*.tmp")))

                upload_directory = temporary_root / "uploads" / created_job["id"]
                output_directory = temporary_root / "outputs" / created_job["id"]
                manager.delete_job(created_job["id"])

                self.assertFalse(upload_directory.exists())
                self.assertFalse(output_directory.exists())
                self.assertEqual([], manager.list_jobs())
            finally:
                manager.shutdown()

    def test_job_persists_selected_fallback_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            try:
                created_job = manager.create_job(
                    "fallback.mp4", io.BytesIO(video_bytes), len(video_bytes), renderer_mode="2d"
                )
                completed_job = self._wait_for_completion(manager, created_job["id"])
                self.assertEqual("2d", completed_job["renderer_mode"])
            finally:
                manager.shutdown()

    def test_active_job_can_be_cancelled_then_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=cancellable_video_processor,
            )
            try:
                created_job = manager.create_job("cancel.mp4", io.BytesIO(video_bytes), len(video_bytes))
                self._wait_for_status(manager, created_job["id"], {"processing"})

                cancelling_job = manager.cancel_job(created_job["id"])
                cancelled_job = self._wait_for_status(manager, created_job["id"], {"cancelled"})

                self.assertIn(cancelling_job["status"], {"cancelling", "cancelled"})
                self.assertEqual("cancelled", cancelled_job["stage"])
                self.assertIsNone(cancelled_job["error"])
                self.assertEqual("cancelled", cancelled_job["activity_log"][-1]["stage"])
                manager.delete_job(created_job["id"])
                self.assertEqual([], manager.list_jobs())
            finally:
                manager.shutdown()

    def test_metadata_replace_retries_a_transient_windows_lock(self) -> None:
        temporary_path = Path("job.unique.tmp")
        metadata_path = Path("job.json")
        replace_results = [PermissionError("locked"), None]

        with patch.object(Path, "replace", side_effect=replace_results) as replace_mock:
            with patch("infrastructure.job_metadata_store.time.sleep") as sleep_mock:
                _replace_metadata_file(temporary_path, metadata_path)

        self.assertEqual(2, replace_mock.call_count)
        sleep_mock.assert_called_once_with(0.05)

    def test_live_eta_counts_down_then_reports_recalibration(self) -> None:
        record = ProcessingJob(
            job_id="a" * 32,
            source_name="fixture.mp4",
            input_path=Path("fixture.mp4"),
            output_dir=Path("output"),
            status=JobStatus.PROCESSING,
            eta_seconds=120,
            progress_updated_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        )

        eta_seconds, eta_status = live_eta(record)
        self.assertEqual("counting_down", eta_status)
        self.assertGreaterEqual(eta_seconds or 0, 109)
        self.assertLessEqual(eta_seconds or 0, 110)

        record.progress_updated_at = (datetime.now(timezone.utc) - timedelta(seconds=180)).isoformat()
        self.assertEqual((None, "recalibrating"), live_eta(record))

    def test_creation_rolls_back_state_and_directories_when_persistence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(temporary_root / "uploads", temporary_root / "outputs", config_data={})
            try:
                with patch.object(manager._metadata_store, "persist", side_effect=OSError("disk unavailable")):
                    with self.assertLogs("application.processing_jobs", level="ERROR"):
                        with self.assertRaises(OSError):
                            manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                self.assertEqual([], manager.list_jobs())
                self.assertEqual([], list(manager.upload_root.iterdir()))
                self.assertEqual([], list(manager.output_root.iterdir()))
            finally:
                manager.shutdown()

    def test_creation_rolls_back_state_and_directories_when_submission_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(temporary_root / "uploads", temporary_root / "outputs", config_data={})
            try:
                with patch.object(manager._executor, "submit", side_effect=RuntimeError("executor stopped")):
                    with self.assertLogs("application.processing_jobs", level="ERROR"):
                        with self.assertRaises(RuntimeError):
                            manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                self.assertEqual([], manager.list_jobs())
                self.assertEqual([], list(manager.upload_root.iterdir()))
                self.assertEqual([], list(manager.output_root.iterdir()))
            finally:
                manager.shutdown()

    def test_begin_persistence_failure_is_settled_and_observed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            original_persist = manager._metadata_store.persist

            def fail_processing_persist(record: ProcessingJob, force: bool = False) -> bool:
                if record.status is JobStatus.PROCESSING:
                    raise OSError("private metadata path")
                return original_persist(record, force)

            try:
                with patch.object(manager._metadata_store, "persist", side_effect=fail_processing_persist):
                    with self.assertLogs("application.processing_jobs", level="ERROR"):
                        created_job = manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                        failed_job = self._wait_for_status(manager, created_job["id"], {"failed"})
                self.assertEqual("Reconstruction failed. Check the server log for details.", failed_job["error"])
                self.assertNotIn("private metadata path", failed_job["error"])
                self._wait_for_future_cleanup(manager, created_job["id"])
            finally:
                manager.shutdown()

    def test_completion_persistence_failure_becomes_a_clean_failed_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=copy_video_processor,
            )
            original_persist = manager._metadata_store.persist

            def fail_completed_persist(record: ProcessingJob, force: bool = False) -> bool:
                if record.status is JobStatus.COMPLETED:
                    raise OSError("private completed metadata path")
                return original_persist(record, force)

            try:
                with patch.object(manager._metadata_store, "persist", side_effect=fail_completed_persist):
                    with self.assertLogs("application.processing_jobs", level="ERROR"):
                        created_job = manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                        failed_job = self._wait_for_status(manager, created_job["id"], {"failed"})
                self.assertEqual("Reconstruction failed. Check the server log for details.", failed_job["error"])
                self.assertIsNone(failed_job["output_url"])
                metadata_path = manager.output_root / created_job["id"] / "job.json"
                self.assertEqual("failed", json.loads(metadata_path.read_text())["status"])
                self._wait_for_future_cleanup(manager, created_job["id"])
            finally:
                manager.shutdown()

    def test_cancellation_wins_when_processor_raises_a_secondary_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            video_bytes = create_test_video(temporary_root / "fixture.mp4")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs",
                config_data={},
                processor=secondary_error_after_cancel_processor,
            )
            try:
                created_job = manager.create_job("fixture.mp4", io.BytesIO(video_bytes), len(video_bytes))
                self._wait_for_status(manager, created_job["id"], {"processing"})
                with self.assertLogs("application.processing_jobs", level="ERROR"):
                    manager.cancel_job(created_job["id"])
                    cancelled_job = self._wait_for_status(manager, created_job["id"], {"cancelled"})
                self.assertIsNone(cancelled_job["error"])
                self.assertEqual("cancelled", cancelled_job["stage"])
            finally:
                manager.shutdown()

    def _wait_for_completion(self, manager: JobManager, job_id: str) -> dict:
        return self._wait_for_status(manager, job_id, {"completed", "failed"})

    def _wait_for_status(self, manager: JobManager, job_id: str, statuses: set[str]) -> dict:
        deadline = time.monotonic() + JOB_COMPLETION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            job = manager.get_job(job_id)
            if job["status"] in statuses:
                return job
            time.sleep(0.02)
        self.fail("Processing job did not complete before the test timeout")

    def _wait_for_future_cleanup(self, manager: JobManager, job_id: str) -> None:
        deadline = time.monotonic() + JOB_COMPLETION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with manager._lock:
                if job_id not in manager._futures:
                    return
            time.sleep(0.01)
        self.fail("Completed worker future remained registered")

    def test_existing_reconstruction_deletes_only_its_video_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            legacy_directory = temporary_root / "outputs" / "old_preview"
            legacy_directory.mkdir(parents=True)
            legacy_output = legacy_directory / "judge_clip_reconstructed.mp4"
            legacy_output.write_bytes(b"legacy-video")
            sibling_file = legacy_directory / "comparison.jpg"
            sibling_file.write_bytes(b"keep-me")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs" / "jobs",
                config_data={},
                processor=copy_video_processor,
                legacy_output_root=temporary_root / "outputs",
            )
            try:
                legacy_jobs = manager.list_jobs()
                self.assertEqual(1, len(legacy_jobs))
                self.assertTrue(legacy_jobs[0]["is_legacy_output"])
                manager.delete_job(legacy_jobs[0]["id"])
                self.assertFalse(legacy_output.exists())
                self.assertTrue(sibling_file.exists())
                self.assertTrue(legacy_directory.exists())
            finally:
                manager.shutdown()

    def test_missing_legacy_output_can_be_removed_from_the_job_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            legacy_output = temporary_root / "outputs" / "old_reconstructed.mp4"
            legacy_output.parent.mkdir(parents=True)
            legacy_output.write_bytes(b"legacy-video")
            manager = JobManager(
                temporary_root / "uploads",
                temporary_root / "outputs" / "jobs",
                config_data={},
                processor=copy_video_processor,
                legacy_output_root=temporary_root / "outputs",
            )
            try:
                legacy_job = manager.list_jobs()[0]
                legacy_output.unlink()

                manager.delete_job(legacy_job["id"])

                self.assertEqual([], manager.list_jobs())
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
