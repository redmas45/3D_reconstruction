import io
import random
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.processing_jobs import JobManager
from application.reconstruction_pipeline import PipelineOptions, ProgressCallback


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


class ProcessingJobManagerTests(unittest.TestCase):
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
                output_path = manager.output_path(created_job["id"])
                self.assertTrue(output_path.is_file())
                self.assertTrue((output_path.parent / "job.json").is_file())

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

    def _wait_for_completion(self, manager: JobManager, job_id: str) -> dict:
        deadline = time.monotonic() + JOB_COMPLETION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            job = manager.get_job(job_id)
            if job["status"] in {"completed", "failed"}:
                return job
            time.sleep(0.02)
        self.fail("Processing job did not complete before the test timeout")

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


if __name__ == "__main__":
    unittest.main()
