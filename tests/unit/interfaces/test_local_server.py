import json
from http.client import HTTPConnection
import random
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.processing_jobs import JobManager
from application.reconstruction_pipeline import PipelineOptions, ProgressCallback
from domain.cancellation import raise_if_cancelled
from interfaces.http.local_server import _download_content_disposition, build_server


REQUEST_TIMEOUT_SECONDS = 3.0
JOB_COMPLETION_TIMEOUT_SECONDS = 5.0


def create_test_video(video_path: Path) -> bytes:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    for frame_index in range(8):
        frame = np.full((48, 64, 3), 30 + frame_index * 10, dtype=np.uint8)
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
        progress_callback("rendering", 0.8, "Creating HTTP fixture output")
    for _ in range(15):
        raise_if_cancelled(options.cancellation_check)
        time.sleep(0.01)
    output_path = options.output_dir / f"{video_path.stem}_reconstructed.mp4"
    shutil.copyfile(video_path, output_path)
    return output_path


class LocalServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)
        self.manager = JobManager(
            self.temporary_root / "uploads",
            self.temporary_root / "outputs",
            config_data={},
            processor=copy_video_processor,
        )
        self.server = build_server(("127.0.0.1", 0), self.manager, PROJECT_ROOT / "web")
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.manager.shutdown()
        self.temporary_directory.cleanup()

    def test_upload_playback_range_and_physical_deletion(self) -> None:
        video_bytes = create_test_video(self.temporary_root / "fixture.mp4")
        upload_request = Request(
            f"{self.base_url}/api/jobs",
            data=video_bytes,
            method="POST",
            headers={"X-File-Name": "judge-video.mp4", "Content-Type": "video/mp4"},
        )
        with urlopen(upload_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(202, response.status)
            job_id = json.load(response)["job"]["id"]

        completed_job = self._wait_for_completion(job_id)
        self.assertEqual("completed", completed_job["status"])

        range_request = Request(f"{self.base_url}{completed_job['output_url']}", headers={"Range": "bytes=0-9"})
        with urlopen(range_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(206, response.status)
            self.assertEqual("bytes 0-9/", response.headers["Content-Range"][:10])
            self.assertEqual(10, len(response.read()))

        delete_request = Request(f"{self.base_url}/api/jobs/{job_id}", method="DELETE")
        with urlopen(delete_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(200, response.status)
        self.assertFalse((self.temporary_root / "outputs" / job_id).exists())
        self.assertFalse((self.temporary_root / "uploads" / job_id).exists())

    def test_serves_nested_frontend_asset(self) -> None:
        with urlopen(f"{self.base_url}/assets/scripts/app.js", timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(200, response.status)
            self.assertIn(b"fetchProcessingJobs", response.read())
        with urlopen(f"{self.base_url}/", timeout=REQUEST_TIMEOUT_SECONDS) as response:
            page_content = response.read()
            self.assertIn(b'id="theme-toggle"', page_content)
            self.assertIn(b'id="renderer-mode"', page_content)
        with urlopen(f"{self.base_url}/assets/styles/app.css", timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertIn(b'[data-theme="light"]', response.read())
        with urlopen(f"{self.base_url}/assets/scripts/api-client.js", timeout=REQUEST_TIMEOUT_SECONDS) as response:
            api_client_content = response.read()
            self.assertIn(b'X-Renderer-Mode', api_client_content)
            self.assertIn(b'cancelProcessingJob', api_client_content)
        with urlopen(f"{self.base_url}/assets/icons/favicon.svg", timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(200, response.status)

    def test_rejects_unsupported_upload_before_queueing(self) -> None:
        request = Request(
            f"{self.base_url}/api/jobs",
            data=b"not a video",
            method="POST",
            headers={"X-File-Name": "evidence.txt"},
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(400, context.exception.code)
        self.assertEqual([], self.manager.list_jobs())

    def test_rejects_non_loopback_host_header(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=REQUEST_TIMEOUT_SECONDS)
        connection.putrequest("GET", "/api/health", skip_host=True)
        connection.putheader("Host", "attacker.example")
        connection.endheaders()
        response = connection.getresponse()
        try:
            self.assertEqual(421, response.status)
        finally:
            response.read()
            connection.close()

    def test_server_refuses_non_loopback_bind_address(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            build_server(("0.0.0.0", 0), self.manager, PROJECT_ROOT / "web")

    def test_download_header_supports_unicode_legacy_filename(self) -> None:
        header = _download_content_disposition("测试 reconstruction.mp4")

        header.encode("latin-1")
        self.assertIn("filename*=UTF-8''%E6%B5%8B%E8%AF%95%20reconstruction.mp4", header)

    def test_shutdown_cancels_partial_http_requests(self) -> None:
        connection = socket.create_connection(("127.0.0.1", self.server.server_port), timeout=REQUEST_TIMEOUT_SECONDS)
        connection.sendall(b"GET /api/health HTTP/1.1\r\nHost:")
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with self.server._active_connections_lock:
                if self.server._active_connections:
                    break
            time.sleep(0.01)
        else:
            self.fail("Partial HTTP request was not tracked")

        self.server.cancel_active_requests()
        connection.close()
        while time.monotonic() < deadline:
            with self.server._active_connections_lock:
                if not self.server._active_connections:
                    return
            time.sleep(0.01)
        self.fail("Cancelled HTTP request did not exit")

    def test_oversized_range_is_rejected_cleanly(self) -> None:
        video_bytes = create_test_video(self.temporary_root / "range-fixture.mp4")
        upload_request = Request(
            f"{self.base_url}/api/jobs",
            data=video_bytes,
            method="POST",
            headers={"X-File-Name": "range-video.mp4", "Content-Type": "video/mp4"},
        )
        with urlopen(upload_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            job_id = json.load(response)["job"]["id"]
        completed_job = self._wait_for_completion(job_id)
        range_header = f"bytes={'9' * 100}-"
        request = Request(f"{self.base_url}{completed_job['output_url']}", headers={"Range": range_header})
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(416, context.exception.code)

    def test_cancel_persistence_error_returns_clean_server_error(self) -> None:
        with patch.object(self.manager, "cancel_job", side_effect=OSError("private path")):
            request = Request(f"{self.base_url}/api/jobs/{'a' * 32}/cancel", method="POST", data=b"")
            with self.assertLogs("interfaces.http.local_server", level="ERROR"):
                with self.assertRaises(HTTPError) as context:
                    urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(500, context.exception.code)
        self.assertNotIn("private path", context.exception.read().decode("utf-8"))

    def test_active_job_can_be_cancelled_through_api(self) -> None:
        video_bytes = create_test_video(self.temporary_root / "cancel-fixture.mp4")
        upload_request = Request(
            f"{self.base_url}/api/jobs",
            data=video_bytes,
            method="POST",
            headers={"X-File-Name": "cancel-video.mp4", "Content-Type": "video/mp4"},
        )
        with urlopen(upload_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            job_id = json.load(response)["job"]["id"]

        cancel_request = Request(f"{self.base_url}/api/jobs/{job_id}/cancel", method="POST", data=b"")
        with urlopen(cancel_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            self.assertEqual(202, response.status)

        cancelled_job = self._wait_for_completion(job_id)
        self.assertEqual("cancelled", cancelled_job["status"])

    def _wait_for_completion(self, job_id: str) -> dict:
        deadline = time.monotonic() + JOB_COMPLETION_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with urlopen(f"{self.base_url}/api/jobs/{job_id}", timeout=REQUEST_TIMEOUT_SECONDS) as response:
                job = json.load(response)["job"]
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(0.02)
        self.fail("HTTP processing job did not complete before the test timeout")


if __name__ == "__main__":
    unittest.main()
