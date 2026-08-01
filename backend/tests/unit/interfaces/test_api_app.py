"""The HTTP surface, exercised through FastAPI's test client.

The manager is replaced with a fake so these stay fast and never launch a render. What
is under test is the API contract the browser depends on: status codes, payload shapes,
and that an artifact which does not exist yet is a clean 404 rather than a crash.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from application.processing_jobs import JobConflictError, JobNotFoundError  # noqa: E402
from interfaces.api import app as api  # noqa: E402

JOB = {
    "id": "abc123", "source_name": "clip.mp4", "status": "processing",
    "stage": "detecting", "progress": 0.4, "detail": "Tracking",
}


class FakeManager:
    def __init__(self) -> None:
        self.jobs = {JOB["id"]: dict(JOB)}
        self.cancelled: list[str] = []
        self.deleted: list[str] = []
        self.created: list[str] = []

    def list_jobs(self):
        return list(self.jobs.values())

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise JobNotFoundError(f"Unknown job {job_id}")
        return self.jobs[job_id]

    def create_job(self, source_name, reader, content_length):
        self.created.append(source_name)
        return dict(JOB, source_name=source_name, status="queued")

    def cancel_job(self, job_id):
        self.get_job(job_id)
        self.cancelled.append(job_id)
        return dict(self.jobs[job_id], status="cancelled")

    def delete_job(self, job_id):
        self.get_job(job_id)
        self.deleted.append(job_id)

    def output_path(self, job_id):
        self.get_job(job_id)
        raise JobConflictError("Job has not completed")

    def shutdown(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeManager()
    monkeypatch.setattr(api, "manager", lambda: fake)
    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path / "outputs")
    with fastapi_testclient.TestClient(api.app) as test_client:
        test_client.fake = fake
        yield test_client


class TestSystem:
    def test_health_reports_every_dependency(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert {"blender", "ffmpeg", "ffprobe"} <= set(payload)

    def test_health_reports_the_prebuilt_library_state(self, client):
        library = client.get("/api/health").json()["actor_library"]
        assert "available" in library and "catalog_digest" in library

    def test_health_lists_the_classes_that_can_be_rendered(self, client):
        classes = client.get("/api/health").json()["renderable_classes"]
        assert any(item["class_name"] == "person" for item in classes)

    def test_configuration_hides_the_eighty_class_name_table(self, client):
        """It is noise for the interface and would dominate the payload."""
        payload = client.get("/api/config").json()
        assert "classes" not in payload["yolo"]
        assert payload["gap"]["missing_fraction"] == 0.25


class TestJobs:
    def test_jobs_are_listed(self, client):
        assert client.get("/api/jobs").json()["jobs"][0]["id"] == "abc123"

    def test_a_job_is_fetched_by_id(self, client):
        assert client.get("/api/jobs/abc123").json()["stage"] == "detecting"

    def test_an_unknown_job_is_a_clean_404(self, client):
        assert client.get("/api/jobs/nope").status_code == 404

    def test_an_upload_creates_a_job(self, client):
        response = client.post(
            "/api/jobs", files={"video": ("clip.mp4", b"\x00" * 64, "video/mp4")},
        )
        assert response.status_code == 201
        assert client.fake.created == ["clip.mp4"]

    def test_an_empty_upload_is_rejected(self, client):
        response = client.post(
            "/api/jobs", files={"video": ("clip.mp4", b"", "video/mp4")},
        )
        assert response.status_code == 400

    def test_a_job_can_be_cancelled(self, client):
        assert client.post("/api/jobs/abc123/cancel").status_code == 200
        assert client.fake.cancelled == ["abc123"]

    def test_cancelling_an_unknown_job_is_a_404(self, client):
        assert client.post("/api/jobs/nope/cancel").status_code == 404

    def test_a_job_can_be_deleted(self, client):
        assert client.delete("/api/jobs/abc123").status_code == 204
        assert client.fake.deleted == ["abc123"]


class TestArtifacts:
    def test_the_timeline_is_null_before_gaps_are_chosen(self, client):
        assert client.get("/api/jobs/abc123/timeline").json()["timeline"] is None

    def test_clues_and_story_are_null_before_they_exist(self, client):
        assert client.get("/api/jobs/abc123/clues").json()["clues"] is None
        assert client.get("/api/jobs/abc123/story").json()["story"] is None

    def test_artifact_endpoints_404_for_an_unknown_job(self, client):
        assert client.get("/api/jobs/nope/timeline").status_code == 404

    def test_a_missing_plate_is_a_404_with_a_readable_reason(self, client):
        response = client.get("/api/jobs/abc123/plate")
        assert response.status_code == 404
        assert "not available yet" in response.json()["detail"]

    def test_an_incomplete_video_is_a_404_rather_than_a_500(self, client):
        assert client.get("/api/jobs/abc123/video").status_code == 404

    def test_a_missing_gap_video_is_a_404(self, client):
        assert client.get("/api/jobs/abc123/gaps/0/video").status_code == 404

    def test_a_missing_truth_clip_is_a_404(self, client):
        assert client.get("/api/jobs/abc123/gaps/0/truth").status_code == 404


class TestStream:
    def test_the_stream_ends_once_the_job_is_finished(self, client):
        client.fake.jobs["abc123"]["status"] = "completed"
        with client.stream("GET", "/api/jobs/abc123/stream") as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "event: update" in body
        assert "event: done" in body

    def test_the_stream_carries_job_state_and_artifact_slots(self, client):
        client.fake.jobs["abc123"]["status"] = "failed"
        with client.stream("GET", "/api/jobs/abc123/stream") as response:
            body = "".join(response.iter_text())
        assert '"timeline"' in body and '"clues"' in body and '"render"' in body

    def test_streaming_an_unknown_job_is_a_404(self, client):
        assert client.get("/api/jobs/nope/stream").status_code == 404
