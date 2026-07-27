"""FastAPI backend for the local reconstruction interface.

Wraps the existing `JobManager` rather than reimplementing job handling: uploads,
queueing, progress, cancellation, and persistence across restarts already work and are
tested. What this adds is the HTTP surface a browser needs, and a live event stream so
the interface shows the pipeline moving instead of a spinner.

Run it with:

    python -m uvicorn interfaces.api.app:app --app-dir src --port 8000 --reload

**Local only.** It binds loopback, is unauthenticated, and accepts video uploads that it
decodes and executes a heavy pipeline over. It is a developer tool for a single user on
their own machine, not something to put on a network.
"""

import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from application.processing_jobs import (
    JobConflictError,
    JobManager,
    JobNotFoundError,
)
from application.reconstruction_pipeline import load_config
from domain.actor_library import library_state
from domain.actor_proxies import catalog_report
from domain.video_upload import UploadValidationError
from interfaces.api import artifacts


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs"
LIBRARY_DIRECTORY = PROJECT_ROOT / "assets" / "actors"

# How often the event stream re-reads job state. Fast enough to feel live, slow enough
# that a browser tab left open overnight is not a busy loop.
STREAM_POLL_SECONDS = 0.5
# The interface is served by Vite in development, on its own origin.
DEVELOPMENT_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Shut the worker pool down cleanly so a running render is not orphaned."""
    yield
    if _manager is not None:
        _manager.shutdown()


app = FastAPI(
    title="Forensic 3D Reconstruction",
    description="Local API for evidence-grounded video gap reconstruction",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(DEVELOPMENT_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_manager: JobManager | None = None


def manager() -> JobManager:
    """One manager for the process, created on first use.

    Deferred rather than created at import so the module can be imported by tests and
    tooling without spinning up a worker pool and touching the filesystem.
    """
    global _manager
    if _manager is None:
        _manager = JobManager(
            upload_root=UPLOAD_ROOT,
            output_root=OUTPUT_ROOT,
            config_data=load_config(),
        )
    return _manager




def _job_or_404(job_id: str) -> dict:
    try:
        return manager().get_job(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _output_dir(job_id: str) -> Path:
    _job_or_404(job_id)
    return OUTPUT_ROOT / job_id


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """Everything the interface needs to tell the operator what is ready."""
    from infrastructure.blender_runner import find_blender_executable
    from infrastructure.media_tools import find_media_tool

    def _probe(callable_):
        try:
            return {"available": True, "path": str(callable_())}
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return {"available": False, "detail": str(error)}

    return {
        "status": "ok",
        "blender": _probe(find_blender_executable),
        "ffmpeg": _probe(lambda: find_media_tool("ffmpeg")),
        "ffprobe": _probe(lambda: find_media_tool("ffprobe")),
        "actor_library": library_state(LIBRARY_DIRECTORY),
        "renderable_classes": catalog_report(),
    }


@app.get("/api/config")
def configuration() -> dict:
    config = load_config()
    return {
        "gap": config["gap"],
        "renderer": config["renderer"],
        "yolo": {
            key: value for key, value in config["yolo"].items() if key != "classes"
        },
        "reasoning_enabled": config["reasoning"]["enabled"],
    }


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": manager().list_jobs()}


@app.post("/api/jobs", status_code=201)
async def create_job(video: UploadFile = File(...)) -> dict:
    """Accept an upload and queue it. Returns as soon as the job is queued."""
    payload = await video.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    try:
        # Off the event loop: the manager writes the file and probes it with ffprobe.
        return await asyncio.to_thread(
            manager().create_job, video.filename or "upload.mp4",
            io.BytesIO(payload), len(payload),
        )
    except UploadValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_or_404(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        return manager().cancel_job(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except JobConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    try:
        manager().delete_job(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except JobConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Server-sent events carrying job state and artifacts as they appear.

    Polls rather than hooking the pipeline's callback, because the pipeline runs in a
    worker thread and the artifacts it writes are the durable record — reading those
    means a browser that reconnects sees the true current state rather than having
    missed the events it was away for.
    """
    _job_or_404(job_id)

    async def events():
        previous: str | None = None
        while True:
            if await request.is_disconnected():
                return
            try:
                job = manager().get_job(job_id)
            except JobNotFoundError:
                yield "event: deleted\ndata: {}\n\n"
                return
            output_dir = OUTPUT_ROOT / job_id
            payload = {
                "job": job,
                "timeline": artifacts.timeline(output_dir),
                "clues": artifacts.clues(output_dir),
                "story": artifacts.story(output_dir),
                "render": artifacts.render_progress(output_dir),
                "has_plate": artifacts.plate_path(output_dir) is not None,
            }
            encoded = json.dumps(payload, default=str)
            if encoded != previous:
                previous = encoded
                yield f"event: update\ndata: {encoded}\n\n"
            if job.get("status") in {"completed", "failed", "cancelled"}:
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/timeline")
def job_timeline(job_id: str) -> dict:
    return {"timeline": artifacts.timeline(_output_dir(job_id))}


@app.get("/api/jobs/{job_id}/clues")
def job_clues(job_id: str) -> dict:
    return {"clues": artifacts.clues(_output_dir(job_id))}


@app.get("/api/jobs/{job_id}/story")
def job_story(job_id: str) -> dict:
    return {"story": artifacts.story(_output_dir(job_id))}


@app.get("/api/jobs/{job_id}/diagnostics")
def job_diagnostics(job_id: str) -> dict:
    return {"diagnostics": artifacts.diagnostics(_output_dir(job_id))}


def _file_or_404(path: Path | None, media_type: str, description: str) -> FileResponse:
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"{description} is not available yet.")
    return FileResponse(path, media_type=media_type)


@app.get("/api/jobs/{job_id}/plate")
def job_plate(job_id: str) -> FileResponse:
    return _file_or_404(
        artifacts.plate_path(_output_dir(job_id)), "image/png", "The recovered plate",
    )


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    try:
        path = manager().output_path(job_id)
    except (JobNotFoundError, JobConflictError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return _file_or_404(path, "video/mp4", "The reconstructed video")


@app.get("/api/jobs/{job_id}/gaps/{gap_index}/video")
def job_gap_video(job_id: str, gap_index: int) -> FileResponse:
    return _file_or_404(
        artifacts.gap_video_path(_output_dir(job_id), gap_index),
        "video/mp4", f"Gap {gap_index}",
    )


@app.get("/api/jobs/{job_id}/gaps/{gap_index}/truth")
def job_gap_truth(job_id: str, gap_index: int) -> FileResponse:
    """The hidden footage, for comparison. Presentation only — never fed back in."""
    return _file_or_404(
        artifacts.truth_video_path(_output_dir(job_id), gap_index),
        "video/mp4", f"Hidden footage for gap {gap_index}",
    )
