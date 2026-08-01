import logging
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from domain.cancellation import CancellationCheck, CancellationRequestedError, raise_if_cancelled


BLENDER_EXECUTABLE_ENVIRONMENT_KEY = "BLENDER_EXECUTABLE"
WINDOWS_BLENDER_EXECUTABLE = Path(
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
)
DEFAULT_RENDER_STALL_TIMEOUT_SECONDS = 7_200
PROCESS_POLL_SECONDS = 0.2
PROCESS_TERMINATION_TIMEOUT_SECONDS = 5.0
PROGRESS_MARKER_PATTERN = re.compile(r"RECON_PROGRESS\s+(\d+)\s+(\d+)")
BlenderProgressCallback = Callable[[int, int], None]
LOGGER = logging.getLogger(__name__)


class BlenderUnavailableError(RuntimeError):
    pass


class BlenderRenderError(RuntimeError):
    pass


class RenderHeartbeat:
    def __init__(self) -> None:
        self._last_progress = time.monotonic()
        self._lock = threading.Lock()

    def mark_progress(self) -> None:
        with self._lock:
            self._last_progress = time.monotonic()

    def has_stalled(self, timeout_seconds: int) -> bool:
        with self._lock:
            seconds_without_progress = time.monotonic() - self._last_progress
        return seconds_without_progress >= timeout_seconds


@dataclass(frozen=True)
class BlenderRenderRequest:
    plan_path: Path
    output_path: Path
    report_path: Path
    blend_path: Path
    log_path: Path
    mode: str = "preview"


@dataclass(frozen=True)
class BlenderRenderResult:
    output_path: Path
    report_path: Path
    blend_path: Path
    log_path: Path


def find_blender_executable() -> Path:
    configured_path = os.environ.get(BLENDER_EXECUTABLE_ENVIRONMENT_KEY)
    candidates = [Path(configured_path)] if configured_path else []
    discovered_path = shutil.which("blender")
    candidates.extend([Path(discovered_path)] if discovered_path else [])
    candidates.append(WINDOWS_BLENDER_EXECUTABLE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BlenderUnavailableError(
        "Blender 4.5 LTS was not found. Set BLENDER_EXECUTABLE to its executable path."
    )


def render_with_blender(
    project_root: Path,
    request: BlenderRenderRequest,
    timeout_seconds: int = DEFAULT_RENDER_STALL_TIMEOUT_SECONDS,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: BlenderProgressCallback | None = None,
) -> BlenderRenderResult:
    _validate_request(request)
    blender_executable = find_blender_executable()
    render_script = (project_root / "blender" / "render_gap.py").resolve()
    if not render_script.is_file():
        raise BlenderRenderError(f"Blender render script is missing: {render_script}")
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_blender_command(blender_executable, render_script, request)
    return_code = _run_blender(
        command, request.log_path, project_root, timeout_seconds, cancellation_check, progress_callback,
    )
    if return_code != 0:
        raise BlenderRenderError(
            f"Blender render failed with exit code {return_code}. See {request.log_path}"
        )
    _validate_outputs(request)
    return BlenderRenderResult(request.output_path, request.report_path, request.blend_path, request.log_path)


def build_blender_command(executable: Path, script: Path, request: BlenderRenderRequest) -> list[str]:
    return [
        str(executable), "--background", "--python", str(script), "--",
        "--plan", str(request.plan_path.resolve()),
        "--output", str(request.output_path.resolve()),
        "--report", str(request.report_path.resolve()),
        "--blend", str(request.blend_path.resolve()),
        "--mode", request.mode,
    ]


def _run_blender(
    command: list[str],
    log_path: Path,
    project_root: Path,
    timeout_seconds: int,
    cancellation_check: CancellationCheck | None,
    progress_callback: BlenderProgressCallback | None = None,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    heartbeat = RenderHeartbeat()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        if process.stdout is None:
            raise BlenderRenderError("Blender output stream could not be opened")
        output_thread = _start_output_capture(process.stdout, log_file, progress_callback, heartbeat)
        try:
            return _wait_for_blender(
                process, deadline, timeout_seconds, cancellation_check, heartbeat,
            )
        finally:
            if process.poll() is None:
                _terminate_process(process)
            output_thread.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)


def _start_output_capture(
    output_stream: TextIO,
    log_file: TextIO,
    progress_callback: BlenderProgressCallback | None,
    heartbeat: RenderHeartbeat,
) -> threading.Thread:
    thread = threading.Thread(
        target=_capture_output,
        args=(output_stream, log_file, progress_callback, heartbeat),
        name="blender-output",
        daemon=True,
    )
    thread.start()
    return thread


def _capture_output(
    output_stream: TextIO,
    log_file: TextIO,
    progress_callback: BlenderProgressCallback | None,
    heartbeat: RenderHeartbeat,
) -> None:
    active_callback = progress_callback
    for line in output_stream:
        log_file.write(line)
        log_file.flush()
        progress = _parse_progress_line(line)
        if progress is None:
            continue
        heartbeat.mark_progress()
        if active_callback is None:
            continue
        try:
            active_callback(*progress)
        except Exception:
            LOGGER.exception("Disabling Blender frame progress after its callback failed")
            active_callback = None


def _parse_progress_line(line: str) -> tuple[int, int] | None:
    match = PROGRESS_MARKER_PATTERN.search(line)
    if match is None:
        return None
    current_frame, total_frames = (int(value) for value in match.groups())
    if total_frames < 1 or current_frame < 0:
        return None
    return min(current_frame, total_frames), total_frames


def _wait_for_blender(
    process: subprocess.Popen[str],
    deadline: float,
    timeout_seconds: int,
    cancellation_check: CancellationCheck | None,
    heartbeat: RenderHeartbeat | None = None,
) -> int:
    while process.poll() is None:
        if cancellation_check is not None and cancellation_check():
            _terminate_process(process)
            raise CancellationRequestedError("Blender rendering was cancelled")
        timeout_expired = (
            heartbeat.has_stalled(timeout_seconds)
            if heartbeat is not None
            else time.monotonic() >= deadline
        )
        if timeout_expired:
            _terminate_process(process)
            raise BlenderRenderError(
                f"Blender produced no frame progress for {timeout_seconds} seconds"
            )
        time.sleep(PROCESS_POLL_SECONDS)
    raise_if_cancelled(cancellation_check)
    return int(process.returncode or 0)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    _signal_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process(process, signal.SIGKILL)
        process.wait()


def _signal_process(process: subprocess.Popen[str], signal_number: signal.Signals) -> None:
    if os.name == "nt":
        if signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal_number)
    except ProcessLookupError:
        return


def _validate_request(request: BlenderRenderRequest) -> None:
    if request.mode not in {"preview", "animation", "sparse_animation"}:
        raise ValueError("Blender render mode must be preview, animation, or sparse_animation")
    if not request.plan_path.is_file():
        raise FileNotFoundError(f"Reconstruction plan is missing: {request.plan_path}")
    if request.plan_path.resolve() == request.output_path.resolve():
        raise ValueError("Plan and render output paths must be different")


def _validate_outputs(request: BlenderRenderRequest) -> None:
    render_exists = (
        request.output_path.is_dir()
        if request.mode == "sparse_animation"
        else request.output_path.is_file()
    )
    missing_paths = []
    if not render_exists:
        missing_paths.append(request.output_path)
    missing_paths.extend(
        output_path for output_path in (request.report_path, request.blend_path)
        if not output_path.is_file()
    )
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise BlenderRenderError(f"Blender finished without required outputs: {missing_text}")
