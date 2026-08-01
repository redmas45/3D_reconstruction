"""Host-side client for the persistent Blender service (Implementation_plan.md §6).

Replaces the per-gap `blender --background` launch. One process is started for a whole
job and kept warm, so Blender startup, shader compilation, and OptiX kernel compilation
are paid once instead of once per gap — measured at roughly 7–8 minutes of waste per
job in §6.1, and confirmed by the M0 benchmark's 8x first-frame penalty.

Responsibilities kept here rather than in the caller:

  * process lifecycle and the stdout reader thread
  * routing events to the request that is waiting for them
  * stall detection, using any event for a request as its heartbeat (§6.7)
  * crash detection, so a dead process fails pending requests instead of hanging

Resume is deliberately *not* handled here. The service reports what it rendered; the
render loop owns which frames still need doing, because that decision belongs to the
cache layer (§9 L6).
"""

import itertools
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from infrastructure.blender_protocol import (
    EVENT_ERROR,
    EVENT_READY,
    MINIMUM_VALID_PNG_BYTES,
    PNG_MAGIC,
    PROTOCOL_VERSION,
    ProtocolError,
    ServiceCommand,
    ServiceEvent,
    close_job_command,
    decode_event,
    encode_command,
    frame_filename,
    hello_command,
    open_job_command,
    prepare_gap_command,
    render_frames_command,
    reset_gap_command,
    shutdown_command,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_STALL_TIMEOUT_SECONDS = 300.0
SHUTDOWN_GRACE_SECONDS = 10.0
TERMINATE_GRACE_SECONDS = 5.0
READER_POLL_SECONDS = 0.1

ProgressCallback = Callable[[dict], None]


class BlenderServiceError(RuntimeError):
    """Base class for every failure of the persistent Blender service."""


class BlenderServiceCrashed(BlenderServiceError):
    """The Blender process exited while a request was outstanding."""


class BlenderServiceStalled(BlenderServiceError):
    """No event arrived for a request within the stall timeout."""


class BlenderCommandFailed(BlenderServiceError):
    """Blender received the command and reported a failure."""


class ProcessLike(Protocol):
    """The subset of `subprocess.Popen` this client depends on."""

    stdin: object
    stdout: object

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


ProcessFactory = Callable[[], ProcessLike]


def build_service_command(
    blender_executable: Path, service_script: Path, job_root: Path,
) -> list[str]:
    """`--factory-startup` skips user prefs and add-ons: faster, and deterministic (§6.2)."""
    return [
        str(blender_executable),
        "--background",
        "--factory-startup",
        "--python", str(service_script),
        "--",
        "--job-root", str(job_root),
        "--protocol-version", str(PROTOCOL_VERSION),
    ]


class BlenderService:
    """A warm Blender process driven by line-delimited JSON."""

    def __init__(
        self,
        process_factory: ProcessFactory,
        log_path: Path | None = None,
        stall_timeout_seconds: float = DEFAULT_STALL_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self._process_factory = process_factory
        self._log_path = log_path
        self._stall_timeout_seconds = stall_timeout_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self._process: ProcessLike | None = None
        self._reader: threading.Thread | None = None
        self._mailboxes: dict[str, queue.Queue] = {}
        self._mailbox_lock = threading.Lock()
        self._request_counter = itertools.count(1)
        self._log_file = None
        self._ready_event = threading.Event()
        self._stream_closed = threading.Event()
        self._exit_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict:
        """Launch Blender and wait for its ready banner. Returns its capability report."""
        if self._process is not None:
            raise BlenderServiceError("Service is already running")
        self._exit_reason = None
        self._ready_event.clear()
        self._stream_closed.clear()
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8", errors="replace")
        self._process = self._process_factory()
        self._reader = threading.Thread(target=self._read_stdout, name="blender-stdout", daemon=True)
        self._reader.start()
        if not self._ready_event.wait(self._startup_timeout_seconds):
            self._force_stop()
            raise BlenderServiceStalled(
                f"Blender did not report ready within {self._startup_timeout_seconds:.0f}s"
            )
        return self.request(hello_command(self._next_request_id()))

    @property
    def is_alive(self) -> bool:
        """Usable for requests.

        A closed stdout is treated as dead even before the OS has reaped the process.
        Otherwise a caller that catches a crash and immediately checks `is_alive` to
        decide whether to restart would race the reaper and get `True` for a corpse.
        """
        if self._process is None or self._stream_closed.is_set():
            return False
        return self._process.poll() is None

    def shutdown(self, timeout_seconds: float = SHUTDOWN_GRACE_SECONDS) -> None:
        """Ask Blender to exit, then escalate. Safe to call more than once."""
        if self._process is None:
            return
        if self.is_alive:
            try:
                self._send(shutdown_command(self._next_request_id()))
                self._process.wait(timeout=timeout_seconds)
            except (BlenderServiceError, subprocess.TimeoutExpired, OSError):
                self._force_stop()
        self._cleanup()

    def abort(self) -> None:
        """Stop Blender immediately, failing any outstanding request.

        Used for operator cancellation, and by the §6.7 stall path. Completed frames
        stay on disk, so the next run resumes from them rather than restarting the gap.
        """
        self._force_stop()

    def restart(self) -> dict:
        """Recycle the process — used after a crash and for the §6.6 periodic recycle."""
        self.shutdown()
        return self.start()

    def __enter__(self) -> "BlenderService":
        self.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.shutdown()

    # -- requests ----------------------------------------------------------

    def request(
        self,
        command: ServiceCommand,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Send a command and wait for its terminal event.

        Any event for this request — including progress — resets the stall deadline,
        so a long render is fine but a silent one is not (§6.7).
        """
        if not self.is_alive:
            raise BlenderServiceCrashed(self._exit_reason or "Blender service is not running")
        mailbox: queue.Queue = queue.Queue()
        with self._mailbox_lock:
            self._mailboxes[command.request_id] = mailbox
        try:
            self._send(command)
            return self._await_result(command, mailbox, timeout_seconds, on_progress)
        finally:
            with self._mailbox_lock:
                self._mailboxes.pop(command.request_id, None)

    def _await_result(
        self,
        command: ServiceCommand,
        mailbox: queue.Queue,
        timeout_seconds: float,
        on_progress: ProgressCallback | None,
    ) -> dict:
        overall_deadline = time.monotonic() + timeout_seconds
        while True:
            wait_seconds = self._next_wait_seconds(overall_deadline)
            try:
                event = mailbox.get(timeout=wait_seconds)
            except queue.Empty:
                self._force_stop()
                raise BlenderServiceStalled(
                    f"No response to '{command.command}' within "
                    f"{min(self._stall_timeout_seconds, timeout_seconds):.0f}s"
                ) from None
            if event is None:
                raise BlenderServiceCrashed(
                    self._exit_reason or f"Blender exited during '{command.command}'"
                )
            if event.is_terminal:
                return self._terminal_payload(command, event)
            if on_progress is not None:
                on_progress(event.payload)

    def _next_wait_seconds(self, overall_deadline: float) -> float:
        remaining_overall = max(0.0, overall_deadline - time.monotonic())
        return min(self._stall_timeout_seconds, remaining_overall) or READER_POLL_SECONDS

    @staticmethod
    def _terminal_payload(command: ServiceCommand, event: ServiceEvent) -> dict:
        if event.event == EVENT_ERROR:
            raise BlenderCommandFailed(f"{command.command}: {event.error_message}")
        return event.payload

    def _send(self, command: ServiceCommand) -> None:
        if self._process is None or self._process.stdin is None:
            raise BlenderServiceCrashed("Blender service has no input stream")
        try:
            self._process.stdin.write(encode_command(command))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            raise BlenderServiceCrashed(f"Could not send '{command.command}': {error}") from error

    def _next_request_id(self) -> str:
        return f"req-{next(self._request_counter)}"

    # -- stdout pump -------------------------------------------------------

    def _read_stdout(self) -> None:
        """Route protocol events; tee everything else to the log."""
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._consume_line(line)
        except (OSError, ValueError) as error:
            LOGGER.warning("Blender stdout reader stopped: %s", error)
        finally:
            self._stream_closed.set()
            self._reap_process()
            self._exit_reason = self._describe_exit()
            self._release_waiters()

    def _reap_process(self) -> None:
        """Collect the exit status so `_describe_exit` can report a real code."""
        if self._process is None:
            return
        try:
            self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            LOGGER.debug("Blender did not exit promptly after closing stdout")

    def _consume_line(self, line: str) -> None:
        try:
            event = decode_event(line)
        except ProtocolError as error:
            # A corrupt protocol line must not be mistaken for log noise; record it and
            # let the waiting request time out rather than silently hanging forever.
            LOGGER.error("Malformed Blender protocol line: %s", error)
            self._write_log(line)
            return
        if event is None:
            self._write_log(line)
            return
        if event.event == EVENT_READY:
            self._ready_event.set()
            return
        self._deliver(event)

    def _deliver(self, event: ServiceEvent) -> None:
        with self._mailbox_lock:
            mailbox = self._mailboxes.get(event.request_id or "")
        if mailbox is None:
            LOGGER.debug("Dropping event for unknown request %s", event.request_id)
            return
        mailbox.put(event)

    def _release_waiters(self) -> None:
        """Wake every pending request so a crash surfaces immediately, not on timeout."""
        with self._mailbox_lock:
            mailboxes = list(self._mailboxes.values())
        for mailbox in mailboxes:
            mailbox.put(None)
        self._ready_event.set()

    def _describe_exit(self) -> str:
        code = self._process.poll() if self._process is not None else None
        if code is None:
            return "Blender output stream closed while the process was still running"
        return f"Blender exited with code {code}"

    def _write_log(self, line: str) -> None:
        if self._log_file is None:
            return
        try:
            self._log_file.write(line if line.endswith("\n") else line + "\n")
            self._log_file.flush()
        except (OSError, ValueError):
            self._log_file = None

    # -- teardown ----------------------------------------------------------

    def _force_stop(self) -> None:
        if self._process is None or not self.is_alive:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            try:
                self._process.kill()
            except OSError:
                LOGGER.warning("Could not kill the Blender process")

    def _cleanup(self) -> None:
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=TERMINATE_GRACE_SECONDS)
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
        self._log_file = None
        self._process = None
        self._reader = None
        with self._mailbox_lock:
            self._mailboxes.clear()

    # -- typed command helpers --------------------------------------------

    def open_job(self, job_manifest_path: Path, timeout_seconds: float = 600.0) -> dict:
        return self.request(
            open_job_command(self._next_request_id(), str(job_manifest_path)), timeout_seconds,
        )

    def prepare_gap(self, gap_index: int, storyboard_path: Path) -> dict:
        return self.request(
            prepare_gap_command(self._next_request_id(), gap_index, str(storyboard_path))
        )

    def render_frames(
        self,
        gap_index: int,
        frame_indexes: list[int],
        output_directory: Path,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        on_progress: ProgressCallback | None = None,
        regions: list[list[float]] | None = None,
    ) -> dict:
        return self.request(
            render_frames_command(
                self._next_request_id(), gap_index, frame_indexes, str(output_directory),
                regions=regions,
            ),
            timeout_seconds,
            on_progress,
        )

    def reset_gap(self) -> dict:
        return self.request(reset_gap_command(self._next_request_id()))

    def close_job(self) -> dict:
        return self.request(close_job_command(self._next_request_id()))


def frame_output_path(output_directory: Path, frame_index: int) -> Path:
    return output_directory / frame_filename(frame_index)


def is_complete_frame(path: Path) -> bool:
    """A frame counts as done only if it is a readable PNG.

    Blender renders to a temporary name and renames on success, so a truncated file
    should never appear — this is the second line of defence for the case where the
    process died between write and rename.
    """
    try:
        if path.stat().st_size <= MINIMUM_VALID_PNG_BYTES:
            return False
        with path.open("rb") as image_file:
            return image_file.read(8) == PNG_MAGIC
    except OSError:
        return False


def missing_frame_indexes(output_directory: Path, frame_indexes: list[int]) -> list[int]:
    """Frames still to render after an interruption (§6.7).

    The seed of the §9 L6 cache: the render loop asks what is missing rather than
    assuming a gap is either wholly done or wholly undone, which is how v2 lost an
    entire gap's work to one interruption.
    """
    return [
        frame_index for frame_index in frame_indexes
        if not is_complete_frame(frame_output_path(output_directory, frame_index))
    ]


def spawn_blender_process(
    blender_executable: Path,
    service_script: Path,
    job_root: Path,
    project_root: Path,
    environment_overlay: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Default process factory: a real headless Blender speaking the protocol.

    `environment_overlay` carries the §6.5 kernel-cache redirect. It is merged over the
    inherited environment rather than replacing it, because Blender needs the ambient
    graphics and library paths to start at all.
    """
    environment = None
    if environment_overlay:
        environment = {**os.environ, **environment_overlay}
    return subprocess.Popen(
        build_service_command(blender_executable, service_script, job_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(project_root),
        env=environment,
    )
