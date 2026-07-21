import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from domain.cancellation import CancellationCheck, CancellationRequestedError, raise_if_cancelled


FFMPEG_PACKAGE_PATTERN = "Gyan.FFmpeg*"
MEDIA_COMMAND_TIMEOUT_SECONDS = 3_600
FRAME_RATE_TOLERANCE = 0.001
MEDIA_PROCESS_POLL_SECONDS = 0.2
MEDIA_TERMINATION_TIMEOUT_SECONDS = 5.0


class MediaToolUnavailableError(RuntimeError):
    pass


class MediaProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoContract:
    width: int
    height: int
    fps: float
    frame_count: int


def find_media_tool(tool_name: str) -> Path:
    discovered_path = shutil.which(tool_name)
    if discovered_path:
        return Path(discovered_path).resolve()
    for candidate in _winget_tool_candidates(tool_name):
        if candidate.is_file():
            return candidate.resolve()
    raise MediaToolUnavailableError(f"{tool_name} is required but was not found")


def encode_with_source_audio(
    video_only_path: Path,
    source_path: Path,
    output_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    ffmpeg_path = find_media_tool("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_path), "-y",
        "-i", str(video_only_path),
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run_media_command(command, output_path.parent / "ffmpeg_mux.log", cancellation_check)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MediaProcessingError("FFmpeg completed without producing the final video")
    return output_path


def inspect_video_contract(video_path: Path) -> VideoContract:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MediaProcessingError(f"Cannot inspect rendered video: {video_path}")
    try:
        return VideoContract(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def validate_video_contract(video_path: Path, expected: VideoContract) -> VideoContract:
    actual = inspect_video_contract(video_path)
    mismatches: list[str] = []
    if (actual.width, actual.height) != (expected.width, expected.height):
        mismatches.append(f"resolution {actual.width}x{actual.height}")
    if abs(actual.fps - expected.fps) > FRAME_RATE_TOLERANCE:
        mismatches.append(f"fps {actual.fps:.6f}")
    if actual.frame_count != expected.frame_count:
        mismatches.append(f"frame count {actual.frame_count}")
    if mismatches:
        raise MediaProcessingError("Rendered video contract mismatch: " + ", ".join(mismatches))
    return actual


def probe_media(video_path: Path) -> dict:
    ffprobe_path = find_media_tool("ffprobe")
    command = [
        str(ffprobe_path), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video_path),
    ]
    completed_process = subprocess.run(
        command, capture_output=True, text=True, timeout=MEDIA_COMMAND_TIMEOUT_SECONDS, check=False
    )
    if completed_process.returncode != 0:
        raise MediaProcessingError(f"ffprobe failed for {video_path.name}")
    return json.loads(completed_process.stdout)


def _run_media_command(
    command: list[str],
    log_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> None:
    deadline = time.monotonic() + MEDIA_COMMAND_TIMEOUT_SECONDS
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            return_code = _wait_for_media_process(process, deadline, cancellation_check)
        finally:
            if process.poll() is None:
                _terminate_media_process(process)
    if return_code != 0:
        raise MediaProcessingError(f"FFmpeg failed with exit code {return_code}; see {log_path}")


def _wait_for_media_process(
    process: subprocess.Popen[str],
    deadline: float,
    cancellation_check: CancellationCheck | None,
) -> int:
    while process.poll() is None:
        if cancellation_check is not None and cancellation_check():
            _terminate_media_process(process)
            raise CancellationRequestedError("Media encoding was cancelled")
        if time.monotonic() >= deadline:
            _terminate_media_process(process)
            raise MediaProcessingError("FFmpeg exceeded the media-processing timeout")
        time.sleep(MEDIA_PROCESS_POLL_SECONDS)
    raise_if_cancelled(cancellation_check)
    return int(process.returncode or 0)


def _terminate_media_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=MEDIA_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _winget_tool_candidates(tool_name: str) -> list[Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return []
    package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates: list[Path] = []
    for package_directory in package_root.glob(FFMPEG_PACKAGE_PATTERN):
        candidates.extend(package_directory.glob(f"ffmpeg-*/bin/{tool_name}.exe"))
    return candidates
