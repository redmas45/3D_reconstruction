import json
import math
import os
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2

from domain.cancellation import CancellationCheck, CancellationRequestedError, raise_if_cancelled


FFMPEG_PACKAGE_PATTERN = "Gyan.FFmpeg*"
MEDIA_COMMAND_TIMEOUT_SECONDS = 3_600
FRAME_RATE_TOLERANCE = 0.001
DECLARED_FRAME_RATE_RELATIVE_TOLERANCE = 0.001
DECODED_FRAME_RATE_RELATIVE_TOLERANCE = 0.001
MEDIA_PROCESS_POLL_SECONDS = 0.2
MEDIA_TERMINATION_TIMEOUT_SECONDS = 5.0


class MediaToolUnavailableError(RuntimeError):
    pass


class MediaProcessingError(RuntimeError):
    pass


class UnsupportedVideoTimingError(MediaProcessingError):
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
    video_contract = inspect_video_contract(video_only_path)
    if video_contract.fps <= 0.0 or video_contract.frame_count < 1:
        raise MediaProcessingError("Rendered video has an invalid duration")
    duration_seconds = video_contract.frame_count / video_contract.fps
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
        "-t", f"{duration_seconds:.9f}",
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
        raw_contract = (
            float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            float(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if any(not math.isfinite(value) for value in raw_contract):
            raise MediaProcessingError("Rendered video has invalid metadata")
        return VideoContract(
            width=int(raw_contract[0]),
            height=int(raw_contract[1]),
            fps=raw_contract[2],
            frame_count=int(raw_contract[3]),
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


def probe_media(
    video_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> dict:
    ffprobe_path = find_media_tool("ffprobe")
    command = [
        str(ffprobe_path), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video_path),
    ]
    return_code, stdout = _run_probe_command(command, cancellation_check)
    if return_code != 0:
        raise MediaProcessingError(f"ffprobe failed for {video_path.name}")
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise MediaProcessingError("ffprobe returned invalid media metadata") from error
    if not isinstance(report, dict):
        raise MediaProcessingError("ffprobe returned an invalid media report")
    return report


def _run_probe_command(
    command: list[str], cancellation_check: CancellationCheck | None,
) -> tuple[int, str]:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output_file:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_file:
            process = subprocess.Popen(
                command,
                stdout=output_file,
                stderr=error_file,
                text=True,
                start_new_session=os.name != "nt",
            )
            try:
                return_code = _wait_for_media_process(
                    process, time.monotonic() + MEDIA_COMMAND_TIMEOUT_SECONDS, cancellation_check,
                )
            finally:
                if process.poll() is None:
                    _terminate_media_process(process)
            output_file.seek(0)
            return return_code, output_file.read()


def validate_constant_frame_rate(
    video_path: Path,
    decoded_fps: float,
    cancellation_check: CancellationCheck | None = None,
) -> None:
    media_report = probe_media(video_path, cancellation_check)
    video_stream = next(
        (stream for stream in media_report.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise UnsupportedVideoTimingError("The input does not contain a video stream")
    average_rate = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    nominal_rate = _parse_frame_rate(video_stream.get("r_frame_rate"))
    if average_rate <= 0.0 or nominal_rate <= 0.0:
        raise UnsupportedVideoTimingError("The input video has an unreadable frame-rate contract")
    relative_difference = abs(average_rate - nominal_rate) / max(average_rate, nominal_rate)
    if relative_difference > DECLARED_FRAME_RATE_RELATIVE_TOLERANCE:
        raise UnsupportedVideoTimingError(
            "Variable-frame-rate video is not supported; convert the input to constant frame rate first"
        )
    decoded_difference = abs(decoded_fps - average_rate) / max(decoded_fps, average_rate)
    if decoded_difference > DECODED_FRAME_RATE_RELATIVE_TOLERANCE:
        raise UnsupportedVideoTimingError("OpenCV and FFmpeg reported inconsistent source frame rates")


def _parse_frame_rate(value: object) -> float:
    try:
        return float(Fraction(str(value)))
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0


def _run_media_command(
    command: list[str],
    log_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> None:
    deadline = time.monotonic() + MEDIA_COMMAND_TIMEOUT_SECONDS
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
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
    _signal_media_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=MEDIA_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_media_process(process, signal.SIGKILL)
        process.wait()


def _signal_media_process(process: subprocess.Popen[str], signal_number: signal.Signals) -> None:
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


def _winget_tool_candidates(tool_name: str) -> list[Path]:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return []
    package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidates: list[Path] = []
    for package_directory in package_root.glob(FFMPEG_PACKAGE_PATTERN):
        candidates.extend(package_directory.glob(f"ffmpeg-*/bin/{tool_name}.exe"))
    return candidates
