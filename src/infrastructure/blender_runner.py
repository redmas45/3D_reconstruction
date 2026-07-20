import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


BLENDER_EXECUTABLE_ENVIRONMENT_KEY = "BLENDER_EXECUTABLE"
WINDOWS_BLENDER_EXECUTABLE = Path(
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
)
DEFAULT_RENDER_TIMEOUT_SECONDS = 1_800


class BlenderUnavailableError(RuntimeError):
    pass


class BlenderRenderError(RuntimeError):
    pass


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
    timeout_seconds: int = DEFAULT_RENDER_TIMEOUT_SECONDS,
) -> BlenderRenderResult:
    _validate_request(request)
    blender_executable = find_blender_executable()
    render_script = (project_root / "blender" / "render_gap.py").resolve()
    if not render_script.is_file():
        raise BlenderRenderError(f"Blender render script is missing: {render_script}")
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_blender_command(blender_executable, render_script, request)
    completed_process = _run_blender(command, request.log_path, project_root, timeout_seconds)
    if completed_process.returncode != 0:
        raise BlenderRenderError(
            f"Blender render failed with exit code {completed_process.returncode}. See {request.log_path}"
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
    command: list[str], log_path: Path, project_root: Path, timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        completed_process = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        log_path.write_text(_timeout_log(error), encoding="utf-8")
        raise BlenderRenderError(f"Blender exceeded the {timeout_seconds}-second render timeout") from error
    log_path.write_text(completed_process.stdout + completed_process.stderr, encoding="utf-8")
    return completed_process


def _validate_request(request: BlenderRenderRequest) -> None:
    if request.mode not in {"preview", "animation"}:
        raise ValueError("Blender render mode must be preview or animation")
    if not request.plan_path.is_file():
        raise FileNotFoundError(f"Reconstruction plan is missing: {request.plan_path}")
    if request.plan_path.resolve() == request.output_path.resolve():
        raise ValueError("Plan and render output paths must be different")


def _validate_outputs(request: BlenderRenderRequest) -> None:
    missing_paths = [
        output_path for output_path in (request.output_path, request.report_path, request.blend_path)
        if not output_path.is_file()
    ]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise BlenderRenderError(f"Blender finished without required outputs: {missing_text}")


def _timeout_log(error: subprocess.TimeoutExpired) -> str:
    standard_output = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
    standard_error = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
    return standard_output + standard_error
