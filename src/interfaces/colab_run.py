"""Colab orchestration, owned by the repository rather than the notebook.

Implementation_plan.md §28 requires the notebook to import shared code rather than
carry its own copy of the pipeline. Everything here previously lived inside a single
notebook cell, where it could not be tested, reviewed, or reused — and drifted from
the code it drove.

The notebook keeps only what genuinely needs `google.colab`: mounting Drive, the two
file-upload widgets, and displaying the preview. Everything else is here.

No `google.colab` import in this module, so the whole thing is testable off Colab.
"""

import copy
import hashlib
import json
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from application.reconstruction_pipeline import PipelineOptions, load_config, process_video
from domain.configuration import validate_configuration
from domain.render_runtime_budget import (
    RepresentativePreviewApprovalRequired,
    approve_representative_preview,
)


AZURE_SECRET_NAMES = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
)
MAXIMUM_ENV_BYTES = 64 * 1024
SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg", ".wmv",
})

CHECKPOINT_INTERVAL_SECONDS = 30
# A gap counts as complete when either renderer's full artifact set is present. The
# actor path writes its video and report at the top of the gap directory; the older
# full-scene path writes into a `blender/` subdirectory.
COMPLETED_GAP_ARTIFACT_SETS = (
    ("gap_actors.mp4", "actor_render_report.json"),
    ("blender/gap_blender.mp4", "blender/render_report.json", "blender/scene.blend"),
)
# Rendered frames worth mirroring mid-gap, so a disconnect loses minutes not a gap.
SPARSE_FRAME_GLOBS = ("gap_*/blender/renders/frames_*", "gap_*/layers/*")

ProgressCallback = Callable[[str, float, str], None]
ApprovalPrompt = Callable[[Path], bool]


class EnvironmentFileError(ValueError):
    """The uploaded .env file could not be parsed."""


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def parse_environment_file(payload: bytes) -> dict[str, str]:
    """Read Azure values from an uploaded .env without writing it to disk.

    Only the three known names are extracted; anything else in the operator's local
    .env is ignored rather than loaded into the Colab process.
    """
    if len(payload) > MAXIMUM_ENV_BYTES:
        raise EnvironmentFileError("The uploaded .env file is unexpectedly large.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EnvironmentFileError("The .env file must be UTF-8 text.") from error
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        entry = _parse_environment_line(raw_line, line_number)
        if entry is not None and entry[0] in AZURE_SECRET_NAMES:
            values[entry[0]] = entry[1]
    return values


def _parse_environment_line(raw_line: str, line_number: int) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    name, separator, value = line.partition("=")
    if not separator:
        raise EnvironmentFileError(f"Invalid .env assignment on line {line_number}.")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return name.strip(), value


def missing_azure_names(values: dict[str, str]) -> list[str]:
    return [name for name in AZURE_SECRET_NAMES if not values.get(name, "").strip()]


def validate_video_selection(video_path: Path) -> Path:
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video was not found: {video_path}")
    return video_path


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ColabRenderSettings:
    """Colab-specific overrides applied over the checked-in configuration.

    The defaults describe the actor-composite path: only detected entities are drawn,
    cropped to the box they occupy, onto a plate recovered from the real footage. The
    runtime-budget gate that the full-scene path needed is off here, because that gate
    exists to stop a multi-hour render surprising the operator and this path does not
    produce one. Set `render_mode="full_scene"` to get the old renderer and its gate.
    """

    render_mode: str = "actor_composite"
    engine: str = "BLENDER_EEVEE_NEXT"
    cycles_compute_device: str = "OPTIX"
    cycles_samples: int = 8
    cycles_use_denoising: bool = True
    production_scale_percent: int = 40
    reconstruction_fps: int = 12
    max_render_entities: int = 3
    parallel_gap_renders: int = 1
    stall_timeout_seconds: int = 900
    maximum_predicted_render_seconds: int = 7200
    allow_runtime_budget_override: bool = True
    interactive_preview_approval: bool = False

    @property
    def uses_actor_path(self) -> bool:
        return self.render_mode == "actor_composite"


def build_runtime_configuration(
    configuration_path: Path, settings: ColabRenderSettings,
) -> dict:
    """Apply Colab overrides and validate before anything expensive starts."""
    configuration = copy.deepcopy(load_config(configuration_path))
    renderer = configuration["renderer"]
    renderer["render_mode"] = settings.render_mode
    renderer["engine"] = settings.engine
    renderer["cycles_compute_device"] = settings.cycles_compute_device
    renderer["cycles_samples"] = settings.cycles_samples
    renderer["cycles_use_denoising"] = settings.cycles_use_denoising
    renderer["production_scale_percent"] = settings.production_scale_percent
    renderer["scale_percent"] = settings.production_scale_percent
    renderer["target_fps"] = settings.reconstruction_fps
    renderer["max_parallel_gap_renders"] = settings.parallel_gap_renders
    renderer["gap_render_stall_timeout_seconds"] = settings.stall_timeout_seconds
    renderer["maximum_predicted_render_seconds"] = settings.maximum_predicted_render_seconds
    renderer["runtime_budget_enabled"] = not settings.uses_actor_path
    renderer["allow_runtime_budget_override"] = settings.allow_runtime_budget_override
    renderer["interactive_preview_approval"] = settings.interactive_preview_approval
    configuration["scene"]["max_render_entities"] = settings.max_render_entities
    validate_configuration(configuration)
    return configuration


def build_run_identifier(
    video_path: Path,
    configuration: dict,
    seed: int,
    blender_version: str,
    project_root: Path,
) -> str:
    """Content-addressed run name, so a resumed session finds its own artifacts.

    Worker counts and timeouts are excluded: they change with the Colab machine and
    must not invalidate an otherwise identical run's checkpoints.
    """
    with video_path.open("rb") as stream:
        video_digest = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    checkpoint_configuration = copy.deepcopy(configuration)
    for volatile_key in ("max_parallel_gap_renders", "gap_render_stall_timeout_seconds"):
        checkpoint_configuration["renderer"].pop(volatile_key, None)
    configuration_digest = hashlib.sha256(
        json.dumps(checkpoint_configuration, sort_keys=True).encode()
    ).hexdigest()[:10]
    commit = _git_commit(project_root)
    return (
        f"{video_path.stem}_{video_digest}_seed{seed}_"
        f"cfg{configuration_digest}_git{commit}_blender{blender_version}"
    )


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], cwd=str(project_root), text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "nogit"


# --------------------------------------------------------------------------
# Drive checkpointing
# --------------------------------------------------------------------------

class DriveCheckpointer:
    """Mirrors completed work to Drive so a lost session resumes instead of restarting.

    Drive is slow per file, so completed gaps are copied through a temporary directory
    and renamed — a half-copied gap must never look complete to the next session.
    """

    def __init__(
        self,
        local_gaps: Path,
        drive_gaps: Path,
        mirrored_directories: list[tuple[Path, Path]] | None = None,
    ) -> None:
        self._local_gaps = local_gaps
        self._drive_gaps = drive_gaps
        # Whole directories mirrored verbatim in both directions — the clean plate, which
        # costs a full video decode to rebuild and is identical for every gap.
        self._mirrored_directories = mirrored_directories or []
        self._lock = threading.Lock()
        self._checkpointed: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def restore(self) -> bool:
        restored = False
        for local_directory, drive_directory in self._mirrored_directories:
            if drive_directory.is_dir():
                shutil.copytree(drive_directory, local_directory, dirs_exist_ok=True)
                restored = True
        if not self._drive_gaps.is_dir():
            return restored
        shutil.copytree(self._drive_gaps, self._local_gaps, dirs_exist_ok=True)
        return True

    def save_mirrored_directories(self) -> int:
        saved = 0
        for local_directory, drive_directory in self._mirrored_directories:
            if not local_directory.is_dir():
                continue
            shutil.copytree(local_directory, drive_directory, dirs_exist_ok=True)
            saved += 1
        return saved

    def save_completed_gaps(self) -> list[str]:
        if not self._local_gaps.is_dir():
            return []
        saved: list[str] = []
        with self._lock:
            for gap_directory in sorted(self._local_gaps.glob("gap_*")):
                if self._save_one_gap(gap_directory):
                    saved.append(gap_directory.name)
        return saved

    def _save_one_gap(self, gap_directory: Path) -> bool:
        if gap_directory.name in self._checkpointed:
            return False
        if not any(
            all((gap_directory / name).is_file() for name in artifact_set)
            for artifact_set in COMPLETED_GAP_ARTIFACT_SETS
        ):
            return False
        destination = self._drive_gaps / gap_directory.name
        staging = self._drive_gaps / f"{gap_directory.name}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(gap_directory, staging)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        self._checkpointed.add(gap_directory.name)
        return True

    def save_sparse_frames(self) -> int:
        if not self._local_gaps.is_dir():
            return 0
        copied = 0
        with self._lock:
            for pattern in SPARSE_FRAME_GLOBS:
                for frame_directory in self._local_gaps.glob(pattern):
                    if frame_directory.is_dir():
                        copied += self._mirror_frames(frame_directory)
        return copied

    def _mirror_frames(self, frame_directory: Path) -> int:
        destination = self._drive_gaps / frame_directory.relative_to(self._local_gaps)
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for frame_path in frame_directory.glob("frame_*.png"):
            target = destination / frame_path.name
            if target.is_file() and target.stat().st_size == frame_path.stat().st_size:
                continue
            shutil.copy2(frame_path, target)
            copied += 1
        manifest = frame_directory / "frame_manifest.json"
        if manifest.is_file():
            # Copied last so the manifest never references frames that are not there yet.
            shutil.copy2(manifest, destination / manifest.name)
        return copied

    def start_background_mirror(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._mirror_loop, name="drive-checkpoint", daemon=True,
        )
        self._thread.start()

    def _mirror_loop(self) -> None:
        while not self._stop_event.wait(CHECKPOINT_INTERVAL_SECONDS):
            self.save_mirrored_directories()
            self.save_sparse_frames()
            self.save_completed_gaps()

    def stop(self, timeout_seconds: float = 15.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None
        self.save_mirrored_directories()
        self.save_sparse_frames()
        self.save_completed_gaps()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

@dataclass
class ReconstructionRun:
    video_path: Path
    configuration: dict
    run_identifier: str
    local_output: Path
    drive_run: Path
    work_directory: Path
    checkpointer: DriveCheckpointer = field(init=False)

    def __post_init__(self) -> None:
        checkpoints = self.drive_run / "checkpoints"
        self.checkpointer = DriveCheckpointer(
            self.work_directory / "gaps",
            checkpoints / "gaps",
            mirrored_directories=[
                (self.work_directory / "plate", checkpoints / "plate"),
            ],
        )


def prepare_run(
    video_path: Path,
    configuration: dict,
    run_identifier: str,
    content_root: Path,
    drive_root: Path,
) -> ReconstructionRun:
    with video_path.open("rb") as stream:
        video_digest = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
    local_output = content_root / run_identifier
    work_directory = local_output / "_work" / f"{video_path.stem}_{video_digest}"
    return ReconstructionRun(
        video_path=video_path,
        configuration=configuration,
        run_identifier=run_identifier,
        local_output=local_output,
        drive_run=drive_root / run_identifier,
        work_directory=work_directory,
    )


def execute_run(
    run: ReconstructionRun,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    approval_prompt: ApprovalPrompt | None = None,
) -> Path:
    """Run the pipeline, mirroring to Drive and pausing for preview approval.

    `approval_prompt` receives the representative preview and returns whether to
    continue. Returning False stops the run rather than silently spending the full
    render budget (§6 runtime gate).
    """
    import random

    if run.checkpointer.restore():
        _emit(progress_callback, "queued", 0.0, "Restored completed gaps from Drive")
    options = PipelineOptions(run.configuration, run.local_output, reuse_work=True)
    run.checkpointer.start_background_mirror()
    try:
        return _process_with_approval(
            run, options, random.Random(seed), progress_callback, approval_prompt,
        )
    finally:
        run.checkpointer.stop()


def _process_with_approval(
    run: ReconstructionRun,
    options: PipelineOptions,
    rng: object,
    progress_callback: ProgressCallback | None,
    approval_prompt: ApprovalPrompt | None,
) -> Path:
    while True:
        try:
            return process_video(run.video_path, options, rng, progress_callback)
        except RepresentativePreviewApprovalRequired as request:
            run.checkpointer.save_sparse_frames()
            run.checkpointer.save_completed_gaps()
            if approval_prompt is None or not approval_prompt(request.preview_path):
                raise RuntimeError(
                    "Full render stopped because the representative preview was not approved."
                ) from None
            approve_representative_preview(request.approval_path, request.signature)


def _emit(callback: ProgressCallback | None, stage: str, progress: float, detail: str) -> None:
    if callback is not None:
        callback(stage, progress, detail)
