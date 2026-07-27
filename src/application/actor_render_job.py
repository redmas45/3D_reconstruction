"""Renders every gap of one video through a single warm Blender process (§6).

This is the piece that replaces `blender_pipeline.render_blender_gap`. The old path
launched `blender --background` once per gap and rebuilt a synthetic street each time,
which §6.1 measures at roughly 7-8 minutes of pure startup waste per job. Here one
process serves the whole video: the shell is built once, and each gap only instances
actors against it.

The loop per gap is deliberately boring:

    prepare_gap -> render the sparse frames that are missing -> composite -> encode

Everything expensive is either cached or skipped. Frames already on disk are not
re-rendered, so an interrupted job resumes at frame granularity rather than losing a
gap. The frame directory is keyed by the gap specification's digest, so a changed plan
gets a fresh directory instead of silently reusing layers rendered for a different path.

Two things this module owns that the pieces below it deliberately do not:

  * **Process lifecycle.** Gaps render sequentially because there is one GPU. The
    process is recycled every `GAP_RECYCLE_INTERVAL` gaps (§6.6) and after any crash,
    and a recycle re-opens the job so the next gap still finds a shell.
  * **Camera grouping.** The warm shell is built for one camera. If calibration differs
    between gaps the job is re-opened for the new camera rather than rendering actors
    through the wrong lens.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy

from application.exemplar_gap_renderer import coverage, render_exemplar_layers
from application.actor_gap_renderer import (
    ActorRenderError,
    GAP_SPECIFICATION_FILENAME,
    SparseFrame,
    build_gap_specification,
    build_job_manifest,
    composite_sparse_frames,
    expand_to_source_frames,
    plan_sparse_frames,
    write_gap_video,
)
from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.camera_projection import supports_projection
from infrastructure.blender_service import (
    BlenderService,
    BlenderServiceError,
    missing_frame_indexes,
    spawn_blender_process,
)
from infrastructure.json_files import write_json_file


LOGGER = logging.getLogger(__name__)

JOB_MANIFEST_FILENAME = "job_manifest.json"
GAP_VIDEO_FILENAME = "gap_actors.mp4"
GAP_REPORT_FILENAME = "actor_render_report.json"
LAYER_DIRECTORY_NAME = "layers"
# Fraction of a gap's entities that must have usable observed footage before the whole
# gap is drawn from photographs. Below it the gap would be a mixture of photographic and
# modelled figures, and the modelled ones would be the only thing anyone looked at.
MINIMUM_EXEMPLAR_COVERAGE = 0.6

# Recycle the process every few gaps. Blender's memory grows slowly across many scene
# edits, and a planned restart between gaps costs seconds where an out-of-memory kill
# mid-render costs the gap.
GAP_RECYCLE_INTERVAL = 6
# Frames are requested in batches so progress and checkpointing arrive during a gap
# rather than only at its end.
RENDER_BATCH_FRAMES = 12
# A crashed Blender is retried once with a fresh process. Completed frames survive on
# disk, so the retry costs only what was actually lost. More attempts than this usually
# means a real fault rather than a transient one, and looping would hide it.
RENDER_ATTEMPTS = 2

FrameProgressCallback = Callable[[int, int], None]


class ActorJobError(RuntimeError):
    """The actor render job could not be completed."""


@dataclass(frozen=True)
class GapRenderOutcome:
    gap_index: int
    video_path: Path
    sparse_frame_count: int
    rendered_frame_count: int
    reused_frame_count: int
    source_frame_count: int


def actor_path_is_supported(plan: dict) -> tuple[bool, str]:
    """Whether this plan can be rendered as composited actors.

    Returned as a reason rather than a bare bool so the caller can say in its log why it
    fell back, instead of the operator seeing a slow render with no explanation.
    """
    camera = plan.get("camera")
    if not isinstance(camera, dict) or not supports_projection(camera):
        return False, "the camera contract has no forward projection"
    if not plan.get("entities"):
        return False, "the plan selected no entities to render"
    return True, "supported"


def gap_specification_digest(specification: dict, manifest: dict) -> str:
    """Identifies the layers a gap's frames belong to.

    Covers the manifest as well as the gap: the same actor path rendered through a
    different camera or engine produces different pixels, and reusing the old ones would
    put the actor in the wrong place.
    """
    payload = json.dumps(
        {"gap": specification, "job": manifest}, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _batched(values: list, size: int) -> list[list]:
    return [values[start:start + size] for start in range(0, len(values), size)]


class ActorRenderJob:
    """One warm Blender process rendering every gap of a video."""

    def __init__(
        self,
        blender_executable: Path,
        project_root: Path,
        job_root: Path,
        plate_for_gap: Callable[[int], numpy.ndarray],
        frame_width: int,
        frame_height: int,
        source_fps: float,
        stall_timeout_seconds: float,
        environment_overlay: dict[str, str] | None = None,
        library_directory: Path | None = None,
        service_factory: Callable[[], BlenderService] | None = None,
        cancellation_check: CancellationCheck | None = None,
        exemplar_banks_for_gap: Callable[[int], dict] | None = None,
    ) -> None:
        self._blender_executable = blender_executable
        self._project_root = project_root
        self._job_root = job_root
        # A background belongs to a scene, not to a video: a gap in the third take of a
        # montage must composite onto that take's background, not onto an average of all
        # of them. The job asks for the one it needs rather than holding a single plate,
        # which keeps it ignorant of how the timeline is divided up.
        self._plate_for_gap = plate_for_gap
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._source_fps = source_fps
        self._stall_timeout_seconds = stall_timeout_seconds
        self._environment_overlay = environment_overlay
        self._library_directory = library_directory
        self._service_factory = service_factory or self._default_service_factory
        self._cancellation_check = cancellation_check
        self._exemplar_banks_for_gap = exemplar_banks_for_gap
        self._service: BlenderService | None = None
        self._open_manifest_digest: str | None = None
        self._gaps_since_recycle = 0

    # -- lifecycle ---------------------------------------------------------

    def _default_service_factory(self) -> BlenderService:
        service_script = self._project_root / "blender" / "service.py"
        if not service_script.is_file():
            raise ActorJobError(f"Blender service script is missing: {service_script}")
        return BlenderService(
            process_factory=lambda: spawn_blender_process(
                self._blender_executable,
                service_script,
                self._job_root,
                self._project_root,
                self._environment_overlay,
            ),
            log_path=self._job_root / "blender_service.log",
            stall_timeout_seconds=self._stall_timeout_seconds,
        )

    def __enter__(self) -> "ActorRenderJob":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._service is None:
            return
        try:
            if self._open_manifest_digest is not None and self._service.is_alive:
                self._service.close_job()
        except BlenderServiceError as error:
            LOGGER.warning("Ignoring failure while closing the Blender job: %s", error)
        finally:
            self._service.shutdown()
            self._service = None
            self._open_manifest_digest = None

    def _ensure_service(self, manifest: dict) -> BlenderService:
        """Start or recycle the process, and open the job for this manifest.

        `open_job` is re-sent whenever the manifest changes or the process was replaced,
        because a fresh process has no shell and a different manifest means a different
        shell.
        """
        digest = gap_specification_digest({}, manifest)
        if self._service is not None and not self._service.is_alive:
            LOGGER.warning("Blender service is no longer alive; starting a replacement")
            self._service.shutdown()
            self._service = None
            self._open_manifest_digest = None
        if self._service is None:
            self._job_root.mkdir(parents=True, exist_ok=True)
            self._service = self._service_factory()
            capability = self._service.start()
            LOGGER.info(
                "Blender service ready: %s, engines %s",
                capability.get("blender_version"), capability.get("engines"),
            )
            self._open_manifest_digest = None
        if self._open_manifest_digest != digest:
            manifest_path = self._job_root / JOB_MANIFEST_FILENAME
            write_json_file(manifest_path, manifest)
            summary = self._service.open_job(manifest_path)
            LOGGER.info(
                "Opened Blender job: engine %s, shadow catcher %s",
                summary.get("engine"), summary.get("shadow_catcher"),
            )
            self._open_manifest_digest = digest
            self._gaps_since_recycle = 0
        return self._service

    def _recycle_if_due(self) -> None:
        if self._gaps_since_recycle < GAP_RECYCLE_INTERVAL:
            return
        LOGGER.info("Recycling the Blender process after %d gaps", self._gaps_since_recycle)
        self.close()
        self._gaps_since_recycle = 0

    # -- rendering ---------------------------------------------------------

    def render_gap(
        self,
        plan: dict,
        gap_directory: Path,
        reuse_work: bool = True,
        progress_callback: FrameProgressCallback | None = None,
    ) -> GapRenderOutcome:
        """Render, composite and encode one gap. Returns where the video landed."""
        raise_if_cancelled(self._cancellation_check)
        supported, reason = actor_path_is_supported(plan)
        if not supported:
            raise ActorJobError(f"Gap {plan.get('gap_index')} cannot use the actor path: {reason}")
        manifest = build_job_manifest(
            plan, self._frame_width, self._frame_height, self._library_directory,
        )
        sparse_frames = plan_sparse_frames(plan, self._frame_width, self._frame_height)
        specification = build_gap_specification(plan, sparse_frames)
        digest = gap_specification_digest(specification, manifest)
        banks = self._banks_for(plan)
        if banks is not None:
            # Real footage of the real subjects beats anything that can be modelled, so
            # it is preferred whenever enough of it exists. Kept in its own directory:
            # the two producers make different pictures from the same specification, and
            # a cache hit across them would silently mix synthetic and photographic
            # actors in one video.
            layer_directory = gap_directory / LAYER_DIRECTORY_NAME / f"{digest}_exemplar"
            report = render_exemplar_layers(
                plan, sparse_frames, banks, self._frame_width, self._frame_height,
                layer_directory, self._cancellation_check,
            )
            LOGGER.info(
                "Gap %s drawn from observed footage: %d of %d entities, %d draws",
                plan.get("gap_index"), report["entities_with_footage"],
                report["entities_planned"], report["actor_draws"],
            )
            self._report_progress(
                progress_callback, len(sparse_frames), len(sparse_frames),
            )
            return self._assemble_gap(
                plan, sparse_frames, layer_directory, gap_directory,
                len(sparse_frames), 0, report,
            )
        layer_directory = gap_directory / LAYER_DIRECTORY_NAME / digest
        layer_directory.mkdir(parents=True, exist_ok=True)
        rendered_count, reused_count = self._render_layers(
            plan, manifest, specification, sparse_frames, layer_directory,
            reuse_work, progress_callback,
        )
        return self._assemble_gap(
            plan, sparse_frames, layer_directory, gap_directory,
            rendered_count, reused_count,
        )

    def _banks_for(self, plan: dict) -> dict | None:
        """Observed cut-outs for this gap, or None to fall back to rendering geometry."""
        if self._exemplar_banks_for_gap is None:
            return None
        banks = self._exemplar_banks_for_gap(int(plan["gap_index"]))
        if not banks:
            return None
        observed = coverage(plan, banks)
        if observed < MINIMUM_EXEMPLAR_COVERAGE:
            LOGGER.info(
                "Gap %s has footage for only %.0f%% of its entities; rendering geometry "
                "instead", plan.get("gap_index"), observed * 100.0,
            )
            return None
        return banks

    def _render_layers(
        self,
        plan: dict,
        manifest: dict,
        specification: dict,
        sparse_frames: list[SparseFrame],
        layer_directory: Path,
        reuse_work: bool,
        progress_callback: FrameProgressCallback | None,
    ) -> tuple[int, int]:
        indexes = [frame.render_index for frame in sparse_frames]
        outstanding = (
            missing_frame_indexes(layer_directory, indexes) if reuse_work else list(indexes)
        )
        reused_count = len(indexes) - len(outstanding)
        if not outstanding:
            LOGGER.info("Gap %s reused every rendered layer", plan.get("gap_index"))
            self._report_progress(progress_callback, len(indexes), len(indexes))
            return 0, reused_count
        self._recycle_if_due()
        specification_path = layer_directory / GAP_SPECIFICATION_FILENAME
        write_json_file(specification_path, specification)
        regions_by_index = {frame.render_index: frame.region for frame in sparse_frames}
        remaining = outstanding
        for attempt in range(RENDER_ATTEMPTS):
            try:
                self._render_batches(
                    plan, manifest, specification_path, remaining, regions_by_index,
                    layer_directory, len(indexes), reused_count, progress_callback,
                )
                break
            except BlenderServiceError as error:
                # Completed frames stay on disk, so a restart resumes rather than
                # repeating the gap. Only frames still missing are re-requested.
                remaining = missing_frame_indexes(layer_directory, indexes)
                if attempt == RENDER_ATTEMPTS - 1 or not remaining:
                    if remaining:
                        raise ActorJobError(
                            f"Blender failed on gap {plan.get('gap_index')} with "
                            f"{len(remaining)} frames outstanding: {error}"
                        ) from error
                    break
                LOGGER.warning(
                    "Blender failed on gap %s (%s); restarting with %d frames left",
                    plan.get("gap_index"), error, len(remaining),
                )
                self.close()
        self._gaps_since_recycle += 1
        return len(outstanding), reused_count

    def _render_batches(
        self,
        plan: dict,
        manifest: dict,
        specification_path: Path,
        outstanding: list[int],
        regions_by_index: dict,
        layer_directory: Path,
        total_frames: int,
        reused_count: int,
        progress_callback: FrameProgressCallback | None,
    ) -> None:
        service = self._ensure_service(manifest)
        gap_index = int(plan["gap_index"])
        service.prepare_gap(gap_index, specification_path)
        completed = total_frames - len(outstanding)
        for batch in _batched(outstanding, RENDER_BATCH_FRAMES):
            raise_if_cancelled(self._cancellation_check)
            service.render_frames(
                gap_index,
                batch,
                layer_directory,
                timeout_seconds=self._stall_timeout_seconds,
                regions=[
                    [
                        regions_by_index[index].minimum_x,
                        regions_by_index[index].minimum_y,
                        regions_by_index[index].maximum_x,
                        regions_by_index[index].maximum_y,
                    ]
                    for index in batch
                ],
            )
            completed += len(batch)
            self._report_progress(progress_callback, completed, total_frames)
        service.reset_gap()

    def _assemble_gap(
        self,
        plan: dict,
        sparse_frames: list[SparseFrame],
        layer_directory: Path,
        gap_directory: Path,
        rendered_count: int,
        reused_count: int,
        actor_source: dict | None = None,
    ) -> GapRenderOutcome:
        raise_if_cancelled(self._cancellation_check)
        plate = self._plate_for_gap(int(plan["gap_index"]))
        if plate.shape[:2] != (self._frame_height, self._frame_width):
            raise ActorJobError(
                f"Gap {plan['gap_index']} was given a {plate.shape[1]}x{plate.shape[0]} "
                f"background for a {self._frame_width}x{self._frame_height} frame"
            )
        composed = composite_sparse_frames(
            plate, sparse_frames, layer_directory, self._cancellation_check,
        )
        source_frame_count = int(plan["frame_count"])
        expanded = expand_to_source_frames(composed, source_frame_count)
        video_path = write_gap_video(
            expanded, self._source_fps, gap_directory / GAP_VIDEO_FILENAME,
        )
        outcome = GapRenderOutcome(
            gap_index=int(plan["gap_index"]),
            video_path=video_path,
            sparse_frame_count=len(sparse_frames),
            rendered_frame_count=rendered_count,
            reused_frame_count=reused_count,
            source_frame_count=source_frame_count,
        )
        write_json_file(gap_directory / GAP_REPORT_FILENAME, {
            "schema_version": 1,
            "mode": "actor_composite",
            "gap_index": outcome.gap_index,
            "sparse_frames": outcome.sparse_frame_count,
            "rendered_frames": outcome.rendered_frame_count,
            "reused_frames": outcome.reused_frame_count,
            "source_frames": outcome.source_frame_count,
            "resolution": [self._frame_width, self._frame_height],
            "fps": self._source_fps,
            # Where this gap's actors came from, and how many of its planned entities
            # could actually be drawn. Persisted rather than only logged: "the street
            # looks emptier than the truth" has two very different causes — entities the
            # planner never selected, and entities it selected that had no usable
            # footage — and only this distinguishes them after the fact.
            "actor_source": actor_source or {"mode": "rendered_geometry"},
            "regions": [frame.region.as_dict() for frame in sparse_frames],
        })
        return outcome

    @staticmethod
    def _report_progress(
        callback: FrameProgressCallback | None, completed: int, total: int,
    ) -> None:
        if callback is None:
            return
        try:
            callback(completed, total)
        except Exception:
            LOGGER.exception("Ignoring a failing render progress callback")


def render_actor_gaps(
    plans: list[dict],
    gap_directories: list[Path],
    job: ActorRenderJob,
    reuse_work: bool = True,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict[int, Path]:
    """Render every gap in order through one job. Returns gap index to video path.

    Sequential by construction: there is one GPU, and the whole point of the warm
    process is that it is shared rather than duplicated.
    """
    if len(plans) != len(gap_directories):
        raise ActorJobError("Every plan requires exactly one gap directory")
    rendered: dict[int, Path] = {}
    with job:
        for position, (plan, gap_directory) in enumerate(zip(plans, gap_directories)):
            outcome = job.render_gap(
                plan,
                gap_directory,
                reuse_work,
                lambda completed, total, index=position: (
                    progress_callback(index, completed, total)
                    if progress_callback is not None else None
                ),
            )
            rendered[outcome.gap_index] = outcome.video_path
    return rendered


__all__ = [
    "ActorJobError",
    "ActorRenderError",
    "ActorRenderJob",
    "GapRenderOutcome",
    "actor_path_is_supported",
    "gap_specification_digest",
    "render_actor_gaps",
]
