import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.evidence_contract import validate_visible_evidence_only
from domain.identity_registry import build_identity_registry, write_identity_registry
from domain.reconstruction_plan_v2 import (
    PlanValidationError,
    build_reconstruction_plan_v2,
    validate_reconstruction_plan_v2,
    write_reconstruction_plan_v2,
)
from infrastructure.blender_runner import (
    DEFAULT_RENDER_STALL_TIMEOUT_SECONDS,
    BlenderProgressCallback,
    BlenderRenderRequest,
    BlenderRenderResult,
    render_with_blender,
)
from infrastructure.camera_motion_estimator import estimate_camera_motion
from infrastructure.media_tools import (
    MediaProcessingError,
    VideoContract,
    encode_png_sequence,
    inspect_video_contract,
)
from infrastructure.video_frames import export_forensic_context_frame


REQUIRED_RENDER_REPORT_FIELDS = frozenset({
    "fps", "frame_count", "mode", "plan_hash", "render_engine", "resolution",
})
RENDER_FPS_TOLERANCE = 0.001


@dataclass(frozen=True)
class PreparedBlenderAssets:
    scene_report: dict
    plan_paths: list[Path]
    identity_registry_path: Path


def prepare_blender_assets(
    video_path: Path,
    scene_report: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
    cancellation_check: CancellationCheck | None = None,
) -> PreparedBlenderAssets:
    raise_if_cancelled(cancellation_check)
    validate_visible_evidence_only(scene_report)
    motion_report = estimate_camera_motion(video_path, scene_report, cancellation_check)
    calibrated_report = {**scene_report, "camera_motion_report": motion_report}
    identity_registry = build_identity_registry(scene_report, video_path, cancellation_check)
    registry_path = work_directory / "entity_registry.json"
    write_identity_registry(identity_registry, registry_path)
    _write_json(work_directory / "camera_motion_report.json", motion_report)
    plan_paths = _write_gap_plans(
        video_path,
        calibrated_report,
        identity_registry,
        hidden_ranges,
        work_directory,
        maximum_entities,
        render_configuration,
        cancellation_check,
    )
    return PreparedBlenderAssets(calibrated_report, plan_paths, registry_path)


def render_blender_gap(
    project_root: Path,
    plan_path: Path,
    gap_directory: Path,
    reuse_render: bool,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: BlenderProgressCallback | None = None,
    stall_timeout_seconds: int = DEFAULT_RENDER_STALL_TIMEOUT_SECONDS,
) -> Path:
    blender_directory = gap_directory / "blender"
    output_path = blender_directory / "gap_blender.mp4"
    report_path = blender_directory / "render_report.json"
    blend_path = blender_directory / "scene.blend"
    log_path = blender_directory / "blender.log"
    frame_directory = _frame_directory(plan_path, blender_directory)
    if reuse_render and _render_cache_is_complete(plan_path, output_path, report_path, blend_path):
        return output_path
    request = BlenderRenderRequest(
        plan_path, frame_directory, report_path, blend_path, log_path, "sparse_animation",
    )
    render_result = render_with_blender(
        project_root,
        request,
        timeout_seconds=stall_timeout_seconds,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    if not isinstance(render_result, BlenderRenderResult):
        return output_path
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    encode_png_sequence(
        frame_directory,
        min(float(plan["fps"]), float(plan["render"].get("target_fps", 10))),
        VideoContract(
            width=_production_resolution(plan["render"])[0],
            height=_production_resolution(plan["render"])[1],
            fps=float(plan["fps"]),
            frame_count=int(plan["frame_count"]),
        ),
        output_path,
        cancellation_check,
    )
    return output_path


def _write_gap_plans(
    video_path: Path,
    scene_report: dict,
    identity_registry: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
    cancellation_check: CancellationCheck | None,
) -> list[Path]:
    plan_paths: list[Path] = []
    for gap_index, hidden_range_items in enumerate(hidden_ranges):
        raise_if_cancelled(cancellation_check)
        hidden_range = (int(hidden_range_items[0]), int(hidden_range_items[1]))
        gap_directory = work_directory / "gaps" / f"gap_{gap_index:02d}" / "blender"
        context_path = export_forensic_context_frame(
            video_path,
            hidden_range[0] - 1,
            scene_report,
            gap_directory / "visible_boundary_context.jpg",
            cancellation_check,
        )
        post_context_path = export_forensic_context_frame(
            video_path,
            min(hidden_range[1] + 1, int(scene_report["video"]["frames"]) - 1),
            scene_report,
            gap_directory / "visible_boundary_context_after.jpg",
            cancellation_check,
        )
        plan = build_reconstruction_plan_v2(
            scene_report,
            identity_registry,
            hidden_range,
            gap_index,
            maximum_entities=maximum_entities,
            context_frame_path=context_path,
            post_context_frame_path=post_context_path,
            render_configuration=render_configuration,
        )
        plan_path = gap_directory / "plan_v2.json"
        write_reconstruction_plan_v2(plan, plan_path)
        plan_paths.append(plan_path)
    _write_plan_manifest(plan_paths, work_directory / "reconstruction_plans_v2.json")
    return plan_paths


def _write_plan_manifest(plan_paths: list[Path], output_path: Path) -> None:
    manifest: list[dict] = []
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest.append({
            "gap_index": plan["gap_index"],
            "hidden_range": plan["hidden_range"],
            "rendered_entities": plan["selection_report"]["rendered_count"],
            "overall_confidence": plan["overall_confidence"],
            "calibration_confidence": plan["camera"]["calibration_confidence"],
            "plan_path": str(plan_path),
        })
    _write_json(output_path, manifest)


def _write_json(output_path: Path, payload: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _render_cache_is_complete(
    plan_path: Path,
    output_path: Path,
    report_path: Path,
    blend_path: Path,
) -> bool:
    required_paths = (plan_path, output_path, report_path, blend_path)
    if not all(_is_nonempty_file(path) for path in required_paths):
        return False
    try:
        plan_bytes = plan_path.read_bytes()
        plan = json.loads(plan_bytes)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            return False
        validate_reconstruction_plan_v2(plan)
    except (
        AttributeError, KeyError, OSError, UnicodeDecodeError,
        json.JSONDecodeError, PlanValidationError, TypeError, ValueError,
    ):
        return False
    if not _render_report_matches_plan(plan_bytes, plan, report):
        return False
    return _cached_video_matches_plan(output_path, plan)


def _render_report_matches_plan(plan_bytes: bytes, plan: dict, report: object) -> bool:
    if not isinstance(report, dict) or not REQUIRED_RENDER_REPORT_FIELDS.issubset(report):
        return False
    try:
        render_contract = plan["render"]
        expected_resolution = _production_resolution(render_contract)
        return all((
            report["plan_hash"] == hashlib.sha256(plan_bytes).hexdigest(),
            report["mode"] in {"animation", "sparse_animation"},
            report["render_engine"] == render_contract["engine"],
            int(report["frame_count"]) == int(plan["frame_count"]),
            report["resolution"] == expected_resolution,
            abs(float(report["fps"]) - float(plan["fps"])) <= RENDER_FPS_TOLERANCE,
        ))
    except (KeyError, TypeError, ValueError):
        return False


def _production_resolution(render_contract: dict) -> list[int]:
    scale = int(render_contract["production_scale_percent"]) / 100.0
    return [
        _even_render_dimension(round(int(render_contract["source_width"]) * scale)),
        _even_render_dimension(round(int(render_contract["source_height"]) * scale)),
    ]


def _cached_video_matches_plan(output_path: Path, plan: dict) -> bool:
    try:
        video_contract = inspect_video_contract(output_path)
        expected_resolution = _production_resolution(plan["render"])
        return all((
            [video_contract.width, video_contract.height] == expected_resolution,
            video_contract.frame_count == int(plan["frame_count"]),
            abs(video_contract.fps - float(plan["fps"])) <= RENDER_FPS_TOLERANCE,
        ))
    except (KeyError, MediaProcessingError, OSError, TypeError, ValueError):
        return False


def _even_render_dimension(value: int) -> int:
    bounded_value = max(2, value)
    return bounded_value if bounded_value % 2 == 0 else bounded_value + 1


def _frame_directory(plan_path: Path, blender_directory: Path) -> Path:
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()[:12]
    return blender_directory / "renders" / f"frames_{plan_hash}"


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
