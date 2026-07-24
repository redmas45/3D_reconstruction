import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scene_builder import build_scene
from render_passes import (
    configure_diagnostic_passes,
    diagnostic_layer_files,
    set_diagnostic_output_enabled,
)


def parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a validated forensic reconstruction plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--mode", choices=("preview", "animation", "sparse_animation"), default="preview")
    return parser.parse_args(arguments)


def load_plan(plan_path: str) -> dict:
    with Path(plan_path).open("r", encoding="utf-8") as plan_file:
        plan = json.load(plan_file)
    if plan.get("schema_version") != 2:
        raise ValueError("Blender accepts only reconstruction plan schema version 2")
    return plan


def render_preview(scene: bpy.types.Scene, plan: dict, output_path: Path) -> None:
    midpoint = max(1, round(plan["frame_count"] / 2))
    scene.frame_set(midpoint)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def render_animation(scene: bpy.types.Scene, plan: dict, output_path: Path) -> None:
    render_contract = plan.get("render", {})
    scale = int(render_contract.get("production_scale_percent", 100)) / 100.0
    scene.render.resolution_x = _even_render_dimension(
        round(int(render_contract["source_width"]) * scale)
    )
    scene.render.resolution_y = _even_render_dimension(
        round(int(render_contract["source_height"]) * scale)
    )
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(output_path)
    bpy.app.handlers.render_post.append(report_render_progress)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.render_post.remove(report_render_progress)


def render_sparse_animation(scene: bpy.types.Scene, plan: dict, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    render_contract = plan.get("render", {})
    scale = int(render_contract.get("production_scale_percent", 100)) / 100.0
    scene.render.resolution_x = _even_render_dimension(
        round(int(render_contract["source_width"]) * scale)
    )
    scene.render.resolution_y = _even_render_dimension(
        round(int(render_contract["source_height"]) * scale)
    )
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = bool(
        plan.get("environment", {}).get("hybrid_backplate_enabled", False),
    )
    total_frames = int(plan["frame_count"])
    diagnostic_frames = set(_diagnostic_pose_frames(
        total_frames,
        int(render_contract.get("diagnostic_pose_count", 5)),
    ))
    plan["_diagnostic_pass_report"] = configure_diagnostic_passes(scene, output_directory)
    manifest_entries = []
    for frame_index in range(1, total_frames + 1):
        set_diagnostic_output_enabled(scene, frame_index in diagnostic_frames)
        output_path = output_directory / f"frame_{frame_index:06d}.png"
        if not _valid_png(output_path):
            _render_atomic_frame(scene, frame_index, output_path)
        manifest_entries.append(_frame_manifest_entry(output_path, frame_index))
        _write_frame_manifest(output_directory, plan, manifest_entries)
        print(f"RECON_PROGRESS {frame_index} {total_frames}", flush=True)
    set_diagnostic_output_enabled(scene, False)


def _render_atomic_frame(scene: bpy.types.Scene, frame_index: int, output_path: Path) -> None:
    temporary_path = output_path.with_name(f"{output_path.stem}.rendering.png")
    temporary_path.unlink(missing_ok=True)
    scene.frame_set(frame_index)
    scene.render.filepath = str(temporary_path)
    bpy.ops.render.render(write_still=True)
    if not _valid_png(temporary_path):
        raise RuntimeError(f"Blender did not produce a valid PNG for frame {frame_index}")
    temporary_path.replace(output_path)


def _valid_png(path: Path) -> bool:
    try:
        with path.open("rb") as image_file:
            return path.stat().st_size > 64 and image_file.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _frame_manifest_entry(path: Path, frame_index: int) -> dict:
    return {
        "frame_index": frame_index,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "status": "complete",
    }


def _write_frame_manifest(output_directory: Path, plan: dict, frames: list[dict]) -> None:
    manifest_path = output_directory / "frame_manifest.json"
    temporary_path = output_directory / "frame_manifest.json.tmp"
    payload = {
        "schema_version": 1,
        "plan_digest": _json_digest(plan),
        "decision_digest": plan.get("reasoning_decision_v2", {}).get("hypothesis_digest"),
        "render_profile_digest": _json_digest(plan.get("render", {})),
        "target_fps": plan["fps"],
        "target_frame_count": plan["frame_count"],
        "completed_frame_count": len(frames),
        "frames": frames,
    }
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sparse_runtime_plan(plan: dict) -> dict:
    runtime_plan = copy.deepcopy(plan)
    source_fps = float(plan["fps"])
    target_fps = min(source_fps, float(plan.get("render", {}).get("target_fps", 10)))
    target_count = max(2, round(float(plan["duration_seconds"]) * target_fps))
    runtime_plan["fps"] = target_fps
    runtime_plan["frame_count"] = target_count
    for entity in runtime_plan.get("entities", []):
        for waypoint in entity["path_prediction"]["waypoints"]:
            source_frame = int(waypoint["frame"])
            normalized = (source_frame - int(plan["hidden_range"]["start"])) / max(1, int(plan["frame_count"]) - 1)
            waypoint["frame"] = 1 + round(normalized * (target_count - 1))
    return runtime_plan


def report_render_progress(scene: bpy.types.Scene) -> None:
    print(f"RECON_PROGRESS {scene.frame_current} {scene.frame_end}", flush=True)


def write_report(
    arguments: argparse.Namespace, plan: dict, started_at: float, output_path: Path,
) -> None:
    plan_bytes = Path(arguments.plan).read_bytes()
    report = {
        "blender_version": bpy.app.version_string,
        "render_engine": bpy.context.scene.render.engine,
        "render_compute_device": plan["render"].get("cycles_compute_device", "graphics"),
        "cycles_samples": plan["render"].get("cycles_samples"),
        "mode": arguments.mode,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "frame_count": plan["frame_count"],
        "rendered_frame_count": bpy.context.scene.frame_end,
        "resolution": _actual_resolution(bpy.context.scene),
        "fps": plan["fps"],
        "rendered_fps": bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
        "source_fps": plan["fps"],
        "source_frame_count": plan["frame_count"],
        "plan_hash": hashlib.sha256(plan_bytes).hexdigest(),
        "calibration_confidence": plan["camera"]["calibration_confidence"],
        "calibration_report": plan["camera"]["calibration_report"],
        "overall_confidence": plan["overall_confidence"],
        "identity_registry": plan["identity_registry"],
        "entity_fidelity_counts": _fidelity_counts(plan),
        "motion_profiles": _motion_profiles(plan),
        "selection_report": plan["selection_report"],
        "boundary_residuals": _boundary_residuals(plan),
        "output_path": str(output_path),
        "frame_manifest_path": (
            str(output_path / "frame_manifest.json")
            if arguments.mode == "sparse_animation" else None
        ),
        "diagnostic_passes": _diagnostic_pass_report(plan, output_path),
        "warnings": _render_warnings(plan),
    }
    Path(arguments.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _diagnostic_pass_report(plan: dict, output_path: Path) -> dict:
    configured = plan.get("_diagnostic_pass_report", {})
    if not configured.get("available") or not output_path.is_dir():
        return configured
    return {
        **configured,
        "files": diagnostic_layer_files(output_path),
    }


def _diagnostic_pose_frames(frame_count: int, pose_count: int) -> list[int]:
    bounded_count = min(max(1, pose_count), frame_count)
    if bounded_count == 1:
        return [1]
    return sorted({
        1 + round(index * (frame_count - 1) / (bounded_count - 1))
        for index in range(bounded_count)
    })


def _actual_resolution(scene: bpy.types.Scene) -> list[int]:
    scale = scene.render.resolution_percentage / 100.0
    return [round(scene.render.resolution_x * scale), round(scene.render.resolution_y * scale)]


def _even_render_dimension(value: int) -> int:
    bounded_value = max(2, value)
    return bounded_value if bounded_value % 2 == 0 else bounded_value + 1


def _fidelity_counts(plan: dict) -> dict[str, int]:
    counts = {"supported": 0, "plausible": 0, "weak": 0}
    for entity in plan["entities"]:
        tier = str(entity["fidelity_tier"])
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _motion_profiles(plan: dict) -> list[dict]:
    return [
        {
            "id": entity["id"],
            "clip": entity.get("motion_profile", {}).get(
                "clip",
                entity.get("animation", {}).get("state", "idle"),
            ),
            "source": entity.get("motion_profile", {}).get(
                "source",
                "kinematic_fallback",
            ),
            "pose_confidence": entity.get("motion_profile", {}).get(
                "pose_confidence",
                0.0,
            ),
        }
        for entity in plan["entities"]
        if entity["kind"] == "person"
    ]


def _boundary_residuals(plan: dict) -> list[dict]:
    return [
        {
            "id": entity["id"],
            "post_gap_position_residual_meters": entity["boundary_evidence"]["post_gap_position_residual_meters"],
            "heading_disagreement_degrees": entity["boundary_evidence"]["heading_disagreement_degrees"],
        }
        for entity in plan["entities"]
    ]


def _render_warnings(plan: dict) -> list[str]:
    warnings = []
    if plan["camera"].get("mode") == "generic_ground_prior":
        warnings.append("Camera geometry uses a generic ground prior, not a solved calibration")
    if plan["camera"].get("motion_model") == "dynamic_camera":
        warnings.append("Dynamic source footage is shown as a labelled stabilized forensic view")
    if plan["camera"]["calibration_confidence"] < 0.75:
        warnings.append("Camera calibration requires visual review")
    weak_count = sum(entity["fidelity_tier"] == "weak" for entity in plan["entities"])
    if weak_count:
        warnings.append(f"{weak_count} low-confidence entities use simplified proxy fidelity")
    uncertain_count = sum(entity["lifecycle"] == "uncertain" for entity in plan["entities"])
    if uncertain_count:
        warnings.append(f"{uncertain_count} heading-conflicted identities remain explicitly uncertain")
    excluded_count = len(plan["selection_report"]["excluded_ids"])
    if excluded_count:
        warnings.append(f"{excluded_count} entities remain report-only after presentation filtering")
    return warnings


def main() -> None:
    arguments = parse_arguments()
    started_at = time.time()
    source_plan = load_plan(arguments.plan)
    plan = _sparse_runtime_plan(source_plan) if arguments.mode == "sparse_animation" else source_plan
    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene(plan)
    Path(arguments.blend).resolve().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(arguments.blend).resolve()))
    if arguments.mode == "preview":
        render_preview(scene, plan, output_path)
    elif arguments.mode == "sparse_animation":
        render_sparse_animation(scene, plan, output_path)
        source_plan["_diagnostic_pass_report"] = plan.get("_diagnostic_pass_report", {})
    else:
        render_animation(scene, plan, output_path)
    write_report(arguments, source_plan, started_at, output_path)


if __name__ == "__main__":
    main()
