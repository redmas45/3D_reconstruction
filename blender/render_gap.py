import argparse
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


def parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a validated forensic reconstruction plan")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--mode", choices=("preview", "animation"), default="preview")
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
    scene.render.resolution_percentage = int(plan.get("render", {}).get("production_scale_percent", 100))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(animation=True)


def write_report(
    arguments: argparse.Namespace, plan: dict, started_at: float, output_path: Path,
) -> None:
    plan_bytes = Path(arguments.plan).read_bytes()
    report = {
        "blender_version": bpy.app.version_string,
        "render_engine": bpy.context.scene.render.engine,
        "mode": arguments.mode,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "frame_count": plan["frame_count"],
        "resolution": _actual_resolution(bpy.context.scene),
        "fps": bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
        "plan_hash": hashlib.sha256(plan_bytes).hexdigest(),
        "calibration_confidence": plan["camera"]["calibration_confidence"],
        "calibration_report": plan["camera"]["calibration_report"],
        "overall_confidence": plan["overall_confidence"],
        "identity_registry": plan["identity_registry"],
        "entity_fidelity_counts": _fidelity_counts(plan),
        "selection_report": plan["selection_report"],
        "boundary_residuals": _boundary_residuals(plan),
        "output_path": str(output_path),
        "warnings": _render_warnings(plan),
    }
    Path(arguments.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _actual_resolution(scene: bpy.types.Scene) -> list[int]:
    scale = scene.render.resolution_percentage / 100.0
    return [round(scene.render.resolution_x * scale), round(scene.render.resolution_y * scale)]


def _fidelity_counts(plan: dict) -> dict[str, int]:
    counts = {"supported": 0, "plausible": 0, "weak": 0}
    for entity in plan["entities"]:
        counts[entity["fidelity_tier"]] += 1
    return counts


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
    plan = load_plan(arguments.plan)
    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene(plan)
    Path(arguments.blend).resolve().parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(arguments.blend).resolve()))
    if arguments.mode == "preview":
        render_preview(scene, plan, output_path)
    else:
        render_animation(scene, plan, output_path)
    write_report(arguments, plan, started_at, output_path)


if __name__ == "__main__":
    main()
