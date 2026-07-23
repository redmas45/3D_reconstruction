from pathlib import Path

from infrastructure.json_files import read_json_file, write_json_file


PRESENTATION_SCHEMA_VERSION = 1
MAXIMUM_PRESENTED_CLUES = 8
PUBLIC_REASONING_FILENAME = "reasoning_public.json"


def build_presentation_manifest(
    video_info: dict,
    gap_selection: dict,
    scene_report: dict,
    blender_plan_paths: list[Path],
    work_directory: Path,
    output_path: Path,
    renderer_mode: str,
) -> dict:
    reasoning = _read_reasoning(work_directory)
    plans = _read_plans(blender_plan_paths)
    fps = float(video_info["fps"])
    gaps = _presentation_gaps(
        gap_selection["hidden_ranges"], fps, reasoning, plans,
    )
    duration_seconds = float(video_info["frames"]) / fps
    return {
        "schema_version": PRESENTATION_SCHEMA_VERSION,
        "status": "completed",
        "title": "AI-inferred evidence reconstruction",
        "disclosure": (
            "Reconstructed intervals are hypotheses derived only from the visible "
            "75% of the video; they are not recovered ground truth."
        ),
        "source": {
            "duration_seconds": round(duration_seconds, 3),
            "fps": fps,
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "observed_fraction": round(
                1.0 - float(gap_selection["missing_fraction_actual"]), 4,
            ),
        },
        "story": _story_contract(reasoning, scene_report),
        "top_clues": _top_clues(reasoning),
        "gaps": gaps,
        "render": _render_contract(renderer_mode, plans),
        "output": {
            "filename": output_path.name,
            "gap_count": len(gaps),
            "reconstructed_fraction": round(
                float(gap_selection["missing_fraction_actual"]), 4,
            ),
        },
    }


def write_presentation_manifest(manifest: dict, output_path: Path) -> None:
    write_json_file(output_path, manifest)


def _read_reasoning(work_directory: Path) -> dict:
    path = work_directory / PUBLIC_REASONING_FILENAME
    if not path.is_file():
        return {}
    payload = read_json_file(path)
    return payload if isinstance(payload, dict) else {}


def _read_plans(plan_paths: list[Path]) -> dict[int, dict]:
    plans: dict[int, dict] = {}
    for plan_path in plan_paths:
        payload = read_json_file(plan_path)
        if isinstance(payload, dict) and isinstance(payload.get("gap_index"), int):
            plans[int(payload["gap_index"])] = payload
    return plans


def _presentation_gaps(
    hidden_ranges: list[list[int]],
    fps: float,
    reasoning: dict,
    plans: dict[int, dict],
) -> list[dict]:
    summaries = {
        int(item["gap_index"]): item
        for item in reasoning.get("gap_summaries", [])
        if isinstance(item, dict) and isinstance(item.get("gap_index"), int)
    }
    decisions = {
        int(item["gap_index"]): item
        for item in reasoning.get("decisions", [])
        if isinstance(item, dict) and isinstance(item.get("gap_index"), int)
    }
    return [
        _gap_contract(index, hidden_range, fps, summaries, decisions, plans)
        for index, hidden_range in enumerate(hidden_ranges)
    ]


def _gap_contract(
    gap_index: int,
    hidden_range: list[int],
    fps: float,
    summaries: dict[int, dict],
    decisions: dict[int, dict],
    plans: dict[int, dict],
) -> dict:
    start_frame, end_frame = int(hidden_range[0]), int(hidden_range[1])
    summary = summaries.get(gap_index, {})
    decision = decisions.get(gap_index, {})
    plan = plans.get(gap_index, {})
    confidence = summary.get(
        "confidence", decision.get("confidence", plan.get("overall_confidence", 0.0)),
    )
    return {
        "gap_index": gap_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": round(start_frame / fps, 3),
        "end_seconds": round((end_frame + 1) / fps, 3),
        "duration_seconds": round((end_frame - start_frame + 1) / fps, 3),
        "confidence": round(float(confidence), 4),
        "before_observed": str(summary.get("before_observed", "Visible evidence before the gap.")),
        "inside_inferred": str(summary.get(
            "inside_inferred", decision.get("gap_summary", "Bounded motion hypothesis."),
        )),
        "after_observed": str(summary.get("after_observed", "Visible evidence after the gap.")),
        "unknowns": _text_items(summary.get("unknowns", decision.get("unknowns", []))),
        "clue_ids": _text_items(decision.get("clue_ids", [])),
        "entity_count": len(plan.get("entities", [])),
        "calibration_confidence": round(
            float(plan.get("camera", {}).get("calibration_confidence", 0.0)), 4,
        ),
    }


def _story_contract(reasoning: dict, scene_report: dict) -> dict:
    fallback_summary = str(
        scene_report.get("scene_summary", "Visible evidence was analyzed across the source video."),
    )
    return {
        "headline": str(reasoning.get("headline", "Evidence-grounded reconstruction")),
        "summary": str(reasoning.get("whole_video_summary", fallback_summary)),
        "confidence": round(float(reasoning.get("confidence", 0.0)), 4),
        "causal_link_supported": bool(reasoning.get("causal_link_supported", False)),
        "points": [
            str(item["statement"])
            for item in reasoning.get("story_points", [])
            if isinstance(item, dict) and isinstance(item.get("statement"), str)
        ],
    }


def _top_clues(reasoning: dict) -> list[dict]:
    clues = [
        item for item in reasoning.get("clues", [])
        if isinstance(item, dict) and isinstance(item.get("statement"), str)
    ]
    ranked = sorted(
        clues,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    return [
        {
            "id": str(item.get("id", "")),
            "category": str(item.get("category", "evidence")),
            "statement": str(item["statement"]),
            "confidence": round(float(item.get("confidence", 0.0)), 4),
        }
        for item in ranked[:MAXIMUM_PRESENTED_CLUES]
    ]


def _render_contract(renderer_mode: str, plans: dict[int, dict]) -> dict:
    first_plan = plans[min(plans)] if plans else {}
    render = first_plan.get("render", {})
    environment = first_plan.get("environment", {})
    return {
        "mode": renderer_mode,
        "engine": str(render.get("engine", "2.5D")),
        "target_fps": int(render.get("target_fps", 0)),
        "hybrid_static_backplate": bool(environment.get("hybrid_backplate_enabled", False)),
        "production_hud_mode": str(render.get("production_hud_mode", "minimal")),
    }


def _text_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
