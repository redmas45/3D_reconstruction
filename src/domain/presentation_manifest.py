from pathlib import Path

from infrastructure.json_files import read_json_file, write_json_file


PRESENTATION_SCHEMA_VERSION = 3
MAXIMUM_PRESENTED_CLUES = 8
MAXIMUM_PRESENTED_GAP_CLUES = 6
MAXIMUM_PRESENTED_ENTITIES_PER_GAP = 12
MAXIMUM_PRESENTED_REJECTIONS_PER_ENTITY = 4
MAXIMUM_PRESENTED_EVIDENCE_REFERENCES = 8
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
    observed_fraction = round(
        1.0 - float(gap_selection["missing_fraction_actual"]), 4,
    )
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
            "observed_fraction": observed_fraction,
        },
        "story": _story_contract(reasoning, scene_report),
        "evidence_overview": _evidence_overview(
            reasoning, scene_report, duration_seconds, observed_fraction,
        ),
        "method": _method_contract(reasoning, gaps),
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
    clues = {
        str(item["id"]): item
        for item in reasoning.get("clues", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    decisions = {
        int(item["gap_index"]): item
        for item in reasoning.get("decisions", [])
        if isinstance(item, dict) and isinstance(item.get("gap_index"), int)
    }
    return [
        _gap_contract(index, hidden_range, fps, summaries, decisions, clues, plans)
        for index, hidden_range in enumerate(hidden_ranges)
    ]


def _gap_contract(
    gap_index: int,
    hidden_range: list[int],
    fps: float,
    summaries: dict[int, dict],
    decisions: dict[int, dict],
    clues: dict[str, dict],
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
        "evidence_references": _text_items(
            decision.get("evidence_references", []),
        )[:MAXIMUM_PRESENTED_EVIDENCE_REFERENCES],
        "clues": _gap_clues(decision, clues),
        "entities": _entity_decisions(decision),
        "event_beats": _event_beats(decision),
        "entity_count": len(plan.get("entities", [])),
        "calibration_confidence": round(
            float(plan.get("camera", {}).get("calibration_confidence", 0.0)), 4,
        ),
        "patch": _patch_contract(summary, decision, plan),
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
        "planning_mode": str(reasoning.get("mode", "unknown")),
        "deployment": str(reasoning.get("deployment") or "deterministic fallback"),
        "warning": str(reasoning.get("warning") or ""),
        "unknowns": _text_items(reasoning.get("unknowns", [])),
        "points": [
            str(item["statement"])
            for item in reasoning.get("story_points", [])
            if isinstance(item, dict) and isinstance(item.get("statement"), str)
        ],
    }


def _evidence_overview(
    reasoning: dict,
    scene_report: dict,
    duration_seconds: float,
    observed_fraction: float,
) -> dict:
    people_count = int(scene_report.get("people_count", 0))
    vehicle_count = int(scene_report.get("vehicle_count", 0))
    tracked_entity_count = len(scene_report.get("tracks", []))
    clue_count = len(reasoning.get("clues", []))
    observed_seconds = duration_seconds * observed_fraction
    return {
        "summary": _visible_evidence_summary(
            reasoning, observed_seconds, people_count, vehicle_count,
        ),
        "observed_seconds": round(observed_seconds, 3),
        "missing_seconds": round(duration_seconds - observed_seconds, 3),
        "tracked_entity_count": tracked_entity_count,
        "people_count": people_count,
        "vehicle_count": vehicle_count,
        "clue_count": clue_count,
    }


def _visible_evidence_summary(
    reasoning: dict,
    observed_seconds: float,
    people_count: int,
    vehicle_count: int,
) -> str:
    entity_summary = (
        f"{people_count} people and {vehicle_count} vehicles"
        if people_count or vehicle_count
        else "the visible actors and scene context"
    )
    strongest_clues = _top_clues(reasoning)
    clue_summary = (
        f" The strongest recorded clue was: {strongest_clues[0]['statement']}"
        if strongest_clues else ""
    )
    return (
        f"Across {observed_seconds:.1f} seconds of visible footage, the system "
        f"tracked {entity_summary} and measured their continuity around every gap."
        f"{clue_summary}"
    )


def _method_contract(reasoning: dict, gaps: list[dict]) -> dict:
    planning_mode = str(reasoning.get("mode", "deterministic fallback"))
    planner = "Azure-assisted" if planning_mode == "azure" else "Deterministic"
    return {
        "label": "Public decision trace",
        "description": (
            "A judge-facing explanation of the evidence and validated decisions. "
            "It is not private model chain-of-thought."
        ),
        "steps": [
            _method_step("observe", "Observe", "Analyze only the visible 75% of the video."),
            _method_step(
                "measure", "Measure",
                "Track entities, directions, continuity, boundary frames, and camera stability.",
            ),
            _method_step(
                "decide", "Decide",
                f"{planner} planning compares bounded hypotheses and validates every reference.",
            ),
            _method_step(
                "patch", "Patch",
                f"Render and stitch {len(gaps)} inferred intervals while preserving the source timeline.",
            ),
        ],
    }


def _method_step(identifier: str, title: str, description: str) -> dict:
    return {
        "id": identifier,
        "title": title,
        "description": description,
        "status": "completed",
    }


def _patch_contract(summary: dict, decision: dict, plan: dict) -> dict:
    environment = plan.get("environment", {})
    method = (
        "Stylized 3D actors composited over observed scene context"
        if environment.get("hybrid_backplate_enabled")
        else "Stylized 3D scene reconstruction"
    )
    return {
        "method": method,
        "summary": str(summary.get(
            "inside_inferred",
            decision.get("gap_summary", "Bounded motion hypothesis."),
        )),
        "boundary_basis": (
            "Motion starts from the last visible state; the first visible state "
            "after the gap is used as a consistency check."
        ),
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


def _gap_clues(decision: dict, clues: dict[str, dict]) -> list[dict]:
    presented = []
    for clue_id in _text_items(decision.get("clue_ids", [])):
        clue = clues.get(clue_id)
        if clue is None:
            continue
        presented.append({
            "id": clue_id,
            "statement": str(clue.get("statement", "Visible evidence clue")),
            "confidence": round(float(clue.get("confidence", 0.0)), 4),
        })
    return presented[:MAXIMUM_PRESENTED_GAP_CLUES]


def _entity_decisions(decision: dict) -> list[dict]:
    entities = decision.get("entities", [])
    if not isinstance(entities, list):
        return []
    return [
        _entity_decision(item)
        for item in entities[:MAXIMUM_PRESENTED_ENTITIES_PER_GAP]
        if isinstance(item, dict)
    ]


def _entity_decision(entity: dict) -> dict:
    rejected = entity.get("rejected_hypotheses", [])
    rejected_items = rejected if isinstance(rejected, list) else []
    return {
        "entity_id": str(entity.get("entity_id", "entity")),
        "selected_hypothesis_id": str(entity.get("selected_hypothesis_id", "unknown")),
        "decision_summary": str(entity.get("decision_summary", "Bounded hypothesis selected.")),
        "confidence": round(float(entity.get("confidence", 0.0)), 4),
        "rejected_hypotheses": [
            {
                "id": str(item.get("id", "alternative")),
                "reason": str(item.get("reason", "Less supported by visible evidence.")),
            }
            for item in rejected_items[:MAXIMUM_PRESENTED_REJECTIONS_PER_ENTITY]
            if isinstance(item, dict)
        ],
    }


def _event_beats(decision: dict) -> list[dict]:
    beats = decision.get("event_beats", [])
    if not isinstance(beats, list):
        return []
    return [
        {
            "time_fraction": round(float(item.get("time_fraction", 0.0)), 4),
            "action": str(item.get("action", "continue")),
            "entity_ids": _text_items(item.get("entity_ids", [])),
        }
        for item in beats
        if isinstance(item, dict)
    ]


def _text_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]
