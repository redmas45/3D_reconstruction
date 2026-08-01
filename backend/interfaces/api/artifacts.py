"""Reads a running job's artifacts off disk for the UI.

The pipeline already writes everything the interface needs — gap selection, the clue
catalog, the decision trace, the presentation manifest. This module surfaces them as
they appear rather than only at the end, which is what lets the timeline populate while
detection is still running.

Every reader returns None when its artifact is not written yet. That is the normal
state for most of a run, not an error, and the UI renders whatever is available.
"""

import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)

WORK_DIRECTORY_NAME = "_work"


def _read_json(path: Path):
    """Tolerant of a file being written as we read it, which is expected here."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def work_directory(output_dir: Path) -> Path | None:
    """The per-video working directory, whose name embeds the source digest."""
    root = Path(output_dir) / WORK_DIRECTORY_NAME
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def timeline(output_dir: Path) -> dict | None:
    """Visible ranges and hidden gaps, available as soon as gaps are chosen."""
    work = work_directory(output_dir)
    if work is None:
        return None
    selection = _read_json(work / "gap_selection.json")
    if not isinstance(selection, dict):
        return None
    video = selection.get("source_video_contract") or {}
    fps = float(video.get("fps") or 0.0) or 30.0
    frames = int(video.get("frames") or 0)
    return {
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": round(frames / fps, 3) if frames else None,
        "visible_ranges": [
            {
                "start_frame": int(start), "end_frame": int(end),
                "start_seconds": round(int(start) / fps, 3),
                "end_seconds": round(int(end) / fps, 3),
            }
            for start, end in selection.get("visible_ranges", [])
        ],
        "hidden_ranges": [
            {
                "gap_index": index,
                "start_frame": int(start), "end_frame": int(end),
                "start_seconds": round(int(start) / fps, 3),
                "end_seconds": round(int(end) / fps, 3),
                "duration_seconds": round((int(end) - int(start) + 1) / fps, 3),
            }
            for index, (start, end) in enumerate(selection.get("hidden_ranges", []))
        ],
        "missing_fraction": selection.get("missing_fraction_actual"),
    }


def clues(output_dir: Path) -> dict | None:
    """Tracked entities and their evidence, available once detection finishes."""
    work = work_directory(output_dir)
    if work is None:
        return None
    catalog = _read_json(work / "evidence" / "clue_catalog.json")
    scene = _read_json(work / "scene_report.json")
    if catalog is None and scene is None:
        return None
    tracks = (scene or {}).get("tracks", []) if isinstance(scene, dict) else []
    return {
        "catalog": catalog,
        "entity_count": len(tracks),
        "entities": [
            {
                "id": track.get("id"),
                "class_name": track.get("class_name"),
                "first_frame": track.get("first_frame"),
                "last_frame": track.get("last_frame"),
                "frame_count": track.get("frames_seen"),
                "confidence": track.get("avg_confidence"),
                # What the tracker measured, which is what makes a clue a clue rather
                # than just a box: which way it went and how fast.
                "direction": track.get("direction"),
                "speed_px_sec": track.get("speed_px_sec"),
                "continuity": track.get("continuity_confidence"),
            }
            for track in tracks
        ],
    }


def story(output_dir: Path) -> dict | None:
    """The narrative and per-gap reasoning, available once the planner has run."""
    work = work_directory(output_dir)
    if work is None:
        return None
    presentation = _read_json(work / "presentation_manifest.json")
    if isinstance(presentation, dict):
        return {
            "source": "presentation_manifest",
            "story": presentation.get("story"),
            "top_clues": presentation.get("top_clues"),
            "gaps": presentation.get("gaps"),
            "method": presentation.get("method"),
            "evidence_overview": presentation.get("evidence_overview"),
            "disclosure": presentation.get("disclosure"),
        }
    trace = _read_json(work / "decision_trace.json")
    if isinstance(trace, dict):
        # Available well before the manifest, so reasoning is visible while the render is
        # still running rather than only when the whole job finishes.
        decisions = trace.get("decisions")
        decisions = decisions if isinstance(decisions, list) else []
        metadata = trace.get("metadata") or {}
        return {
            "source": "decision_trace",
            "story": {"summary": [
                decision["gap_summary"] for decision in decisions
                if isinstance(decision, dict) and decision.get("gap_summary")
            ]},
            "method": {
                # An unconfigured planner falls back to deterministic reasoning, and the
                # interface must say which produced the story rather than implying the
                # model was involved.
                "mode": "deterministic" if metadata.get("warning") else "azure_assisted",
                "warning": metadata.get("warning"),
            },
            "gaps": [
                {
                    "gap_index": decision.get("gap_index"),
                    "narrative": decision.get("gap_summary"),
                    "evidence_count": len(decision.get("evidence_references") or []),
                }
                for decision in decisions if isinstance(decision, dict)
            ],
        }
    return None


def render_progress(output_dir: Path) -> dict | None:
    """Per-gap render state, so the UI can show gaps completing one by one."""
    work = work_directory(output_dir)
    if work is None:
        return None
    gaps_root = work / "gaps"
    if not gaps_root.is_dir():
        return None
    gaps = []
    for directory in sorted(gaps_root.glob("gap_*")):
        report = _read_json(directory / "actor_render_report.json")
        video = directory / "gap_actors.mp4"
        gaps.append({
            "gap_index": int(directory.name.split("_")[-1]),
            "completed": video.is_file(),
            "report": report,
            "layer_count": len(list(directory.glob("layers/*/frame_*.png"))),
        })
    return {"gaps": gaps, "completed_count": sum(1 for gap in gaps if gap["completed"])}


def plate_path(output_dir: Path) -> Path | None:
    """The recovered background, which is worth showing: it is what makes this work."""
    work = work_directory(output_dir)
    if work is None:
        return None
    candidate = work / "plate" / "clean_plate.png"
    return candidate if candidate.is_file() else None


def gap_video_path(output_dir: Path, gap_index: int) -> Path | None:
    work = work_directory(output_dir)
    if work is None:
        return None
    for name in ("gap_actors.mp4", "blender/gap_blender.mp4"):
        candidate = work / "gaps" / f"gap_{int(gap_index):02d}" / name
        if candidate.is_file():
            return candidate
    return None


def truth_video_path(output_dir: Path, gap_index: int) -> Path | None:
    """The hidden footage, for side-by-side comparison after the run.

    Only ever read here, at presentation time — never by any reconstruction stage.
    """
    work = work_directory(output_dir)
    if work is None:
        return None
    matches = sorted((work / "segments").glob(f"hidden_{int(gap_index):02d}_*.mp4"))
    return matches[0] if matches else None


def diagnostics(output_dir: Path) -> dict | None:
    work = work_directory(output_dir)
    if work is None:
        return None
    return _read_json(work / "diagnostic_report.json")
