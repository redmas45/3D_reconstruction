"""Exports a bounded set of visible frames and crops for multimodal reasoning."""

import hashlib
from pathlib import Path

import cv2

from domain.cancellation import CancellationCheck, raise_if_cancelled
from infrastructure.json_files import write_json_file


VISUAL_EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_GLOBAL_KEYFRAMES = 8
DEFAULT_BOUNDARY_FRAMES_PER_SIDE = 2
DEFAULT_CROPS_PER_TRACK = 2
JPEG_QUALITY = 86


def export_visible_evidence(
    video_path: Path,
    scene_report: dict,
    plans: list[dict],
    evidence_directory: Path,
    configuration: dict,
    cancellation_check: CancellationCheck | None = None,
) -> dict:
    selections = _frame_selections(scene_report, plans, configuration)
    manifest = _empty_manifest(scene_report)
    if not video_path.is_file():
        write_json_file(evidence_directory / "visual_evidence_manifest.json", manifest)
        return manifest
    frames = _read_selected_frames(video_path, selections, cancellation_check)
    manifest["images"] = _write_images(
        frames, selections, evidence_directory, scene_report, cancellation_check,
    )
    manifest["manifest_digest"] = _manifest_digest(manifest)
    write_json_file(evidence_directory / "visual_evidence_manifest.json", manifest)
    return manifest


def validate_visual_evidence_manifest(manifest: dict, scene_report: dict) -> None:
    hidden_ranges = [
        (int(item["start"]), int(item["end"])) for item in scene_report.get("hidden_ranges", [])
    ]
    for image in manifest.get("images", []):
        frame_index = int(image["frame"])
        if any(start <= frame_index <= end for start, end in hidden_ranges):
            raise ValueError(f"Visual evidence includes hidden frame {frame_index}")
        path = Path(str(image["path"]))
        if not path.is_file() or _file_sha256(path) != image["sha256"]:
            raise ValueError(f"Visual evidence image is missing or changed: {path.name}")


def _frame_selections(scene_report: dict, plans: list[dict], configuration: dict) -> list[dict]:
    visible_frames = _visible_frame_indexes(scene_report)
    selections = _global_selections(visible_frames, configuration)
    selections.extend(_boundary_selections(plans, visible_frames, configuration))
    selections.extend(_crop_selections(scene_report, plans, visible_frames, configuration))
    unique = {(item["kind"], item["frame"], item.get("track_id"), item.get("gap_index")): item for item in selections}
    return sorted(unique.values(), key=lambda item: (item["frame"], item["kind"]))


def _visible_frame_indexes(scene_report: dict) -> list[int]:
    indexes: list[int] = []
    for item in scene_report.get("visible_ranges", []):
        indexes.extend(range(int(item["start"]), int(item["end"]) + 1))
    if indexes:
        return indexes
    frame_count = int(scene_report.get("video", {}).get("frames", 0))
    hidden = [(int(item["start"]), int(item["end"])) for item in scene_report.get("hidden_ranges", [])]
    return [index for index in range(frame_count) if not any(start <= index <= end for start, end in hidden)]


def _global_selections(visible_frames: list[int], configuration: dict) -> list[dict]:
    limit = max(1, int(configuration.get("max_global_keyframes", DEFAULT_GLOBAL_KEYFRAMES)))
    if len(visible_frames) <= limit:
        indexes = visible_frames
    elif limit == 1:
        indexes = [visible_frames[len(visible_frames) // 2]]
    else:
        indexes = [visible_frames[round(position * (len(visible_frames) - 1) / (limit - 1))] for position in range(limit)]
    return [{"kind": "global_keyframe", "frame": index} for index in indexes]


def _boundary_selections(plans: list[dict], visible_frames: list[int], configuration: dict) -> list[dict]:
    visible = set(visible_frames)
    count = max(1, int(configuration.get(
        "boundary_frames_per_side", DEFAULT_BOUNDARY_FRAMES_PER_SIDE,
    )))
    selections = []
    for plan in plans:
        start, end = int(plan["hidden_range"]["start"]), int(plan["hidden_range"]["end"])
        gap_index = int(plan["gap_index"])
        for frame_index in range(start - count, start):
            if frame_index in visible:
                selections.append({"kind": "pre_boundary", "frame": frame_index, "gap_index": gap_index})
        for frame_index in range(end + 1, end + count + 1):
            if frame_index in visible:
                selections.append({"kind": "post_boundary", "frame": frame_index, "gap_index": gap_index})
    return selections


def _crop_selections(
    scene_report: dict,
    plans: list[dict],
    visible_frames: list[int],
    configuration: dict,
) -> list[dict]:
    relevant_ids = {str(entity["id"]) for plan in plans for entity in plan.get("entities", [])}
    visible = set(visible_frames)
    count = max(1, int(configuration.get("crops_per_track", DEFAULT_CROPS_PER_TRACK)))
    selections = []
    for track in scene_report.get("tracks", []):
        if str(track.get("id")) not in relevant_ids:
            continue
        detections = [item for item in track.get("detections", []) if int(item["frame"]) in visible]
        for detection in _even_samples(detections, count):
            selections.append({
                "kind": "entity_crop",
                "frame": int(detection["frame"]),
                "track_id": str(track["id"]),
                "bbox": [int(value) for value in detection["bbox"]],
            })
    return selections


def _even_samples(items: list[dict], count: int) -> list[dict]:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(position * (len(items) - 1) / (count - 1))] for position in range(count)]


def _read_selected_frames(
    video_path: Path,
    selections: list[dict],
    cancellation_check: CancellationCheck | None,
) -> dict[int, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video: {video_path.name}")
    frames: dict[int, object] = {}
    try:
        for frame_index in sorted({int(item["frame"]) for item in selections}):
            raise_if_cancelled(cancellation_check)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Cannot read visible evidence frame {frame_index}")
            frames[frame_index] = frame
    finally:
        capture.release()
    return frames


def _write_images(
    frames: dict[int, object],
    selections: list[dict],
    evidence_directory: Path,
    scene_report: dict,
    cancellation_check: CancellationCheck | None,
) -> list[dict]:
    images = []
    for image_index, selection in enumerate(selections):
        raise_if_cancelled(cancellation_check)
        frame_index = int(selection["frame"])
        frame = frames[frame_index]
        relative_directory = "crops" if selection["kind"] == "entity_crop" else "keyframes"
        output_path = evidence_directory / relative_directory / f"{image_index:03d}_{selection['kind']}_{frame_index:06d}.jpg"
        image = _crop_frame(frame, selection.get("bbox"), scene_report)
        _write_jpeg(output_path, image)
        images.append({
            "id": f"image_{image_index:03d}",
            "kind": selection["kind"],
            "frame": frame_index,
            "gap_index": selection.get("gap_index"),
            "track_id": selection.get("track_id"),
            "path": str(output_path.resolve()),
            "sha256": _file_sha256(output_path),
        })
    return images


def _crop_frame(frame: object, bbox: object, scene_report: dict) -> object:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return frame
    width = int(scene_report["video"]["width"])
    height = int(scene_report["video"]["height"])
    x1, y1, x2, y2 = bbox
    padding = max(4, round(max(x2 - x1, y2 - y1) * 0.15))
    left, top = max(0, x1 - padding), max(0, y1 - padding)
    right, bottom = min(width, x2 + padding), min(height, y2 + padding)
    return frame[top:bottom, left:right]


def _write_jpeg(output_path: Path, image: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"Cannot write visual evidence image: {output_path.name}")


def _empty_manifest(scene_report: dict) -> dict:
    return {
        "schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_policy": "visible_frames_only",
        "source_sha256": scene_report.get("video", {}).get("sha256"),
        "images": [],
        "manifest_digest": "",
    }


def _manifest_digest(manifest: dict) -> str:
    digest = hashlib.sha256()
    for image in manifest["images"]:
        digest.update(str(image["id"]).encode("utf-8"))
        digest.update(str(image["sha256"]).encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
