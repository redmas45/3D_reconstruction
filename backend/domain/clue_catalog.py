"""Builds deterministic, visible-only clues for reconstruction reasoning."""

import hashlib
import json
from collections import Counter


CLUE_CATALOG_SCHEMA_VERSION = 1
MAXIMUM_CLUE_TEXT_LENGTH = 320


class ClueCatalogValidationError(ValueError):
    pass


def build_clue_catalog(scene_report: dict, plans: list[dict]) -> dict:
    clues = _scene_clues(scene_report)
    tracks_by_id = {
        str(track["id"]): track for track in scene_report.get("tracks", [])
        if isinstance(track, dict) and "id" in track
    }
    for plan in sorted(plans, key=lambda item: int(item["gap_index"])):
        clues.extend(_gap_clues(plan, tracks_by_id))
    catalog = {
        "schema_version": CLUE_CATALOG_SCHEMA_VERSION,
        "evidence_policy": "visible_frames_and_gap_boundaries_only",
        "clues": clues,
    }
    catalog["clue_digest"] = _canonical_digest(catalog)
    return validate_clue_catalog(catalog)


def validate_clue_catalog(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != CLUE_CATALOG_SCHEMA_VERSION:
        raise ClueCatalogValidationError("Clue catalog schema version is invalid")
    clues = value.get("clues")
    if not isinstance(clues, list):
        raise ClueCatalogValidationError("Clue catalog clues must be a list")
    validated = [_validate_clue(item) for item in clues]
    identifiers = [item["id"] for item in validated]
    if len(set(identifiers)) != len(identifiers):
        raise ClueCatalogValidationError("Clue identifiers must be unique")
    return {**value, "clues": validated}


def public_clues(catalog: dict) -> list[dict]:
    return [
        {
            "id": clue["id"],
            "scope": clue["scope"],
            "category": clue["category"],
            "statement": clue["statement"],
            "confidence": clue["confidence"],
        }
        for clue in catalog["clues"]
    ]


def _scene_clues(scene_report: dict) -> list[dict]:
    tracks = [item for item in scene_report.get("tracks", []) if isinstance(item, dict)]
    class_counts = Counter(str(item.get("class_name", "unknown")) for item in tracks)
    camera_report = scene_report.get("camera_motion_report", {})
    camera_mode = str(camera_report.get("classification", camera_report.get("mode", "unknown")))
    return [
        _clue(
            "scene_tracks",
            "scene",
            "entity_inventory",
            f"{len(tracks)} visible-evidence tracks were measured ({_class_summary(class_counts)}).",
            _average_track_confidence(tracks),
            [f"track:{item.get('id')}:visible_observations" for item in tracks],
        ),
        _clue(
            "scene_camera",
            "scene",
            "camera",
            f"Camera motion is classified as {camera_mode}.",
            _camera_confidence(camera_report),
            ["scene:camera_motion_report"],
        ),
        _clue(
            "scene_policy",
            "scene",
            "evidence_policy",
            "All clues were derived from visible frames or immediate visible gap boundaries.",
            1.0,
            ["scene:visible_ranges"],
        ),
    ]


def _gap_clues(plan: dict, tracks_by_id: dict[str, dict]) -> list[dict]:
    gap_index = int(plan["gap_index"])
    clues = [_calibration_clue(plan)]
    for entity in plan.get("entities", []):
        track = tracks_by_id.get(str(entity.get("id")), {})
        clues.extend(_entity_clues(gap_index, entity, track))
    if not plan.get("entities"):
        clues.append(_clue(
            f"gap_{gap_index:02d}_unknown",
            f"gap:{gap_index}",
            "unknown",
            "No renderable entity has reliable boundary support for this gap.",
            0.2,
            [f"gap:{gap_index}:camera_calibration"],
        ))
    return clues


def _calibration_clue(plan: dict) -> dict:
    gap_index = int(plan["gap_index"])
    confidence = float(plan["camera"]["calibration_confidence"])
    label = "strong" if confidence >= 0.75 else "limited" if confidence < 0.5 else "moderate"
    return _clue(
        f"gap_{gap_index:02d}_calibration",
        f"gap:{gap_index}",
        "calibration",
        f"Ground-plane calibration support is {label}.",
        confidence,
        [f"gap:{gap_index}:camera_calibration"],
    )


def _entity_clues(gap_index: int, entity: dict, track: dict) -> list[dict]:
    identifier = str(entity["id"])
    boundary = entity["boundary_evidence"]
    lifecycle = str(entity["lifecycle"])
    direction = str(track.get("direction", "unknown"))
    references = _boundary_references(identifier, boundary)
    confidence = float(entity["confidence"])
    clues = [_clue(
        f"gap_{gap_index:02d}_{identifier}_motion",
        f"gap:{gap_index}:entity:{identifier}",
        "motion",
        f"{identifier} is {lifecycle}; visible motion is {direction} at "
        f"{float(entity['animation']['speed_meters_per_second']):.2f} m/s.",
        confidence,
        references,
    )]
    disagreement = float(boundary.get("heading_disagreement_degrees", 0.0))
    if disagreement >= 35.0:
        clues.append(_clue(
            f"gap_{gap_index:02d}_{identifier}_heading_conflict",
            f"gap:{gap_index}:entity:{identifier}",
            "conflict",
            f"Pre-gap and post-gap headings disagree by {disagreement:.1f} degrees.",
            max(0.1, confidence * 0.65),
            references,
        ))
    return clues


def _boundary_references(identifier: str, boundary: dict) -> list[str]:
    references = []
    if boundary.get("before_frame") is not None:
        references.append(f"track:{identifier}:pre_boundary")
    if boundary.get("after_frame") is not None:
        references.append(f"track:{identifier}:post_boundary")
    return references


def _clue(
    identifier: str,
    scope: str,
    category: str,
    statement: str,
    confidence: float,
    evidence_references: list[str],
) -> dict:
    return {
        "id": identifier,
        "scope": scope,
        "category": category,
        "statement": statement,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "evidence_references": evidence_references,
    }


def _validate_clue(value: object) -> dict:
    if not isinstance(value, dict):
        raise ClueCatalogValidationError("Each clue must be an object")
    required_text = ("id", "scope", "category", "statement")
    if any(not isinstance(value.get(field), str) or not value[field].strip() for field in required_text):
        raise ClueCatalogValidationError("Clue text fields are invalid")
    if len(value["statement"]) > MAXIMUM_CLUE_TEXT_LENGTH:
        raise ClueCatalogValidationError("Clue statement is too long")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ClueCatalogValidationError("Clue confidence must be numeric")
    references = value.get("evidence_references")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ClueCatalogValidationError("Clue evidence references are invalid")
    return {**value, "confidence": round(float(confidence), 4)}


def _class_summary(class_counts: Counter) -> str:
    if not class_counts:
        return "none retained"
    return ", ".join(f"{count} {name}" for name, count in sorted(class_counts.items()))


def _average_track_confidence(tracks: list[dict]) -> float:
    values = [float(track.get("avg_confidence", 0.0)) for track in tracks]
    return sum(values) / len(values) if values else 0.0


def _camera_confidence(camera_report: object) -> float:
    if not isinstance(camera_report, dict):
        return 0.0
    values = [
        float(camera_report.get("static_feature_inlier_score", 0.0)),
        float(camera_report.get("camera_motion_fit_score", 0.0)),
    ]
    return sum(values) / len(values)


def _canonical_digest(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
