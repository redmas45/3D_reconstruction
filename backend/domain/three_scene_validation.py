"""Validates the Three.js scene manifest before it is served to the browser.

Two jobs: enforce the structural contract the renderer relies on, and stand guard
over the evidence boundary. The renderer must be given nothing that could only be
known from hidden (inside-gap) footage, so this validator rejects any manifest whose
backplate frame falls inside a gap and refuses any manifest carrying a key from the
hidden-truth denylist. It is deliberately strict: a manifest that cannot be proven
evidence-safe does not get rendered.
"""


SCENE_MANIFEST_SCHEMA_VERSION = 1
RENDERER_NAME = "threejs"
SUPPORTED_FIDELITY_TIERS = frozenset({"supported", "plausible", "weak"})
MINIMUM_WAYPOINTS = 3
MAXIMUM_FIELD_OF_VIEW_DEGREES = 179.0
MINIMUM_FIELD_OF_VIEW_DEGREES = 1.0
# Keys that would only ever carry inside-gap ground truth. Their presence anywhere in
# the manifest is treated as a leak, not a warning.
FORBIDDEN_KEYS = frozenset({
    "truth", "hidden_truth", "ground_truth", "hidden_frame", "hidden_frames",
    "hidden_positions", "actual_positions", "future_frames",
})


class SceneManifestValidationError(ValueError):
    pass


def validate_three_scene_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise SceneManifestValidationError("Scene manifest must be an object")
    if manifest.get("schema_version") != SCENE_MANIFEST_SCHEMA_VERSION:
        raise SceneManifestValidationError("Scene manifest schema_version must be 1")
    if manifest.get("renderer") != RENDERER_NAME:
        raise SceneManifestValidationError("Scene manifest renderer must be 'threejs'")
    _reject_forbidden_keys(manifest)
    _validate_source(manifest.get("source"))
    _validate_camera(manifest.get("camera_default"), "camera_default")
    gaps = manifest.get("gaps")
    if not isinstance(gaps, list):
        raise SceneManifestValidationError("Scene manifest gaps must be a list")
    hidden_ranges = _hidden_ranges(gaps)
    for gap in gaps:
        _validate_gap(gap, hidden_ranges)


def _validate_source(source: object) -> None:
    if not isinstance(source, dict):
        raise SceneManifestValidationError("Scene manifest source must be an object")
    for field_name in ("width", "height", "frame_count"):
        value = source.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SceneManifestValidationError(f"Scene manifest source {field_name} must be a positive integer")
    fps = source.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise SceneManifestValidationError("Scene manifest source fps must be positive")
    observed = source.get("observed_fraction")
    if not isinstance(observed, (int, float)) or not 0.0 <= float(observed) <= 1.0:
        raise SceneManifestValidationError("Scene manifest observed_fraction must be within [0, 1]")


def _validate_camera(camera: object, label: str) -> None:
    if not isinstance(camera, dict):
        raise SceneManifestValidationError(f"Scene manifest {label} must be an object")
    if not _is_vector3(camera.get("position")) or not _is_vector3(camera.get("look_at")):
        raise SceneManifestValidationError(f"Scene manifest {label} position and look_at must be 3-vectors")
    field_of_view = camera.get("field_of_view_degrees")
    if not isinstance(field_of_view, (int, float)) or not (
        MINIMUM_FIELD_OF_VIEW_DEGREES <= float(field_of_view) <= MAXIMUM_FIELD_OF_VIEW_DEGREES
    ):
        raise SceneManifestValidationError(f"Scene manifest {label} field of view is out of range")
    confidence = camera.get("calibration_confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise SceneManifestValidationError(f"Scene manifest {label} calibration confidence is invalid")


def _validate_gap(gap: object, hidden_ranges: list[tuple[int, int]]) -> None:
    if not isinstance(gap, dict):
        raise SceneManifestValidationError("Each gap must be an object")
    start_frame = gap.get("start_frame")
    end_frame = gap.get("end_frame")
    if not _is_int(start_frame) or not _is_int(end_frame) or int(end_frame) < int(start_frame):
        raise SceneManifestValidationError("Gap frame bounds are invalid")
    _validate_camera(gap.get("camera"), f"gap {gap.get('gap_index')} camera")
    _validate_backplate(gap.get("environment"), hidden_ranges)
    entities = gap.get("entities")
    if not isinstance(entities, list):
        raise SceneManifestValidationError("Gap entities must be a list")
    for entity in entities:
        _validate_entity(entity)


def _validate_backplate(environment: object, hidden_ranges: list[tuple[int, int]]) -> None:
    """The backplate must be a visible frame. A frame inside any gap is hidden footage."""
    if not isinstance(environment, dict):
        raise SceneManifestValidationError("Gap environment must be an object")
    backplate_frame = environment.get("backplate_frame")
    if backplate_frame is None:
        return
    if not _is_int(backplate_frame):
        raise SceneManifestValidationError("Gap backplate_frame must be an integer or null")
    frame = int(backplate_frame)
    if any(start <= frame <= end for start, end in hidden_ranges):
        raise SceneManifestValidationError(
            "Gap backplate_frame falls inside a hidden interval; hidden footage must never be sampled"
        )


def _validate_entity(entity: object) -> None:
    if not isinstance(entity, dict):
        raise SceneManifestValidationError("Each entity must be an object")
    for field_name in ("track_id", "class_name"):
        if not isinstance(entity.get(field_name), str) or not entity[field_name]:
            raise SceneManifestValidationError(f"Entity {field_name} must be a non-empty string")
    confidence = entity.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise SceneManifestValidationError("Entity confidence must be within [0, 1]")
    if entity.get("visual_fidelity_tier") not in SUPPORTED_FIDELITY_TIERS:
        raise SceneManifestValidationError("Entity visual_fidelity_tier is unsupported")
    _validate_waypoints(entity.get("waypoints"))


def _validate_waypoints(waypoints: object) -> None:
    if not isinstance(waypoints, list) or len(waypoints) < MINIMUM_WAYPOINTS:
        raise SceneManifestValidationError("Entity must carry at least three waypoints")
    for waypoint in waypoints:
        if not isinstance(waypoint, dict) or not _is_vector3(waypoint.get("world")):
            raise SceneManifestValidationError("Each waypoint must carry a world 3-vector")
        fraction = waypoint.get("time_fraction")
        if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) <= 1.0:
            raise SceneManifestValidationError("Waypoint time_fraction must be within [0, 1]")


def _reject_forbidden_keys(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in FORBIDDEN_KEYS:
                raise SceneManifestValidationError(
                    f"Scene manifest carries a forbidden hidden-truth key: '{key}'"
                )
            _reject_forbidden_keys(value)
    elif isinstance(node, list):
        for item in node:
            _reject_forbidden_keys(item)


def _hidden_ranges(gaps: list) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for gap in gaps:
        if isinstance(gap, dict) and _is_int(gap.get("start_frame")) and _is_int(gap.get("end_frame")):
            ranges.append((int(gap["start_frame"]), int(gap["end_frame"])))
    return ranges


def _is_vector3(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 3
        and all(isinstance(component, (int, float)) and not isinstance(component, bool) for component in value[:3])
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
