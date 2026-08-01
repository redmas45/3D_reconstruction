"""Persistent in-Blender service loop (Implementation_plan.md §6.2).

One instance serves a whole job: it builds the warm shell once, then answers
`prepare_gap` / `render_frames` / `reset_gap` until told to shut down. This is what
removes the per-gap Blender startup, shader compilation, and OptiX kernel compilation
that §6.1 measures at roughly 7-8 minutes of waste per job.

Launched by the host as:

    blender --background --factory-startup --python blender/service.py -- \
            --job-root <path> --protocol-version 1

Protocol lines go to stdout behind the `@FOR3D@` marker; everything else Blender
prints is ordinary log noise the host tees to a file.

Import boundary (§3): this loads `blender_protocol` by explicit path because the wire
format must have exactly one definition or host and service silently drift apart. That
module is stdlib-only and contains no business logic, which `tests/unit/test_blender_
service_boundary.py` enforces. Nothing else from `backend/` is importable here.
"""

import sys
import time
from pathlib import Path

import bpy


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
PROTOCOL_MODULE_DIRECTORY = PROJECT_ROOT / "backend" / "infrastructure"

for _path in (str(SCRIPT_ROOT), str(PROTOCOL_MODULE_DIRECTORY)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import blender_protocol as protocol  # noqa: E402
import warm_shell  # noqa: E402
from warm_shell import ShellStateError  # noqa: E402


RENDER_FILE_FORMAT = "PNG"
RENDER_COLOR_MODE = "RGBA"
TEMPORARY_SUFFIX = ".rendering"

# Cropping to the actors' projected box is the largest single saving measured in M0.
# The rectangle arrives from the host so the crop the compositor assumes and the crop
# Blender renders are the same number (Implementation_plan.md §5.2).
FULL_FRAME_REGION = (0.0, 0.0, 1.0, 1.0)

# Frame naming and PNG validation come from the shared contract so host-side resume
# and service-side writing can never disagree about which files exist.
MINIMUM_VALID_PNG_BYTES = protocol.MINIMUM_VALID_PNG_BYTES
PNG_MAGIC = protocol.PNG_MAGIC


class ServiceState:
    """Everything the loop carries between commands."""

    def __init__(self, job_root: Path) -> None:
        self.job_root = job_root
        self.job_open = False
        self.prepared_gap_index: int | None = None
        self.started_at = time.monotonic()
        self.rendered_frame_count = 0


def emit(event: protocol.ServiceEvent) -> None:
    sys.stdout.write(protocol.encode_event(event))
    sys.stdout.flush()


def emit_result(request_id: str, payload: dict) -> None:
    emit(protocol.ServiceEvent(protocol.EVENT_RESULT, request_id, payload))


def emit_error(request_id: str | None, message: str, kind: str = "command_failed") -> None:
    emit(protocol.ServiceEvent(protocol.EVENT_ERROR, request_id, {"message": message, "kind": kind}))


def emit_progress(request_id: str, payload: dict) -> None:
    emit(protocol.ServiceEvent(protocol.EVENT_PROGRESS, request_id, payload))


# --------------------------------------------------------------------------
# Capability reporting
# --------------------------------------------------------------------------

def _selectable_engines() -> list[str]:
    """Verified by assignment: Cycles is an add-on and is absent from the RNA enum."""
    render_settings = bpy.context.scene.render
    original = render_settings.engine
    verified = []
    for candidate in ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            render_settings.engine = candidate
        except (TypeError, ValueError):
            continue
        verified.append(candidate)
    render_settings.engine = original
    return verified


def _cycles_devices() -> list[str]:
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return []
    try:
        addon.preferences.get_devices()
    except (AttributeError, RuntimeError):
        return []
    return [device.name for device in addon.preferences.devices if device.use]


def handle_hello(state: ServiceState, request_id: str, _payload: dict) -> None:
    emit_result(request_id, {
        "blender_version": bpy.app.version_string,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "engines": _selectable_engines(),
        "cycles_devices": _cycles_devices(),
        "job_root": str(state.job_root),
        "uptime_seconds": round(time.monotonic() - state.started_at, 3),
    })


# --------------------------------------------------------------------------
# Job and gap lifecycle
# --------------------------------------------------------------------------

def handle_open_job(state: ServiceState, request_id: str, payload: dict) -> None:
    manifest_path = Path(str(payload["job_manifest_path"]))
    manifest = warm_shell.load_json_contract(manifest_path)
    summary = warm_shell.build_shell(manifest)
    _apply_render_settings(manifest.get("render", {}))
    state.job_open = True
    state.prepared_gap_index = None
    emit_result(request_id, summary)


def _apply_render_settings(render_contract: dict) -> None:
    """Output format and sampling. The engine is already chosen by `build_shell`, which
    needs it before it can decide what the shell contains."""
    scene = bpy.context.scene
    scene.render.image_settings.file_format = RENDER_FILE_FORMAT
    scene.render.image_settings.color_mode = RENDER_COLOR_MODE
    scene.render.film_transparent = True
    samples = render_contract.get("samples")
    if samples is None:
        return
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = bool(render_contract.get("denoise", True))
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(samples)


def handle_prepare_gap(state: ServiceState, request_id: str, payload: dict) -> None:
    if not state.job_open:
        emit_error(request_id, "open_job must succeed before prepare_gap", "invalid_state")
        return
    gap_index = payload.get("gap_index")
    if not isinstance(gap_index, int) or gap_index < 0:
        emit_error(request_id, f"unknown gap index {gap_index!r}", "invalid_request")
        return
    specification = warm_shell.load_json_contract(Path(str(payload["storyboard_path"])))
    warm_shell.clear_actors()
    summary = warm_shell.prepare_gap({**specification, "gap_index": gap_index})
    state.prepared_gap_index = gap_index
    emit_result(request_id, summary)


def handle_reset_gap(state: ServiceState, request_id: str, _payload: dict) -> None:
    summary = warm_shell.clear_actors()
    state.prepared_gap_index = None
    emit_result(request_id, summary)


def handle_close_job(state: ServiceState, request_id: str, _payload: dict) -> None:
    warm_shell.clear_actors()
    state.job_open = False
    state.prepared_gap_index = None
    emit_result(request_id, {
        "closed": True,
        "rendered_frame_count": state.rendered_frame_count,
        "uptime_seconds": round(time.monotonic() - state.started_at, 3),
        "census": warm_shell.datablock_census(),
    })


def handle_ping(state: ServiceState, request_id: str, _payload: dict) -> None:
    emit_result(request_id, {
        "ok": True,
        "uptime_seconds": round(time.monotonic() - state.started_at, 3),
    })


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _valid_png(path: Path) -> bool:
    try:
        with path.open("rb") as image_file:
            return path.stat().st_size > MINIMUM_VALID_PNG_BYTES and image_file.read(8) == PNG_MAGIC
    except OSError:
        return False


def apply_render_region(scene: bpy.types.Scene, region: tuple | list | None) -> tuple:
    """Restrict *shading* to a normalized rectangle, keeping the output frame-sized.

    `use_border` is what saves the time: Blender shades only the bordered pixels and
    leaves the rest transparent. `use_crop_to_border` would additionally shrink the
    output image, and that is deliberately left off.

    The reason is a rounding mismatch that is invisible until it corrupts output.
    Blender converts the normalized border to a pixel rectangle with its own rounding,
    which does not always agree with the host's to the pixel. A cropped layer would then
    be a slightly different size than the compositor expects, and the choice would be
    between failing the render or silently shifting the actor by a pixel. Emitting a
    full-size frame removes the conversion entirely: the layer always aligns with the
    plate because they are the same size. The saving was never in the crop.
    """
    bounds = tuple(float(value) for value in region) if region else FULL_FRAME_REGION
    scene.render.use_crop_to_border = False
    if bounds == FULL_FRAME_REGION:
        scene.render.use_border = False
        return FULL_FRAME_REGION
    scene.render.use_border = True
    scene.render.border_min_x, scene.render.border_min_y = bounds[0], bounds[1]
    scene.render.border_max_x, scene.render.border_max_y = bounds[2], bounds[3]
    return bounds


def render_single_frame(
    frame_index: int, output_directory: Path, region: tuple | list | None = None,
) -> dict:
    """Render one frame atomically so an interrupted job never leaves a partial PNG."""
    scene = bpy.context.scene
    applied_region = apply_render_region(scene, region)
    final_path = output_directory / protocol.frame_filename(frame_index)
    temporary_path = final_path.with_suffix(TEMPORARY_SUFFIX + ".png")
    temporary_path.unlink(missing_ok=True)
    scene.frame_set(frame_index)
    scene.render.filepath = str(temporary_path)
    started_at = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - started_at
    if not _valid_png(temporary_path):
        raise RuntimeError(f"Blender produced no valid PNG for frame {frame_index}")
    temporary_path.replace(final_path)
    return {
        "frame_index": frame_index,
        "path": str(final_path),
        "bytes": final_path.stat().st_size,
        "seconds": round(elapsed, 4),
        "region": list(applied_region),
    }


def handle_render_frames(state: ServiceState, request_id: str, payload: dict) -> None:
    if state.prepared_gap_index is None:
        emit_error(request_id, "prepare_gap must succeed before render_frames", "invalid_state")
        return
    frame_indexes = [int(value) for value in payload.get("frame_indexes", [])]
    regions = payload.get("regions")
    if regions is not None and len(regions) != len(frame_indexes):
        emit_error(request_id, "each frame requires exactly one region", "invalid_request")
        return
    output_directory = Path(str(payload["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    rendered = _render_frame_sequence(request_id, frame_indexes, output_directory, regions)
    state.rendered_frame_count += len(rendered)
    emit_result(request_id, {
        "gap_index": state.prepared_gap_index,
        "rendered_frame_indexes": [item["frame_index"] for item in rendered],
        "frames": rendered,
        "census": warm_shell.datablock_census(),
    })


def _render_frame_sequence(
    request_id: str,
    frame_indexes: list[int],
    output_directory: Path,
    regions: list | None = None,
) -> list[dict]:
    """Emit progress per frame — the host treats each marker as a heartbeat (§6.7)."""
    rendered: list[dict] = []
    total = len(frame_indexes)
    for position, frame_index in enumerate(frame_indexes, start=1):
        region = regions[position - 1] if regions else None
        record = render_single_frame(frame_index, output_directory, region)
        rendered.append(record)
        emit_progress(request_id, {
            "frame_index": frame_index,
            "completed": position,
            "total": total,
            "seconds": record["seconds"],
        })
    return rendered


# --------------------------------------------------------------------------
# Dispatch loop
# --------------------------------------------------------------------------

HANDLERS = {
    protocol.COMMAND_HELLO: handle_hello,
    protocol.COMMAND_OPEN_JOB: handle_open_job,
    protocol.COMMAND_PREPARE_GAP: handle_prepare_gap,
    protocol.COMMAND_RENDER_FRAMES: handle_render_frames,
    protocol.COMMAND_RESET_GAP: handle_reset_gap,
    protocol.COMMAND_CLOSE_JOB: handle_close_job,
    protocol.COMMAND_PING: handle_ping,
}


def dispatch(state: ServiceState, command: protocol.ServiceCommand) -> bool:
    """Run one command. Returns False when the loop should stop."""
    if command.command == protocol.COMMAND_SHUTDOWN:
        emit_result(command.request_id, {"stopped": True})
        return False
    handler = HANDLERS.get(command.command)
    if handler is None:
        emit_error(command.request_id, f"unhandled command {command.command}", "invalid_request")
        return True
    try:
        handler(state, command.request_id, command.payload)
    except ShellStateError as error:
        emit_error(command.request_id, str(error), "shell_state")
    except (KeyError, TypeError, ValueError) as error:
        emit_error(command.request_id, f"invalid contract: {error}", "invalid_request")
    except (OSError, RuntimeError) as error:
        emit_error(command.request_id, str(error), "render_failed")
    return True


def parse_arguments(argv: list[str]) -> dict:
    arguments = argv[argv.index("--") + 1:] if "--" in argv else []
    parsed = {"job_root": ".", "protocol_version": str(protocol.PROTOCOL_VERSION)}
    for index in range(0, len(arguments) - 1, 2):
        key = arguments[index].lstrip("-").replace("-", "_")
        parsed[key] = arguments[index + 1]
    return parsed


def serve(state: ServiceState) -> None:
    emit(protocol.ServiceEvent(
        protocol.EVENT_READY, None, {"protocol_version": protocol.PROTOCOL_VERSION},
    ))
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            command = protocol.decode_command(line)
        except protocol.ProtocolError as error:
            emit_error(None, str(error), "protocol_error")
            continue
        if not dispatch(state, command):
            break


def main() -> None:
    arguments = parse_arguments(sys.argv)
    if arguments["protocol_version"] != str(protocol.PROTOCOL_VERSION):
        emit_error(None, "protocol version mismatch", "protocol_error")
        raise SystemExit(2)
    serve(ServiceState(Path(arguments["job_root"])))


if __name__ == "__main__":
    main()
