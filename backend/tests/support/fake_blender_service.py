"""A stand-in for headless Blender that speaks the §6.2 protocol.

Lets the host client be tested over a real subprocess — real pipes, real buffering,
real EOF-on-crash — without needing Blender installed. It deliberately interleaves
noisy non-protocol output the way Cycles does, so log teeing is exercised too.

Behaviour is steered through command payloads:
  open_job manifest path containing "crash-now"  -> exit immediately
  open_job manifest path containing "go-silent"  -> stop responding, stay alive
"""

import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.blender_protocol import (  # noqa: E402
    COMMAND_CLOSE_JOB,
    COMMAND_HELLO,
    COMMAND_OPEN_JOB,
    COMMAND_PING,
    COMMAND_PREPARE_GAP,
    COMMAND_RENDER_FRAMES,
    COMMAND_RESET_GAP,
    COMMAND_SHUTDOWN,
    EVENT_ERROR,
    EVENT_PROGRESS,
    EVENT_READY,
    EVENT_RESULT,
    ProtocolError,
    ServiceEvent,
    decode_command,
    encode_event,
)

CRASH_TOKEN = "crash-now"
SILENCE_TOKEN = "go-silent"
SILENT_SLEEP_SECONDS = 60.0
CRASH_EXIT_CODE = 3


def emit(event: ServiceEvent) -> None:
    sys.stdout.write(encode_event(event))
    sys.stdout.flush()


def noise(text: str) -> None:
    """Non-protocol chatter, as Cycles produces during a render."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def handle_render_frames(request_id: str, payload: dict) -> None:
    frame_indexes = payload.get("frame_indexes", [])
    for position, frame_index in enumerate(frame_indexes, start=1):
        noise(f"Fra:{frame_index} Mem:64.00M | Remaining:00:00.01")
        emit(ServiceEvent(EVENT_PROGRESS, request_id, {
            "frame_index": frame_index,
            "completed": position,
            "total": len(frame_indexes),
        }))
    emit(ServiceEvent(EVENT_RESULT, request_id, {
        "gap_index": payload.get("gap_index"),
        "rendered_frame_indexes": list(frame_indexes),
    }))


def handle_open_job(request_id: str, payload: dict) -> None:
    manifest_path = str(payload.get("job_manifest_path", ""))
    if CRASH_TOKEN in manifest_path:
        sys.stdout.flush()
        raise SystemExit(CRASH_EXIT_CODE)
    if SILENCE_TOKEN in manifest_path:
        time.sleep(SILENT_SLEEP_SECONDS)
        return
    emit(ServiceEvent(EVENT_RESULT, request_id, {"shell_built": True, "assets": 3}))


def handle_prepare_gap(request_id: str, payload: dict) -> None:
    """A negative gap index stands in for any command Blender rejects."""
    gap_index = payload.get("gap_index")
    if not isinstance(gap_index, int) or gap_index < 0:
        emit(ServiceEvent(EVENT_ERROR, request_id, {
            "message": f"unknown gap index {gap_index!r}",
            "kind": "invalid_request",
        }))
        return
    emit(ServiceEvent(EVENT_RESULT, request_id, {"ok": True, "gap_index": gap_index}))


def dispatch(command) -> bool:
    """Returns False when the service should stop."""
    if command.command == COMMAND_SHUTDOWN:
        emit(ServiceEvent(EVENT_RESULT, command.request_id, {"stopped": True}))
        return False
    if command.command == COMMAND_HELLO:
        emit(ServiceEvent(EVENT_RESULT, command.request_id, {
            "blender_version": "fake-4.5",
            "engines": ["CYCLES", "BLENDER_EEVEE_NEXT"],
        }))
        return True
    if command.command == COMMAND_OPEN_JOB:
        handle_open_job(command.request_id, command.payload)
        return True
    if command.command == COMMAND_RENDER_FRAMES:
        handle_render_frames(command.request_id, command.payload)
        return True
    if command.command == COMMAND_PREPARE_GAP:
        handle_prepare_gap(command.request_id, command.payload)
        return True
    if command.command in (COMMAND_PING, COMMAND_RESET_GAP, COMMAND_CLOSE_JOB):
        emit(ServiceEvent(EVENT_RESULT, command.request_id, {"ok": True}))
        return True
    emit(ServiceEvent(EVENT_ERROR, command.request_id, {"message": "unhandled command"}))
    return True


def main() -> None:
    noise("Blender 4.5.10 LTS (fake)")
    noise("Read prefs: some/path/userpref.blend")
    emit(ServiceEvent(EVENT_READY, None, {"protocol_version": 1}))
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            command = decode_command(line)
        except ProtocolError as error:
            emit(ServiceEvent(EVENT_ERROR, None, {"message": str(error)}))
            continue
        if not dispatch(command):
            break


if __name__ == "__main__":
    main()
