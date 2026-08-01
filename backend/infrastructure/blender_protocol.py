"""Wire format for the persistent Blender service (Implementation_plan.md §6.2).

Blender's stdout carries render statistics, add-on chatter, and driver warnings, so
protocol lines are marked with a magic prefix and everything else is treated as log
output. This module is the single definition of that framing, shared by the host
client and the in-Blender service loop.

Everything here is pure: no I/O, no subprocess, no `bpy`. That keeps the wire format
testable without launching Blender, and lets the Blender side import it by path
without dragging in project business logic (the §3 boundary).

Blender is a separate process whose output is parsed as untrusted input per
`rules.md` §9 — every field is validated before it reaches a caller.
"""

import json
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = 1
PROTOCOL_MARKER = "@FOR3D@"

COMMAND_HELLO = "hello"
COMMAND_OPEN_JOB = "open_job"
COMMAND_PREPARE_GAP = "prepare_gap"
COMMAND_RENDER_FRAMES = "render_frames"
COMMAND_RESET_GAP = "reset_gap"
COMMAND_CLOSE_JOB = "close_job"
COMMAND_PING = "ping"
COMMAND_SHUTDOWN = "shutdown"

SUPPORTED_COMMANDS = frozenset({
    COMMAND_HELLO,
    COMMAND_OPEN_JOB,
    COMMAND_PREPARE_GAP,
    COMMAND_RENDER_FRAMES,
    COMMAND_RESET_GAP,
    COMMAND_CLOSE_JOB,
    COMMAND_PING,
    COMMAND_SHUTDOWN,
})

EVENT_READY = "ready"
EVENT_PROGRESS = "progress"
EVENT_RESULT = "result"
EVENT_ERROR = "error"

SUPPORTED_EVENTS = frozenset({EVENT_READY, EVENT_PROGRESS, EVENT_RESULT, EVENT_ERROR})

# Events that conclude a request. Progress may arrive many times before one of these.
TERMINAL_EVENTS = frozenset({EVENT_RESULT, EVENT_ERROR})

MAXIMUM_LINE_CHARACTERS = 1_000_000

# Rendered-frame naming is part of the contract, not an implementation detail: the
# service writes these files and the host decides from their presence what still needs
# rendering (§6.7). Two definitions would silently break resume, so there is one.
FRAME_FILENAME_TEMPLATE = "frame_{index:06d}.png"
MINIMUM_VALID_PNG_BYTES = 64
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def frame_filename(frame_index: int) -> str:
    return FRAME_FILENAME_TEMPLATE.format(index=int(frame_index))


class ProtocolError(ValueError):
    """A protocol line was present but malformed."""


@dataclass(frozen=True)
class ServiceCommand:
    request_id: str
    command: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServiceEvent:
    event: str
    request_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.event in TERMINAL_EVENTS

    @property
    def error_message(self) -> str:
        return str(self.payload.get("message", "Blender reported an unspecified error"))


def encode_command(command: ServiceCommand) -> str:
    """Serialize a command as one newline-terminated line for Blender's stdin."""
    if command.command not in SUPPORTED_COMMANDS:
        raise ProtocolError(f"Unsupported command: {command.command}")
    if not command.request_id:
        raise ProtocolError("Command requires a non-empty request id")
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "id": command.request_id,
        "cmd": command.command,
        "payload": command.payload,
    }
    return json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n"


def decode_command(line: str) -> ServiceCommand:
    """Parse one stdin line inside Blender. Raises ProtocolError on anything invalid."""
    body = _decode_json_object(line)
    version = body.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol version: {version!r}")
    command = body.get("cmd")
    if command not in SUPPORTED_COMMANDS:
        raise ProtocolError(f"Unsupported command: {command!r}")
    request_id = body.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("Command requires a non-empty string id")
    payload = body.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError("Command payload must be an object")
    return ServiceCommand(request_id=request_id, command=command, payload=payload)


def encode_event(event: ServiceEvent) -> str:
    """Serialize an event as one marked, newline-terminated line for Blender's stdout."""
    if event.event not in SUPPORTED_EVENTS:
        raise ProtocolError(f"Unsupported event: {event.event}")
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "event": event.event,
        "id": event.request_id,
        "payload": event.payload,
    }
    return f"{PROTOCOL_MARKER} " + json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n"


def is_protocol_line(line: str) -> bool:
    return line.lstrip().startswith(PROTOCOL_MARKER)


def decode_event(line: str) -> ServiceEvent | None:
    """Parse one stdout line from Blender.

    Returns None for ordinary log output so callers can tee it to a file without
    having to distinguish noise from protocol themselves.
    """
    if not is_protocol_line(line):
        return None
    body = _decode_json_object(line.lstrip()[len(PROTOCOL_MARKER):])
    version = body.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol version: {version!r}")
    event = body.get("event")
    if event not in SUPPORTED_EVENTS:
        raise ProtocolError(f"Unsupported event: {event!r}")
    request_id = body.get("id")
    if request_id is not None and not isinstance(request_id, str):
        raise ProtocolError("Event id must be a string when present")
    payload = body.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError("Event payload must be an object")
    return ServiceEvent(event=event, request_id=request_id, payload=payload)


def _decode_json_object(text: str) -> dict[str, Any]:
    if len(text) > MAXIMUM_LINE_CHARACTERS:
        raise ProtocolError("Protocol line exceeds the maximum permitted length")
    try:
        body = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Protocol line is not valid JSON: {error.msg}") from error
    if not isinstance(body, dict):
        raise ProtocolError("Protocol line must encode a JSON object")
    return body


# --------------------------------------------------------------------------
# Command constructors — keep payload shapes in one place
# --------------------------------------------------------------------------

def hello_command(request_id: str) -> ServiceCommand:
    return ServiceCommand(request_id, COMMAND_HELLO)


def ping_command(request_id: str) -> ServiceCommand:
    return ServiceCommand(request_id, COMMAND_PING)


def open_job_command(request_id: str, job_manifest_path: str) -> ServiceCommand:
    """Payloads reference files on disk rather than inlining data (§6.2)."""
    return ServiceCommand(request_id, COMMAND_OPEN_JOB, {"job_manifest_path": job_manifest_path})


def prepare_gap_command(request_id: str, gap_index: int, storyboard_path: str) -> ServiceCommand:
    return ServiceCommand(
        request_id,
        COMMAND_PREPARE_GAP,
        {"gap_index": gap_index, "storyboard_path": storyboard_path},
    )


def render_frames_command(
    request_id: str,
    gap_index: int,
    frame_indexes: list[int],
    output_directory: str,
    regions: list[list[float]] | None = None,
    passes: list[str] | None = None,
) -> ServiceCommand:
    """Render a set of sparse frames.

    `regions` is one normalized crop rectangle per frame, computed host-side by the
    tested projection in `domain.render_region` — Blender applies it but never derives
    it, so the crop the compositor assumes and the crop Blender renders cannot diverge.
    """
    payload: dict[str, Any] = {
        "gap_index": gap_index,
        "frame_indexes": list(frame_indexes),
        "output_directory": output_directory,
    }
    if regions is not None:
        if len(regions) != len(frame_indexes):
            raise ProtocolError("Each rendered frame requires exactly one render region")
        payload["regions"] = [list(region) for region in regions]
    if passes is not None:
        payload["passes"] = list(passes)
    return ServiceCommand(request_id, COMMAND_RENDER_FRAMES, payload)


def reset_gap_command(request_id: str) -> ServiceCommand:
    return ServiceCommand(request_id, COMMAND_RESET_GAP)


def close_job_command(request_id: str) -> ServiceCommand:
    return ServiceCommand(request_id, COMMAND_CLOSE_JOB)


def shutdown_command(request_id: str) -> ServiceCommand:
    return ServiceCommand(request_id, COMMAND_SHUTDOWN)
