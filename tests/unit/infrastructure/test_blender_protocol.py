import json
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.blender_protocol import (
    EVENT_ERROR,
    EVENT_PROGRESS,
    EVENT_READY,
    EVENT_RESULT,
    PROTOCOL_MARKER,
    PROTOCOL_VERSION,
    ProtocolError,
    ServiceCommand,
    ServiceEvent,
    decode_command,
    decode_event,
    encode_command,
    encode_event,
    is_protocol_line,
    open_job_command,
    render_frames_command,
)


class TestCommandRoundTrip:
    def test_round_trip_preserves_identity_and_payload(self):
        command = ServiceCommand("req-7", "render_frames", {"gap_index": 3})
        decoded = decode_command(encode_command(command))
        assert decoded == command

    def test_encoded_command_is_exactly_one_line(self):
        line = encode_command(ServiceCommand("req-1", "ping", {"note": "multi\nline"}))
        assert line.endswith("\n")
        assert line.count("\n") == 1

    def test_unsupported_command_is_rejected_before_transmission(self):
        with pytest.raises(ProtocolError, match="Unsupported command"):
            encode_command(ServiceCommand("req-1", "delete_everything"))

    def test_empty_request_id_is_rejected(self):
        with pytest.raises(ProtocolError, match="non-empty request id"):
            encode_command(ServiceCommand("", "ping"))


class TestCommandDecodingIsDefensive:
    @pytest.mark.parametrize(
        "line",
        [
            "not json at all",
            "[]",
            '"a string"',
            "123",
        ],
    )
    def test_non_object_payloads_are_rejected(self, line):
        with pytest.raises(ProtocolError):
            decode_command(line)

    def test_version_mismatch_is_rejected(self):
        line = json.dumps({"protocol_version": 99, "id": "a", "cmd": "ping", "payload": {}})
        with pytest.raises(ProtocolError, match="Unsupported protocol version"):
            decode_command(line)

    def test_unknown_command_is_rejected(self):
        line = json.dumps(
            {"protocol_version": PROTOCOL_VERSION, "id": "a", "cmd": "rm_rf", "payload": {}}
        )
        with pytest.raises(ProtocolError, match="Unsupported command"):
            decode_command(line)

    def test_non_object_payload_field_is_rejected(self):
        line = json.dumps(
            {"protocol_version": PROTOCOL_VERSION, "id": "a", "cmd": "ping", "payload": [1, 2]}
        )
        with pytest.raises(ProtocolError, match="payload must be an object"):
            decode_command(line)

    def test_missing_id_is_rejected(self):
        line = json.dumps({"protocol_version": PROTOCOL_VERSION, "cmd": "ping", "payload": {}})
        with pytest.raises(ProtocolError, match="non-empty string id"):
            decode_command(line)

    def test_absurdly_long_line_is_rejected_without_parsing(self):
        with pytest.raises(ProtocolError, match="maximum permitted length"):
            decode_command("x" * 2_000_000)


class TestEventFraming:
    def test_round_trip_preserves_event(self):
        event = ServiceEvent(EVENT_RESULT, "req-2", {"frames": 12})
        decoded = decode_event(encode_event(event))
        assert decoded == event

    def test_encoded_event_carries_the_marker(self):
        line = encode_event(ServiceEvent(EVENT_READY, None, {}))
        assert line.startswith(PROTOCOL_MARKER)
        assert is_protocol_line(line)

    def test_ordinary_blender_output_decodes_to_none(self):
        # Cycles prints lines like this constantly; they must not raise.
        assert decode_event("Fra:1 Mem:64.00M | Remaining:00:02.13") is None
        assert decode_event("") is None
        assert decode_event("Error: something happened in a shader") is None

    def test_marker_inside_ordinary_text_is_not_treated_as_protocol(self):
        assert decode_event("rendering with @FOR3D@-style markers enabled") is None

    def test_malformed_protocol_line_raises_rather_than_being_ignored(self):
        # A corrupted protocol line is a real fault; silently dropping it would
        # strand the request that was waiting on it.
        with pytest.raises(ProtocolError):
            decode_event(f"{PROTOCOL_MARKER} {{not json")

    def test_unknown_event_kind_is_rejected(self):
        line = f"{PROTOCOL_MARKER} " + json.dumps(
            {"protocol_version": PROTOCOL_VERSION, "event": "explode", "id": "a", "payload": {}}
        )
        with pytest.raises(ProtocolError, match="Unsupported event"):
            decode_event(line)

    def test_leading_whitespace_before_marker_is_tolerated(self):
        line = "   " + encode_event(ServiceEvent(EVENT_PROGRESS, "req-3", {"frame": 1}))
        decoded = decode_event(line)
        assert decoded is not None
        assert decoded.payload["frame"] == 1


class TestTerminalEventClassification:
    def test_result_and_error_are_terminal(self):
        assert ServiceEvent(EVENT_RESULT, "a").is_terminal
        assert ServiceEvent(EVENT_ERROR, "a").is_terminal

    def test_progress_and_ready_are_not_terminal(self):
        assert not ServiceEvent(EVENT_PROGRESS, "a").is_terminal
        assert not ServiceEvent(EVENT_READY, None).is_terminal

    def test_error_message_has_a_usable_default(self):
        assert ServiceEvent(EVENT_ERROR, "a").error_message
        assert ServiceEvent(EVENT_ERROR, "a", {"message": "boom"}).error_message == "boom"


class TestCommandConstructors:
    def test_open_job_references_a_path_rather_than_inlining_data(self):
        command = open_job_command("req-1", "/jobs/x/manifest.json")
        assert command.payload == {"job_manifest_path": "/jobs/x/manifest.json"}

    def test_render_frames_copies_the_frame_list(self):
        frames = [1, 2, 3]
        command = render_frames_command("req-1", 0, frames, "/out")
        frames.append(4)
        assert command.payload["frame_indexes"] == [1, 2, 3]
