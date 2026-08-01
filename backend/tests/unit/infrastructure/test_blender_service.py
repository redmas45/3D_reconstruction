import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.blender_protocol import PNG_MAGIC, ping_command
from infrastructure.blender_service import (
    BlenderCommandFailed,
    BlenderService,
    BlenderServiceCrashed,
    BlenderServiceStalled,
    build_service_command,
    frame_output_path,
    is_complete_frame,
    missing_frame_indexes,
)

FAKE_SERVICE_SCRIPT = PROJECT_ROOT / "backend" / "tests" / "support" / "fake_blender_service.py"
SHORT_STALL_SECONDS = 2.0


def _fake_process_factory():
    def factory():
        return subprocess.Popen(
            [sys.executable, str(FAKE_SERVICE_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    return factory


@pytest.fixture
def service(tmp_path):
    instance = BlenderService(
        _fake_process_factory(),
        log_path=tmp_path / "blender.log",
        stall_timeout_seconds=SHORT_STALL_SECONDS,
        startup_timeout_seconds=30.0,
    )
    yield instance
    instance.shutdown()


class TestServiceCommandConstruction:
    def test_command_uses_factory_startup_for_determinism(self):
        command = build_service_command(
            Path("/bin/blender"), Path("/p/blender/service.py"), Path("/jobs/x"),
        )
        assert "--background" in command
        assert "--factory-startup" in command

    def test_job_root_and_protocol_version_are_passed_after_the_separator(self):
        command = build_service_command(
            Path("/bin/blender"), Path("/p/blender/service.py"), Path("/jobs/x"),
        )
        arguments = command[command.index("--") + 1:]
        assert "--job-root" in arguments
        assert "--protocol-version" in arguments


class TestLifecycle:
    def test_start_returns_the_capability_report(self, service):
        capabilities = service.start()
        assert capabilities["blender_version"] == "fake-4.5"
        assert service.is_alive

    def test_shutdown_is_idempotent(self, service):
        service.start()
        service.shutdown()
        service.shutdown()
        assert not service.is_alive

    def test_restart_produces_a_working_process(self, service):
        service.start()
        capabilities = service.restart()
        assert capabilities["blender_version"] == "fake-4.5"
        assert service.request(ping_command("req-restart"))["ok"] is True

    def test_starting_twice_is_refused(self, service):
        service.start()
        with pytest.raises(RuntimeError, match="already running"):
            service.start()

    def test_request_before_start_fails_cleanly(self, service):
        with pytest.raises(BlenderServiceCrashed):
            service.request(ping_command("req-early"))


class TestRequestRouting:
    def test_many_sequential_requests_reuse_one_process(self, service):
        service.start()
        for index in range(20):
            assert service.request(ping_command(f"req-loop-{index}"))["ok"] is True
        assert service.is_alive

    def test_progress_events_are_delivered_before_the_result(self, service):
        service.start()
        seen = []
        result = service.render_frames(
            gap_index=2,
            frame_indexes=[1, 2, 3, 4],
            output_directory=Path("/out"),
            on_progress=seen.append,
        )
        assert [item["frame_index"] for item in seen] == [1, 2, 3, 4]
        assert result["rendered_frame_indexes"] == [1, 2, 3, 4]
        assert result["gap_index"] == 2

    def test_error_events_raise_rather_than_returning_a_payload(self, service):
        service.start()
        with pytest.raises(BlenderCommandFailed, match="unknown gap index"):
            service.prepare_gap(gap_index=-1, storyboard_path=Path("/x/storyboard.json"))

    def test_service_survives_a_rejected_command(self, service):
        service.start()
        with pytest.raises(BlenderCommandFailed):
            service.prepare_gap(gap_index=-1, storyboard_path=Path("/x/storyboard.json"))
        # A rejected command is not a crash; the warm process must stay usable.
        assert service.is_alive
        assert service.prepare_gap(gap_index=0, storyboard_path=Path("/x/s.json"))["ok"] is True


class TestFailureHandling:
    def test_crash_during_a_request_raises_promptly(self, service):
        service.start()
        with pytest.raises(BlenderServiceCrashed):
            service.open_job(Path("/jobs/crash-now/manifest.json"))

    def test_service_reports_dead_after_a_crash(self, service):
        service.start()
        with pytest.raises(BlenderServiceCrashed):
            service.open_job(Path("/jobs/crash-now/manifest.json"))
        assert not service.is_alive

    def test_silent_process_trips_the_stall_timeout(self, service):
        service.start()
        with pytest.raises(BlenderServiceStalled):
            service.open_job(Path("/jobs/go-silent/manifest.json"), timeout_seconds=30.0)

    def test_stalled_process_is_stopped_not_left_running(self, service):
        service.start()
        with pytest.raises(BlenderServiceStalled):
            service.open_job(Path("/jobs/go-silent/manifest.json"), timeout_seconds=30.0)
        assert not service.is_alive

    def test_recovery_after_a_crash_is_possible_via_restart(self, service):
        service.start()
        with pytest.raises(BlenderServiceCrashed):
            service.open_job(Path("/jobs/crash-now/manifest.json"))
        service.restart()
        assert service.request(ping_command("req-after-crash"))["ok"] is True


def _write_valid_png(path: Path) -> None:
    path.write_bytes(PNG_MAGIC + b"\x00" * 200)


class TestResumeFrameAccounting:
    def test_frame_names_are_zero_padded_and_sortable(self, tmp_path):
        assert frame_output_path(tmp_path, 7).name == "frame_000007.png"
        assert frame_output_path(tmp_path, 123456).name == "frame_123456.png"

    def test_absent_frame_is_not_complete(self, tmp_path):
        assert not is_complete_frame(tmp_path / "frame_000001.png")

    def test_truncated_frame_is_not_complete(self, tmp_path):
        path = tmp_path / "frame_000001.png"
        path.write_bytes(PNG_MAGIC)
        assert not is_complete_frame(path)

    def test_file_without_png_magic_is_not_complete(self, tmp_path):
        path = tmp_path / "frame_000001.png"
        path.write_bytes(b"NOT-A-PNG" + b"\x00" * 200)
        assert not is_complete_frame(path)

    def test_directory_in_place_of_a_frame_is_not_complete(self, tmp_path):
        path = tmp_path / "frame_000001.png"
        path.mkdir()
        assert not is_complete_frame(path)

    def test_valid_png_is_complete(self, tmp_path):
        path = tmp_path / "frame_000001.png"
        _write_valid_png(path)
        assert is_complete_frame(path)

    def test_missing_frames_preserve_request_order(self, tmp_path):
        for index in (2, 4):
            _write_valid_png(frame_output_path(tmp_path, index))
        assert missing_frame_indexes(tmp_path, [1, 2, 3, 4, 5]) == [1, 3, 5]

    def test_nothing_missing_when_every_frame_is_present(self, tmp_path):
        for index in (1, 2, 3):
            _write_valid_png(frame_output_path(tmp_path, index))
        assert missing_frame_indexes(tmp_path, [1, 2, 3]) == []

    def test_everything_missing_for_an_empty_directory(self, tmp_path):
        assert missing_frame_indexes(tmp_path, [1, 2, 3]) == [1, 2, 3]


class TestLogTeeing:
    def test_non_protocol_output_is_written_to_the_log(self, service, tmp_path):
        service.start()
        service.render_frames(gap_index=0, frame_indexes=[1, 2], output_directory=Path("/out"))
        service.shutdown()
        log_text = (tmp_path / "blender.log").read_text(encoding="utf-8")
        assert "Blender 4.5.10 LTS (fake)" in log_text
        assert "Fra:1 Mem:64.00M" in log_text

    def test_protocol_lines_do_not_pollute_the_log(self, service, tmp_path):
        service.start()
        service.request(ping_command("req-log"))
        service.shutdown()
        log_text = (tmp_path / "blender.log").read_text(encoding="utf-8")
        assert "@FOR3D@" not in log_text
