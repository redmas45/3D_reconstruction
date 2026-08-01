"""Colab orchestration is repo code now, so it gets tested like repo code.

None of this was reachable by tests while it lived inside a notebook cell.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interfaces.colab_run import (
    AZURE_SECRET_NAMES,
    ColabRenderSettings,
    DriveCheckpointer,
    EnvironmentFileError,
    build_run_identifier,
    build_runtime_configuration,
    missing_azure_names,
    parse_environment_file,
    validate_video_selection,
)

CONFIGURATION_PATH = PROJECT_ROOT / "backend" / "config" / "reconstruction_config.json"


class TestEnvironmentFileParsing:
    def test_reads_the_three_azure_values(self):
        payload = (
            b"AZURE_GROK_API_KEY=abc123\n"
            b"AZURE_GROK_BASE_URL=https://example.services.ai.azure.com/openai/v1\n"
            b"AZURE_GROK_CHAT_DEPLOYMENT=grok-4-20-reasoning\n"
        )
        values = parse_environment_file(payload)
        assert set(values) == set(AZURE_SECRET_NAMES)
        assert values["AZURE_GROK_API_KEY"] == "abc123"

    def test_ignores_comments_and_blank_lines(self):
        payload = b"# a comment\n\nAZURE_GROK_API_KEY=abc\n"
        assert parse_environment_file(payload)["AZURE_GROK_API_KEY"] == "abc"

    def test_accepts_export_prefixed_assignments(self):
        payload = b"export AZURE_GROK_API_KEY=abc\n"
        assert parse_environment_file(payload)["AZURE_GROK_API_KEY"] == "abc"

    @pytest.mark.parametrize("quote", [b"'", b'"'])
    def test_strips_matching_quotes(self, quote):
        payload = b"AZURE_GROK_API_KEY=" + quote + b"abc" + quote + b"\n"
        assert parse_environment_file(payload)["AZURE_GROK_API_KEY"] == "abc"

    def test_unrelated_local_secrets_are_not_loaded(self):
        """An operator's .env often holds unrelated keys; they must stay out of Colab."""
        payload = b"AWS_SECRET_ACCESS_KEY=nope\nAZURE_GROK_API_KEY=abc\n"
        values = parse_environment_file(payload)
        assert "AWS_SECRET_ACCESS_KEY" not in values

    def test_malformed_line_is_rejected(self):
        with pytest.raises(EnvironmentFileError, match="line 1"):
            parse_environment_file(b"this line has no equals sign\n")

    def test_oversized_file_is_rejected_before_decoding(self):
        with pytest.raises(EnvironmentFileError, match="unexpectedly large"):
            parse_environment_file(b"x" * (64 * 1024 + 1))

    def test_non_utf8_payload_is_rejected(self):
        with pytest.raises(EnvironmentFileError, match="UTF-8"):
            parse_environment_file(b"\xff\xfe\x00KEY=value")


class TestMissingCredentialReporting:
    def test_reports_every_absent_name(self):
        assert missing_azure_names({}) == list(AZURE_SECRET_NAMES)

    def test_blank_values_count_as_missing(self):
        values = {name: "   " for name in AZURE_SECRET_NAMES}
        assert missing_azure_names(values) == list(AZURE_SECRET_NAMES)

    def test_complete_values_report_nothing_missing(self):
        values = {name: "set" for name in AZURE_SECRET_NAMES}
        assert missing_azure_names(values) == []


class TestVideoSelection:
    def test_supported_extension_is_accepted(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        assert validate_video_selection(video) == video

    def test_unsupported_extension_is_rejected(self, tmp_path):
        video = tmp_path / "notes.txt"
        video.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported video extension"):
            validate_video_selection(video)

    def test_absent_file_is_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_video_selection(tmp_path / "missing.mp4")


class TestRuntimeConfiguration:
    def test_overrides_are_applied_and_validate(self):
        settings = ColabRenderSettings(engine="CYCLES", cycles_samples=4)
        configuration = build_runtime_configuration(CONFIGURATION_PATH, settings)
        assert configuration["renderer"]["engine"] == "CYCLES"
        assert configuration["renderer"]["cycles_samples"] == 4

    def test_scale_is_applied_to_both_keys_the_renderer_reads(self):
        settings = ColabRenderSettings(production_scale_percent=55)
        configuration = build_runtime_configuration(CONFIGURATION_PATH, settings)
        assert configuration["renderer"]["production_scale_percent"] == 55
        assert configuration["renderer"]["scale_percent"] == 55

    def test_the_checked_in_configuration_is_not_mutated(self):
        before = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings(engine="CYCLES"))
        after = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        assert before == after


class TestRunIdentifier:
    @staticmethod
    def _video(tmp_path: Path, content: bytes = b"video-bytes") -> Path:
        video = tmp_path / "clip.mp4"
        video.write_bytes(content)
        return video

    def test_identifier_is_stable_for_identical_inputs(self, tmp_path):
        video = self._video(tmp_path)
        configuration = build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings())
        first = build_run_identifier(video, configuration, 42, "4.5.10", PROJECT_ROOT)
        second = build_run_identifier(video, configuration, 42, "4.5.10", PROJECT_ROOT)
        assert first == second

    def test_different_video_content_changes_the_identifier(self, tmp_path):
        configuration = build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings())
        first_directory = tmp_path / "a"
        second_directory = tmp_path / "b"
        first_directory.mkdir()
        second_directory.mkdir()
        first = build_run_identifier(
            self._video(first_directory, b"one"), configuration, 42, "4.5.10", PROJECT_ROOT,
        )
        second = build_run_identifier(
            self._video(second_directory, b"two"), configuration, 42, "4.5.10", PROJECT_ROOT,
        )
        assert first != second

    def test_different_seed_changes_the_identifier(self, tmp_path):
        video = self._video(tmp_path)
        configuration = build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings())
        assert build_run_identifier(video, configuration, 1, "4.5.10", PROJECT_ROOT) != \
            build_run_identifier(video, configuration, 2, "4.5.10", PROJECT_ROOT)

    def test_worker_count_does_not_invalidate_checkpoints(self, tmp_path):
        """Colab hands out different machines; that must not orphan a resume."""
        video = self._video(tmp_path)
        one_worker = build_runtime_configuration(
            CONFIGURATION_PATH, ColabRenderSettings(parallel_gap_renders=1),
        )
        three_workers = build_runtime_configuration(
            CONFIGURATION_PATH, ColabRenderSettings(parallel_gap_renders=3),
        )
        assert build_run_identifier(video, one_worker, 42, "4.5.10", PROJECT_ROOT) == \
            build_run_identifier(video, three_workers, 42, "4.5.10", PROJECT_ROOT)

    def test_render_settings_do_invalidate_checkpoints(self, tmp_path):
        video = self._video(tmp_path)
        low = build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings(cycles_samples=2))
        high = build_runtime_configuration(CONFIGURATION_PATH, ColabRenderSettings(cycles_samples=16))
        assert build_run_identifier(video, low, 42, "4.5.10", PROJECT_ROOT) != \
            build_run_identifier(video, high, 42, "4.5.10", PROJECT_ROOT)


def _complete_gap(local_gaps: Path, name: str) -> Path:
    blender = local_gaps / name / "blender"
    blender.mkdir(parents=True, exist_ok=True)
    for artifact in ("gap_blender.mp4", "render_report.json", "scene.blend"):
        (blender / artifact).write_bytes(b"artifact")
    return local_gaps / name


class TestDriveCheckpointer:
    def test_complete_gaps_are_saved(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        _complete_gap(local_gaps, "gap_00")
        checkpointer = DriveCheckpointer(local_gaps, drive_gaps)
        assert checkpointer.save_completed_gaps() == ["gap_00"]
        assert (drive_gaps / "gap_00" / "blender" / "gap_blender.mp4").is_file()

    def test_incomplete_gaps_are_not_saved(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        blender = local_gaps / "gap_00" / "blender"
        blender.mkdir(parents=True)
        (blender / "gap_blender.mp4").write_bytes(b"only one artifact")
        checkpointer = DriveCheckpointer(local_gaps, drive_gaps)
        assert checkpointer.save_completed_gaps() == []

    def test_a_gap_is_saved_only_once(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        _complete_gap(local_gaps, "gap_00")
        checkpointer = DriveCheckpointer(local_gaps, drive_gaps)
        assert checkpointer.save_completed_gaps() == ["gap_00"]
        assert checkpointer.save_completed_gaps() == []

    def test_no_staging_directory_survives_a_save(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        _complete_gap(local_gaps, "gap_00")
        DriveCheckpointer(local_gaps, drive_gaps).save_completed_gaps()
        assert [path.name for path in drive_gaps.iterdir()] == ["gap_00"]

    def test_restore_reports_when_there_is_nothing_to_restore(self, tmp_path):
        checkpointer = DriveCheckpointer(tmp_path / "local", tmp_path / "drive")
        assert checkpointer.restore() is False

    def test_restore_brings_back_saved_gaps(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        _complete_gap(local_gaps, "gap_00")
        DriveCheckpointer(local_gaps, drive_gaps).save_completed_gaps()

        fresh_local = tmp_path / "fresh"
        restored = DriveCheckpointer(fresh_local, drive_gaps)
        assert restored.restore() is True
        assert (fresh_local / "gap_00" / "blender" / "scene.blend").is_file()

    def test_frame_manifest_is_copied_after_its_frames(self, tmp_path):
        """A manifest must never reference frames that have not landed on Drive yet."""
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        frames = local_gaps / "gap_00" / "blender" / "renders" / "frames_x"
        frames.mkdir(parents=True)
        (frames / "frame_000001.png").write_bytes(b"png")
        (frames / "frame_manifest.json").write_text("{}", encoding="utf-8")

        checkpointer = DriveCheckpointer(local_gaps, drive_gaps)
        assert checkpointer.save_sparse_frames() == 1
        destination = drive_gaps / "gap_00" / "blender" / "renders" / "frames_x"
        assert (destination / "frame_000001.png").is_file()
        assert (destination / "frame_manifest.json").is_file()

    def test_unchanged_frames_are_not_recopied(self, tmp_path):
        local_gaps, drive_gaps = tmp_path / "local", tmp_path / "drive"
        frames = local_gaps / "gap_00" / "blender" / "renders" / "frames_x"
        frames.mkdir(parents=True)
        (frames / "frame_000001.png").write_bytes(b"png")
        checkpointer = DriveCheckpointer(local_gaps, drive_gaps)
        assert checkpointer.save_sparse_frames() == 1
        assert checkpointer.save_sparse_frames() == 0
