import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.cancellation import CancellationRequestedError
from infrastructure.media_tools import (
    MediaProcessingError,
    VideoContract,
    _wait_for_media_process,
    encode_with_source_audio,
    validate_constant_frame_rate,
)


class MediaToolsTests(unittest.TestCase):
    def test_declared_variable_frame_rate_is_rejected(self) -> None:
        media_report = {
            "streams": [{
                "codec_type": "video",
                "avg_frame_rate": "24/1",
                "r_frame_rate": "30/1",
            }],
        }

        with patch("infrastructure.media_tools.probe_media", return_value=media_report):
            with self.assertRaisesRegex(MediaProcessingError, "Variable-frame-rate"):
                validate_constant_frame_rate(Path("fixture.mp4"), 24.0)

    def test_matching_constant_frame_rate_is_accepted(self) -> None:
        media_report = {
            "streams": [{
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            }],
        }

        with patch("infrastructure.media_tools.probe_media", return_value=media_report):
            validate_constant_frame_rate(Path("fixture.mp4"), 29.97002997)

    def test_running_ffmpeg_process_is_terminated_on_cancellation(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = 0

        with self.assertRaises(CancellationRequestedError):
            _wait_for_media_process(process, time.monotonic() + 10.0, lambda: True)

        process.terminate.assert_called_once_with()

    def test_audio_mux_uses_video_duration_instead_of_shortest_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "final.mp4"
            captured_command: list[str] = []

            def run_command(command: list[str], log_path: Path, cancellation_check: object) -> None:
                del log_path, cancellation_check
                captured_command.extend(command)
                output_path.write_bytes(b"encoded")

            with patch("infrastructure.media_tools.find_media_tool", return_value=Path("ffmpeg")):
                with patch(
                    "infrastructure.media_tools.inspect_video_contract",
                    return_value=VideoContract(640, 480, 30.0, 300),
                ):
                    with patch("infrastructure.media_tools._run_media_command", side_effect=run_command):
                        encode_with_source_audio(root / "video.mp4", root / "source.mp4", output_path)

            self.assertNotIn("-shortest", captured_command)
            duration_index = captured_command.index("-t") + 1
            self.assertEqual("10.000000000", captured_command[duration_index])


if __name__ == "__main__":
    unittest.main()
