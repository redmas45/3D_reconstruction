import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.cancellation import CancellationRequestedError
from infrastructure.blender_runner import (
    BlenderRenderRequest,
    RenderHeartbeat,
    _parse_progress_line,
    _wait_for_blender,
    build_blender_command,
)


class BlenderRunnerTests(unittest.TestCase):
    def test_command_keeps_contract_paths_and_mode_explicit(self) -> None:
        request = BlenderRenderRequest(
            plan_path=Path("plan.json"),
            output_path=Path("preview.png"),
            report_path=Path("report.json"),
            blend_path=Path("scene.blend"),
            log_path=Path("blender.log"),
            mode="preview",
        )

        command = build_blender_command(Path("blender.exe"), Path("render_gap.py"), request)

        self.assertEqual("--background", command[1])
        self.assertIn(str(Path("plan.json").resolve()), command)
        self.assertEqual("preview", command[-1])

    def test_running_blender_process_is_terminated_on_cancellation(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = 0

        with self.assertRaises(CancellationRequestedError):
            _wait_for_blender(process, time.monotonic() + 10.0, 10, lambda: True)

        process.terminate.assert_called_once_with()

    def test_progress_marker_reports_bounded_frame_counts(self) -> None:
        self.assertEqual((17, 90), _parse_progress_line("RECON_PROGRESS 17 90\n"))
        self.assertEqual((90, 90), _parse_progress_line("RECON_PROGRESS 100 90\n"))
        self.assertIsNone(_parse_progress_line("Fra:17 Mem:120.0M"))

    def test_render_heartbeat_resets_inactivity_timeout(self) -> None:
        with patch("infrastructure.blender_runner.time.monotonic", return_value=100.0):
            heartbeat = RenderHeartbeat()
        with patch("infrastructure.blender_runner.time.monotonic", return_value=161.0):
            self.assertTrue(heartbeat.has_stalled(60))
        with patch("infrastructure.blender_runner.time.monotonic", return_value=170.0):
            heartbeat.mark_progress()
        with patch("infrastructure.blender_runner.time.monotonic", return_value=200.0):
            self.assertFalse(heartbeat.has_stalled(60))


if __name__ == "__main__":
    unittest.main()
