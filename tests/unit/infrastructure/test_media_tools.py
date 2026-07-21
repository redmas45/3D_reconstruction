import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.cancellation import CancellationRequestedError
from infrastructure.media_tools import _wait_for_media_process


class MediaToolsTests(unittest.TestCase):
    def test_running_ffmpeg_process_is_terminated_on_cancellation(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.wait.return_value = 0

        with self.assertRaises(CancellationRequestedError):
            _wait_for_media_process(process, time.monotonic() + 10.0, lambda: True)

        process.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
