import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.application_lock import (
    ApplicationAlreadyRunningError,
    ApplicationInstanceLock,
)


class ApplicationInstanceLockTests(unittest.TestCase):
    def test_only_one_process_owner_can_hold_the_output_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "server.lock"
            first_lock = ApplicationInstanceLock(lock_path)
            second_lock = ApplicationInstanceLock(lock_path)
            first_lock.acquire()
            try:
                with self.assertRaises(ApplicationAlreadyRunningError):
                    second_lock.acquire()
            finally:
                first_lock.release()

            second_lock.acquire()
            second_lock.release()


if __name__ == "__main__":
    unittest.main()
