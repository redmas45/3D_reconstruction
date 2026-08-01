"""Prevents two local app processes from owning the same job directories."""

import os
from pathlib import Path
from typing import BinaryIO


class ApplicationAlreadyRunningError(RuntimeError):
    pass


class ApplicationInstanceLock:
    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path.resolve()
        self._lock_file: BinaryIO | None = None

    def acquire(self) -> None:
        if self._lock_file is not None:
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self._lock_path.open("a+b")
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            _acquire_file_lock(lock_file)
        except OSError as error:
            lock_file.close()
            raise ApplicationAlreadyRunningError(
                "Another reconstruction app is already using this project output directory"
            ) from error
        self._lock_file = lock_file

    def release(self) -> None:
        if self._lock_file is None:
            return
        try:
            _release_file_lock(self._lock_file)
        finally:
            self._lock_file.close()
            self._lock_file = None


def _acquire_file_lock(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
