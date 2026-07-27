"""Carries Blender's compiled GPU kernels across Colab sessions (§6.5).

OptiX kernels are compiled per GPU architecture on first render. A Colab T4 is always
SM75, so the compiled cache is reusable — but Colab wipes local state on every restart,
so the cost is paid again on every single run. §6.1 measures it at 30-60 seconds.

The fix is to keep the cache on Drive as one archive and restore it at session start.
Drive is slow per file and fine for one large file, hence the archive rather than a
symlink into the mount.

Linux-only by design: Blender honours `XDG_CACHE_HOME` there, which is the environment
Colab runs in. On Windows and macOS Blender uses platform-specific locations that this
does not attempt to redirect, so calls become no-ops rather than pretending to work.
"""

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path


LOGGER = logging.getLogger(__name__)

CACHE_ENVIRONMENT_KEY = "XDG_CACHE_HOME"
KERNEL_CACHE_ARCHIVE_NAME = "blender_kernel_cache.tar.gz"
BLENDER_CACHE_SUBDIRECTORY = "blender"

# `filter="data"` refuses absolute paths, parent traversal, and special files while
# extracting. The archive is our own, but it round-trips through Drive, so it is
# treated as untrusted input at the boundary (rules.md §9).
SAFE_EXTRACTION_FILTER = "data"


def is_supported_platform() -> bool:
    """Only Linux keeps its Blender cache where `XDG_CACHE_HOME` points."""
    import sys

    return sys.platform.startswith("linux")


def kernel_cache_environment(cache_root: Path) -> dict[str, str]:
    """Environment overlay that points Blender's cache at `cache_root`."""
    if not is_supported_platform():
        return {}
    cache_root.mkdir(parents=True, exist_ok=True)
    return {CACHE_ENVIRONMENT_KEY: str(cache_root)}


def kernel_cache_directory(cache_root: Path) -> Path:
    return cache_root / BLENDER_CACHE_SUBDIRECTORY


def restore_kernel_cache(archive_path: Path, cache_root: Path) -> bool:
    """Unpack a previously archived cache. Returns whether anything was restored."""
    if not is_supported_platform() or not archive_path.is_file():
        return False
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(path=cache_root, filter=SAFE_EXTRACTION_FILTER)
    except (OSError, tarfile.TarError, ValueError) as error:
        # A corrupt archive must cost a recompile, never the whole run.
        LOGGER.warning("Could not restore Blender kernel cache: %s", error)
        return False
    return kernel_cache_directory(cache_root).is_dir()


def archive_kernel_cache(cache_root: Path, archive_path: Path) -> bool:
    """Pack the compiled cache for the next session. Returns whether one was written.

    Writes to a temporary file and renames, so an interrupted upload cannot leave a
    truncated archive that the next session would try to restore.
    """
    cache_directory = kernel_cache_directory(cache_root)
    if not is_supported_platform() or not cache_directory.is_dir():
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        suffix=".tar.gz", dir=str(archive_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with open(handle, "wb") as temporary_file:
            with tarfile.open(fileobj=temporary_file, mode="w:gz") as archive:
                archive.add(cache_directory, arcname=BLENDER_CACHE_SUBDIRECTORY)
        shutil.move(str(temporary_path), str(archive_path))
    except (OSError, tarfile.TarError) as error:
        LOGGER.warning("Could not archive Blender kernel cache: %s", error)
        temporary_path.unlink(missing_ok=True)
        return False
    return archive_path.is_file()
