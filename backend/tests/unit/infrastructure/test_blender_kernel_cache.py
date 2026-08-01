import sys
import tarfile
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure import blender_kernel_cache
from infrastructure.blender_kernel_cache import (
    BLENDER_CACHE_SUBDIRECTORY,
    CACHE_ENVIRONMENT_KEY,
    archive_kernel_cache,
    kernel_cache_directory,
    kernel_cache_environment,
    restore_kernel_cache,
)


@pytest.fixture
def on_linux(monkeypatch):
    """The cache redirect only applies where Blender honours XDG_CACHE_HOME."""
    monkeypatch.setattr(blender_kernel_cache, "is_supported_platform", lambda: True)


def _populate_cache(cache_root: Path) -> Path:
    kernel_directory = kernel_cache_directory(cache_root) / "4.5" / "kernels"
    kernel_directory.mkdir(parents=True, exist_ok=True)
    (kernel_directory / "optix_sm75.ptx").write_bytes(b"compiled-kernel-bytes")
    return kernel_directory


class TestEnvironmentOverlay:
    def test_overlay_points_blender_at_the_cache_root(self, tmp_path, on_linux):
        overlay = kernel_cache_environment(tmp_path / "cache")
        assert overlay == {CACHE_ENVIRONMENT_KEY: str(tmp_path / "cache")}

    def test_overlay_creates_the_directory(self, tmp_path, on_linux):
        kernel_cache_environment(tmp_path / "cache")
        assert (tmp_path / "cache").is_dir()

    def test_unsupported_platform_yields_no_overlay(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blender_kernel_cache, "is_supported_platform", lambda: False)
        assert kernel_cache_environment(tmp_path / "cache") == {}


class TestRoundTrip:
    def test_archive_then_restore_reproduces_the_kernels(self, tmp_path, on_linux):
        source_root = tmp_path / "source"
        _populate_cache(source_root)
        archive_path = tmp_path / "drive" / "cache.tar.gz"
        assert archive_kernel_cache(source_root, archive_path)

        restored_root = tmp_path / "restored"
        assert restore_kernel_cache(archive_path, restored_root)
        restored_kernel = (
            kernel_cache_directory(restored_root) / "4.5" / "kernels" / "optix_sm75.ptx"
        )
        assert restored_kernel.read_bytes() == b"compiled-kernel-bytes"

    def test_archiving_an_absent_cache_is_a_no_op(self, tmp_path, on_linux):
        assert not archive_kernel_cache(tmp_path / "empty", tmp_path / "out.tar.gz")

    def test_restoring_an_absent_archive_is_a_no_op(self, tmp_path, on_linux):
        assert not restore_kernel_cache(tmp_path / "nothing.tar.gz", tmp_path / "cache")

    def test_no_partial_archive_survives_a_failure(self, tmp_path, on_linux):
        source_root = tmp_path / "source"
        _populate_cache(source_root)
        archive_path = tmp_path / "drive" / "cache.tar.gz"
        archive_kernel_cache(source_root, archive_path)
        # Only the finished archive should exist; no temporary files left behind.
        assert list(archive_path.parent.iterdir()) == [archive_path]


class TestCorruptArchivesDegradeGracefully:
    def test_corrupt_archive_returns_false_rather_than_raising(self, tmp_path, on_linux):
        archive_path = tmp_path / "cache.tar.gz"
        archive_path.write_bytes(b"this is not a gzip stream")
        assert restore_kernel_cache(archive_path, tmp_path / "cache") is False

    def test_archive_escaping_the_cache_root_is_refused(self, tmp_path, on_linux):
        """The archive round-trips through Drive, so extraction is treated as untrusted."""
        archive_path = tmp_path / "evil.tar.gz"
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("original", encoding="utf-8")
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(outside_file, arcname=f"../{outside_file.name}")

        restore_kernel_cache(archive_path, tmp_path / "cache")
        assert outside_file.read_text(encoding="utf-8") == "original"
