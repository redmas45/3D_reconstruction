"""M1 exit criteria, exercised against a real headless Blender.

Implementation_plan.md §11 M1 requires: many frames across several gaps rendered in one
process with zero relaunches, and a warm-versus-cold determinism check (§6.6) proving
that reusing a process does not change output.

Skipped automatically when Blender is unavailable, so the unit suite stays runnable
anywhere. Run explicitly with:

    python -m pytest backend/tests/integration -q
"""

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.blender_runner import BlenderUnavailableError, find_blender_executable
from infrastructure.blender_service import (
    BlenderService,
    BlenderServiceError,
    missing_frame_indexes,
    spawn_blender_process,
)

SERVICE_SCRIPT = PROJECT_ROOT / "backend" / "legacy" / "blender" / "service.py"

# §11 M1 exit criterion: 200 frames across 4 gaps in one process, zero relaunches.
GAP_COUNT = 4
FRAMES_PER_GAP = 50
INTERRUPTED_FRAME_COUNT = 40
FRAMES_BEFORE_KILL = 5
KILL_POLL_SECONDS = 0.05
KILL_POLL_ATTEMPTS = 400
STARTUP_TIMEOUT_SECONDS = 240.0
RENDER_TIMEOUT_SECONDS = 900.0


def _blender_executable() -> Path:
    try:
        return find_blender_executable()
    except BlenderUnavailableError:
        pytest.skip("Blender is not installed on this machine")


pytestmark = pytest.mark.integration


def _job_manifest(resolution: tuple[int, int] = (320, 180)) -> dict:
    """A small resolution keeps the integration test fast; the contract is identical."""
    return {
        "resolution": list(resolution),
        "camera": {
            "focal_length_mm": 35.0,
            "position": [0.0, -12.0, 4.5],
            "rotation_degrees": [68.0, 0.0, 0.0],
        },
        "render": {"engine": "BLENDER_EEVEE_NEXT", "samples": 4},
    }


def _gap_specification(gap_index: int) -> dict:
    """Two people and a car walking a straight path — enough to exercise instancing."""
    actors = [
        {
            "id": f"person_{gap_index}_a",
            "kind": "person",
            "color": [0.2, 0.4, 0.8],
            "keyframes": [
                {"frame": 1, "location": [-2.0, 3.0, 0.875], "heading_degrees": 90.0},
                {"frame": FRAMES_PER_GAP, "location": [3.0, 3.0, 0.875], "heading_degrees": 90.0},
            ],
        },
        {
            "id": f"vehicle_{gap_index}",
            "kind": "vehicle",
            "vehicle_class": "car",
            "color": [0.7, 0.1, 0.1],
            "keyframes": [
                {"frame": 1, "location": [5.0, 9.0, 0.72], "heading_degrees": 180.0},
                {"frame": FRAMES_PER_GAP, "location": [-1.0, 9.0, 0.72], "heading_degrees": 180.0},
            ],
        },
    ]
    return {"gap_index": gap_index, "frame_count": FRAMES_PER_GAP, "actors": actors}


def _write_contracts(root: Path, gap_count: int) -> tuple[Path, list[Path]]:
    manifest_path = root / "job_manifest.json"
    manifest_path.write_text(json.dumps(_job_manifest()), encoding="utf-8")
    gap_paths = []
    for gap_index in range(gap_count):
        gap_path = root / f"gap_{gap_index:02d}.json"
        gap_path.write_text(json.dumps(_gap_specification(gap_index)), encoding="utf-8")
        gap_paths.append(gap_path)
    return manifest_path, gap_paths


def _build_service(tmp_path: Path) -> BlenderService:
    executable = _blender_executable()
    return BlenderService(
        lambda: spawn_blender_process(executable, SERVICE_SCRIPT, tmp_path, PROJECT_ROOT),
        log_path=tmp_path / "blender.log",
        startup_timeout_seconds=STARTUP_TIMEOUT_SECONDS,
    )


def _completed_frames(output_directory: Path, frame_indexes: list[int]) -> list[int]:
    outstanding = set(missing_frame_indexes(output_directory, frame_indexes))
    return [index for index in frame_indexes if index not in outstanding]


def _frame_signatures(output_directory: Path) -> dict[str, tuple[int, float]]:
    """Size and mtime per frame — enough to prove a file was not rewritten."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime)
        for path in sorted(output_directory.glob("frame_*.png"))
    }


def _kill_after_frames(
    service: BlenderService, output_directory: Path, frame_indexes: list[int],
) -> None:
    """Hard-stop Blender once enough frames have landed, simulating a crash."""
    for _ in range(KILL_POLL_ATTEMPTS):
        if len(_completed_frames(output_directory, frame_indexes)) >= FRAMES_BEFORE_KILL:
            service.abort()
            return
        time.sleep(KILL_POLL_SECONDS)


def _pixel_digest(path: Path) -> str:
    """Hash decoded pixels, never the file bytes.

    Blender stamps `Date` and `RenderTime` tEXt chunks into every PNG, so two
    byte-identical renders produce different file hashes one second apart. Comparing
    raw bytes reports a state leak that is not there.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AssertionError(f"Could not decode rendered frame: {path}")
    return hashlib.sha256(numpy.ascontiguousarray(image).tobytes()).hexdigest()


class TestWarmProcessRendersManyGaps:
    def test_all_gaps_render_in_one_process(self, tmp_path):
        manifest_path, gap_paths = _write_contracts(tmp_path, GAP_COUNT)
        service = _build_service(tmp_path)
        capabilities = service.start()
        try:
            assert capabilities["protocol_version"] == 1
            service.open_job(manifest_path)
            for gap_index, gap_path in enumerate(gap_paths):
                self._render_one_gap(service, gap_index, gap_path, tmp_path)
                # The whole point: the process survives every gap.
                assert service.is_alive
            summary = service.close_job()
            assert summary["rendered_frame_count"] == GAP_COUNT * FRAMES_PER_GAP
        finally:
            service.shutdown()

    @staticmethod
    def _render_one_gap(service, gap_index: int, gap_path: Path, tmp_path: Path) -> None:
        service.prepare_gap(gap_index, gap_path)
        output_directory = tmp_path / f"render_{gap_index:02d}"
        result = service.render_frames(
            gap_index,
            list(range(1, FRAMES_PER_GAP + 1)),
            output_directory,
            timeout_seconds=RENDER_TIMEOUT_SECONDS,
        )
        assert result["rendered_frame_indexes"] == list(range(1, FRAMES_PER_GAP + 1))
        for frame in result["frames"]:
            assert Path(frame["path"]).is_file()
            assert frame["bytes"] > 64
        service.reset_gap()

    def test_actor_census_does_not_grow_across_gaps(self, tmp_path):
        """A leaking warm process would show monotonically rising datablock counts (§6.6)."""
        manifest_path, gap_paths = _write_contracts(tmp_path, GAP_COUNT)
        service = _build_service(tmp_path)
        service.start()
        try:
            service.open_job(manifest_path)
            object_counts = []
            for gap_index, gap_path in enumerate(gap_paths):
                service.prepare_gap(gap_index, gap_path)
                reset_summary = service.reset_gap()
                object_counts.append(reset_summary["census"]["objects"])
            assert reset_summary["census"]["actors"] == 0
            assert len(set(object_counts)) == 1, f"datablocks leaked across gaps: {object_counts}"
        finally:
            service.shutdown()


class TestInterruptedRenderResumes:
    """§6.7 / §11 M1: a killed process must cost one frame, not a whole gap."""

    def test_resume_renders_only_the_missing_frames(self, tmp_path):
        manifest_path, gap_paths = _write_contracts(tmp_path, 1)
        output_directory = tmp_path / "resume"
        all_frames = list(range(1, INTERRUPTED_FRAME_COUNT + 1))

        completed_before_kill = self._render_until_killed(
            tmp_path, manifest_path, gap_paths[0], output_directory, all_frames,
        )
        assert completed_before_kill, "kill happened before any frame finished"
        signatures_before = _frame_signatures(output_directory)

        outstanding = missing_frame_indexes(output_directory, all_frames)
        assert len(outstanding) < len(all_frames), "resume would redo the entire gap"

        service = _build_service(tmp_path)
        service.start()
        try:
            service.open_job(manifest_path)
            service.prepare_gap(0, gap_paths[0])
            result = service.render_frames(
                0, outstanding, output_directory, timeout_seconds=RENDER_TIMEOUT_SECONDS,
            )
        finally:
            service.shutdown()

        assert result["rendered_frame_indexes"] == outstanding
        assert missing_frame_indexes(output_directory, all_frames) == []
        # Frames that survived the kill must not have been touched again.
        for name, signature in signatures_before.items():
            assert _frame_signatures(output_directory)[name] == signature, (
                f"{name} was re-rendered despite already being complete"
            )

    @staticmethod
    def _render_until_killed(
        tmp_path: Path,
        manifest_path: Path,
        gap_path: Path,
        output_directory: Path,
        all_frames: list[int],
    ) -> int:
        """Start a long render, hard-kill Blender partway, return frames completed."""
        service = _build_service(tmp_path)
        service.start()
        service.open_job(manifest_path)
        service.prepare_gap(0, gap_path)

        killer = threading.Thread(
            target=_kill_after_frames, args=(service, output_directory, all_frames), daemon=True,
        )
        killer.start()
        with pytest.raises(BlenderServiceError):
            service.render_frames(
                0, all_frames, output_directory, timeout_seconds=RENDER_TIMEOUT_SECONDS,
            )
        killer.join(timeout=30.0)
        service.shutdown()
        return len(_completed_frames(output_directory, all_frames))


class TestWarmAndColdAgree:
    def test_gap_rendered_warm_matches_the_same_gap_rendered_cold(self, tmp_path):
        """§6.6's central assumption: process reuse must not change output."""
        manifest_path, gap_paths = _write_contracts(tmp_path, 2)
        cold_directory = tmp_path / "cold"
        warm_directory = tmp_path / "warm"

        cold_service = _build_service(tmp_path)
        cold_service.start()
        try:
            cold_service.open_job(manifest_path)
            cold_service.prepare_gap(1, gap_paths[1])
            cold_service.render_frames(
                1, [1, 2], cold_directory, timeout_seconds=RENDER_TIMEOUT_SECONDS,
            )
        finally:
            cold_service.shutdown()

        warm_service = _build_service(tmp_path)
        warm_service.start()
        try:
            warm_service.open_job(manifest_path)
            # Render gap 0 first so gap 1 is rendered by an already-used process.
            warm_service.prepare_gap(0, gap_paths[0])
            warm_service.render_frames(
                0, [1, 2], tmp_path / "warm_discard", timeout_seconds=RENDER_TIMEOUT_SECONDS,
            )
            warm_service.reset_gap()
            warm_service.prepare_gap(1, gap_paths[1])
            warm_service.render_frames(
                1, [1, 2], warm_directory, timeout_seconds=RENDER_TIMEOUT_SECONDS,
            )
        finally:
            warm_service.shutdown()

        for frame_index in (1, 2):
            name = f"frame_{frame_index:06d}.png"
            assert _pixel_digest(cold_directory / name) == _pixel_digest(warm_directory / name), (
                f"warm and cold renders of frame {frame_index} differ — "
                "state is leaking between gaps"
            )

    def test_repeating_one_gap_in_a_warm_process_is_deterministic(self, tmp_path):
        """Tighter than the warm/cold check: isolates render nondeterminism from leaks."""
        manifest_path, gap_paths = _write_contracts(tmp_path, 1)
        service = _build_service(tmp_path)
        service.start()
        try:
            service.open_job(manifest_path)
            for attempt in range(2):
                service.prepare_gap(0, gap_paths[0])
                service.render_frames(
                    0, [1], tmp_path / f"attempt_{attempt}",
                    timeout_seconds=RENDER_TIMEOUT_SECONDS,
                )
                service.reset_gap()
        finally:
            service.shutdown()
        first = _pixel_digest(tmp_path / "attempt_0" / "frame_000001.png")
        second = _pixel_digest(tmp_path / "attempt_1" / "frame_000001.png")
        assert first == second
