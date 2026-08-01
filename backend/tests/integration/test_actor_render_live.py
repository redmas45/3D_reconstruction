"""M2 exit criteria, exercised against a real headless Blender.

The unit suite proves each piece in isolation against a fake service. This proves the
thing that only a real render can: that an actor placed by the plan actually lands
*inside* the crop rectangle the region arithmetic computed for it, and that the
composited frame is real footage everywhere else.

That is the failure mode worth spending a real render on. A camera convention error, a
sensor-fit mismatch, or an origin offset all produce a valid video full of untouched
plate — output that looks like a successful run until someone watches it.

Run with:

    python -m pytest backend/tests/integration -q
"""

import json
import sys
from pathlib import Path

import cv2
import numpy
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.actor_gap_renderer import build_job_manifest, plan_sparse_frames
from application.actor_render_job import ActorRenderJob, render_actor_gaps
from infrastructure.blender_runner import BlenderUnavailableError, find_blender_executable

pytestmark = pytest.mark.integration

FRAME_WIDTH = 320
FRAME_HEIGHT = 180
PLATE_VALUE = 64
STALL_TIMEOUT_SECONDS = 600.0
# A composited actor must differ from the flat plate by more than encoder noise.
ACTOR_DIFFERENCE_THRESHOLD = 12
# Outside the crop the frame is the plate, up to lossy-codec noise.
PLATE_TOLERANCE = 6


def _blender_executable() -> Path:
    try:
        return find_blender_executable()
    except BlenderUnavailableError:
        pytest.skip("Blender is not installed on this machine")


def _plan(gap_index: int = 0, frame_count: int = 30, hidden_start: int = 90) -> dict:
    """One person walking across the middle distance, straight toward the camera's right."""
    return {
        "gap_index": gap_index,
        "fps": 30.0,
        "frame_count": frame_count,
        "duration_seconds": frame_count / 30.0,
        "hidden_range": {"start": hidden_start, "end": hidden_start + frame_count - 1},
        "camera": {
            "projection_model": "pinhole_ground_plane_v2",
            "field_of_view_degrees": 54.0,
            "horizon_normalized_y": 0.35,
            "position": [0.0, 0.0, 3.0],
            "ground_mapping": {"near_y": 0.98, "far_y": 0.37},
            "calibration_confidence": 0.8,
        },
        "render": {
            "target_fps": 6.0, "engine": "BLENDER_EEVEE_NEXT", "cycles_samples": 4,
        },
        "entities": [
            {
                "id": "person_1",
                "kind": "person",
                "appearance": {"upper_color": [0.85, 0.15, 0.15]},
                "path_prediction": {
                    "waypoints": [
                        {"role": "start", "frame": hidden_start, "world": [-2.0, 12.0, 0.0]},
                        {"role": "end", "frame": hidden_start + frame_count - 1,
                         "world": [2.0, 12.0, 0.0]},
                    ],
                },
            },
        ],
    }


def _plate() -> numpy.ndarray:
    return numpy.full((FRAME_HEIGHT, FRAME_WIDTH, 3), PLATE_VALUE, dtype=numpy.uint8)


def _job(tmp_path: Path, plate: numpy.ndarray | None = None) -> ActorRenderJob:
    return ActorRenderJob(
        blender_executable=_blender_executable(),
        project_root=PROJECT_ROOT,
        job_root=tmp_path / "job",
        plate_for_gap=(lambda resolved: lambda _: resolved)(
            _plate() if plate is None else plate
        ),
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        source_fps=30.0,
        stall_timeout_seconds=STALL_TIMEOUT_SECONDS,
    )


def _read_frames(video_path: Path) -> list[numpy.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def test_a_rendered_actor_lands_inside_its_own_crop(tmp_path):
    """The M2 exit criterion: the actor is visible, and it is where the region says.

    Asserted on the layers rather than the composite so a failure points at the renderer
    rather than at the compositor.
    """
    plan = _plan()
    with _job(tmp_path) as job:
        outcome = job.render_gap(plan, tmp_path / "gap_00")
    layers = sorted((tmp_path / "gap_00" / "layers").glob("*/frame_*.png"))
    assert len(layers) == outcome.sparse_frame_count
    covered = []
    for layer_path in layers:
        layer = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
        assert layer is not None and layer.shape[2] == 4, f"{layer_path} is not RGBA"
        covered.append(float((layer[..., 3] > 0).mean()))
    assert all(coverage > 0.0 for coverage in covered), (
        f"some rendered layers were entirely transparent: {covered}"
    )


def test_layers_are_frame_sized_and_opaque_only_inside_the_border(tmp_path):
    """Border rendering saves the shading; the output stays frame-sized on purpose.

    Also the check that the ROI is doing anything at all: if `use_border` were ignored,
    the shadow-free background would still be transparent but the covered fraction would
    not be bounded by the region.
    """
    plan = _plan()
    with _job(tmp_path) as job:
        job.render_gap(plan, tmp_path / "gap_00")
    sparse_frames = plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT)
    layer_directory = next((tmp_path / "gap_00" / "layers").iterdir())
    for sparse_frame in sparse_frames:
        layer = cv2.imread(
            str(layer_directory / f"frame_{sparse_frame.render_index:06d}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        assert layer.shape[:2] == (FRAME_HEIGHT, FRAME_WIDTH)
        left, top, right, bottom = sparse_frame.region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        outside = layer[..., 3].copy()
        # One pixel of slack for Blender's own border-to-pixel rounding.
        outside[max(0, top - 1):bottom + 1, max(0, left - 1):right + 1] = 0
        assert outside.max() == 0, "the render leaked outside its border"


def test_the_composite_is_plate_everywhere_the_actor_is_not(tmp_path):
    plan = _plan()
    with _job(tmp_path) as job:
        outcome = job.render_gap(plan, tmp_path / "gap_00")
    frames = _read_frames(outcome.video_path)
    assert len(frames) == plan["frame_count"]
    sparse_frames = plan_sparse_frames(plan, FRAME_WIDTH, FRAME_HEIGHT)
    left, top, right, bottom = sparse_frames[0].region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
    outside = frames[0].copy()
    outside[max(0, top - 1):bottom + 1, max(0, left - 1):right + 1] = PLATE_VALUE
    assert abs(float(outside.mean()) - PLATE_VALUE) < PLATE_TOLERANCE


def test_the_composite_actually_changes_the_frame(tmp_path):
    """A run that renders nothing visible would otherwise pass every structural check."""
    plan = _plan()
    with _job(tmp_path) as job:
        outcome = job.render_gap(plan, tmp_path / "gap_00")
    frames = _read_frames(outcome.video_path)
    difference = max(
        float(numpy.abs(frame.astype(numpy.int16) - PLATE_VALUE).max()) for frame in frames
    )
    assert difference > ACTOR_DIFFERENCE_THRESHOLD


def test_the_actor_moves_across_the_gap(tmp_path):
    """Two frames from opposite ends of the gap must not be identical."""
    plan = _plan()
    with _job(tmp_path) as job:
        outcome = job.render_gap(plan, tmp_path / "gap_00")
    frames = _read_frames(outcome.video_path)
    difference = numpy.abs(
        frames[0].astype(numpy.int16) - frames[-1].astype(numpy.int16)
    ).max()
    assert difference > ACTOR_DIFFERENCE_THRESHOLD


def test_several_gaps_render_through_one_blender_process(tmp_path):
    plans = [_plan(gap_index=index, hidden_start=90 + index * 200) for index in range(3)]
    directories = [tmp_path / f"gap_{index:02d}" for index in range(3)]
    rendered = render_actor_gaps(plans, directories, _job(tmp_path))
    assert sorted(rendered) == [0, 1, 2]
    for gap_index, video_path in rendered.items():
        assert len(_read_frames(video_path)) == plans[gap_index]["frame_count"]
    log = (tmp_path / "job" / "blender_service.log").read_text(encoding="utf-8", errors="replace")
    assert log.count("Blender quit") <= 1


def test_a_resumed_gap_renders_nothing_and_still_produces_the_video(tmp_path):
    plan = _plan()
    with _job(tmp_path) as job:
        job.render_gap(plan, tmp_path / "gap_00")
    with _job(tmp_path) as job:
        outcome = job.render_gap(plan, tmp_path / "gap_00", reuse_work=True)
    assert outcome.rendered_frame_count == 0
    assert len(_read_frames(outcome.video_path)) == plan["frame_count"]


def test_the_shell_reports_the_engine_it_actually_built(tmp_path):
    """EEVEE gets no shadow catcher; the summary must say so rather than imply one."""
    from infrastructure.blender_service import BlenderService, spawn_blender_process
    from infrastructure.json_files import write_json_file

    manifest = build_job_manifest(_plan(), FRAME_WIDTH, FRAME_HEIGHT)
    manifest_path = tmp_path / "job_manifest.json"
    write_json_file(manifest_path, manifest)
    service = BlenderService(
        process_factory=lambda: spawn_blender_process(
            _blender_executable(), PROJECT_ROOT / "backend" / "legacy" / "blender" / "service.py",
            tmp_path, PROJECT_ROOT,
        ),
        stall_timeout_seconds=STALL_TIMEOUT_SECONDS,
    )
    service.start()
    try:
        summary = service.open_job(manifest_path)
    finally:
        service.shutdown()
    assert summary["engine"] == "BLENDER_EEVEE_NEXT"
    assert summary["shadow_catcher"] is False


def test_a_gap_with_no_actors_renders_nothing_at_all(tmp_path):
    """Regression: the prebuilt templates were being rendered into every frame.

    Appending the asset library puts one object per class into the scene as a template
    to copy from. Hiding them in the viewport does not hide them from the camera, so a
    10.5 metre bus parked at the world origin was drawn into every layer and composited
    as a solid block over the plate.

    Rendering a gap with no actors is the sharpest possible statement of the invariant:
    if anything at all is opaque, something is in the scene that should not be.
    """
    from infrastructure.blender_service import BlenderService, spawn_blender_process
    from infrastructure.json_files import write_json_file

    plan = _plan()
    manifest = build_job_manifest(
        plan, FRAME_WIDTH, FRAME_HEIGHT, PROJECT_ROOT / "backend" / "assets" / "actors",
    )
    if not manifest.get("actor_library"):
        pytest.skip("The actor library is not built in this checkout")
    manifest_path = tmp_path / "job_manifest.json"
    write_json_file(manifest_path, manifest)
    empty_gap = tmp_path / "empty_gap.json"
    write_json_file(empty_gap, {"gap_index": 0, "frame_count": 1, "actors": []})

    service = BlenderService(
        process_factory=lambda: spawn_blender_process(
            _blender_executable(), PROJECT_ROOT / "backend" / "legacy" / "blender" / "service.py",
            tmp_path, PROJECT_ROOT,
        ),
        stall_timeout_seconds=STALL_TIMEOUT_SECONDS,
    )
    service.start()
    try:
        summary = service.open_job(manifest_path)
        assert summary["actor_library"]["loaded"] is True, "the library did not load"
        service.prepare_gap(0, empty_gap)
        service.render_frames(0, [1], tmp_path / "layers", regions=[[0.0, 0.0, 1.0, 1.0]])
    finally:
        service.shutdown()

    layer = cv2.imread(str(tmp_path / "layers" / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
    assert layer is not None and layer.shape[2] == 4
    assert layer[..., 3].max() == 0, (
        "an actorless gap rendered opaque pixels; something is visible to the camera "
        "that should not be"
    )
