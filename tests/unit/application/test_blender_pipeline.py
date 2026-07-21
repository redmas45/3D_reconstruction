import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.blender_pipeline import _production_resolution, render_blender_gap
from domain.reconstruction_plan_v2 import build_reconstruction_plan_v2, write_reconstruction_plan_v2
from infrastructure.media_tools import MediaProcessingError, VideoContract


class BlenderPipelineCacheTests(unittest.TestCase):
    def test_scaled_cache_resolution_uses_encodable_even_dimensions(self) -> None:
        render_contract = {
            "source_width": 640,
            "source_height": 480,
            "production_scale_percent": 99,
        }

        self.assertEqual([634, 476], _production_resolution(render_contract))

    def test_matching_landscape_render_contract_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan_path = _write_plan(temporary_root / "plan.json", 1920, 1080)
            gap_directory = temporary_root / "gap"
            _write_cache(gap_directory, plan_path)

            with patch(
                "application.blender_pipeline.inspect_video_contract",
                return_value=VideoContract(1920, 1080, 29.97, 30),
            ):
                with patch("application.blender_pipeline.render_with_blender") as render_mock:
                    output_path = render_blender_gap(
                        PROJECT_ROOT, plan_path, gap_directory, reuse_render=True,
                    )

            self.assertEqual(gap_directory / "blender" / "gap_blender.mp4", output_path)
            render_mock.assert_not_called()

    def test_corrupt_cached_video_triggers_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan_path = _write_plan(temporary_root / "plan.json", 1920, 1080)
            gap_directory = temporary_root / "gap"
            _write_cache(gap_directory, plan_path)

            with patch(
                "application.blender_pipeline.inspect_video_contract",
                side_effect=MediaProcessingError("corrupt video"),
            ):
                with patch("application.blender_pipeline.render_with_blender") as render_mock:
                    render_blender_gap(PROJECT_ROOT, plan_path, gap_directory, reuse_render=True)

            render_mock.assert_called_once()

    def test_invalid_portrait_cache_is_rerendered(self) -> None:
        invalid_reports = {
            "stale hash": lambda report: {**report, "plan_hash": "0" * 64},
            "missing contract field": lambda report: {
                key: value for key, value in report.items() if key != "resolution"
            },
            "wrong resolution": lambda report: {**report, "resolution": [1280, 720]},
        }
        for case_name, alter_report in invalid_reports.items():
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary_directory:
                temporary_root = Path(temporary_directory)
                plan_path = _write_plan(temporary_root / "plan.json", 720, 1280)
                gap_directory = temporary_root / "gap"
                report_path = _write_cache(gap_directory, plan_path)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report_path.write_text(json.dumps(alter_report(report)), encoding="utf-8")

                with patch("application.blender_pipeline.render_with_blender") as render_mock:
                    render_blender_gap(PROJECT_ROOT, plan_path, gap_directory, reuse_render=True)

                render_mock.assert_called_once()

    def test_corrupt_cache_report_is_rerendered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            plan_path = _write_plan(temporary_root / "plan.json", 720, 1280)
            gap_directory = temporary_root / "gap"
            report_path = _write_cache(gap_directory, plan_path)
            report_path.write_text("{not-json", encoding="utf-8")

            with patch("application.blender_pipeline.render_with_blender") as render_mock:
                render_blender_gap(PROJECT_ROOT, plan_path, gap_directory, reuse_render=True)

            render_mock.assert_called_once()


def _write_plan(plan_path: Path, width: int, height: int) -> Path:
    scene_report = {
        "video": {"width": width, "height": height, "fps": 29.97, "frames": 300},
        "tracks": [],
        "camera_motion_report": {},
    }
    registry = {"schema_version": 1, "generator_version": "test", "identities": {}}
    plan = build_reconstruction_plan_v2(scene_report, registry, (30, 59), gap_index=0)
    write_reconstruction_plan_v2(plan, plan_path)
    return plan_path


def _write_cache(gap_directory: Path, plan_path: Path) -> Path:
    blender_directory = gap_directory / "blender"
    blender_directory.mkdir(parents=True)
    output_path = blender_directory / "gap_blender.mp4"
    report_path = blender_directory / "render_report.json"
    blend_path = blender_directory / "scene.blend"
    output_path.write_bytes(b"rendered-video")
    blend_path.write_bytes(b"blend-scene")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report_path.write_text(json.dumps(_matching_report(plan_path, plan)), encoding="utf-8")
    return report_path


def _matching_report(plan_path: Path, plan: dict) -> dict:
    render_contract = plan["render"]
    return {
        "plan_hash": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "mode": "animation",
        "render_engine": render_contract["engine"],
        "frame_count": plan["frame_count"],
        "resolution": _production_resolution(render_contract),
        "fps": plan["fps"],
    }


if __name__ == "__main__":
    unittest.main()
