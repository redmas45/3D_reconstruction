import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.blender_runner import BlenderRenderRequest, build_blender_command


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


if __name__ == "__main__":
    unittest.main()
