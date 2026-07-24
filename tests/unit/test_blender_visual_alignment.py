import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BlenderVisualAlignmentTests(unittest.TestCase):
    def test_visual_scale_is_bounded_against_extreme_detections(self) -> None:
        module = _load_visual_alignment()

        self.assertEqual(3.0, module.bounded_visual_scale(0.30, 0.01))
        self.assertEqual(0.35, module.bounded_visual_scale(0.01, 0.30))

    def test_visual_scale_preserves_matching_projection(self) -> None:
        module = _load_visual_alignment()

        self.assertAlmostEqual(1.0, module.bounded_visual_scale(0.12, 0.12))


def _load_visual_alignment() -> types.ModuleType:
    bpy_module = types.ModuleType("bpy")
    bpy_module.types = SimpleNamespace(Scene=object, Object=object)
    extras_module = types.ModuleType("bpy_extras.object_utils")
    extras_module.world_to_camera_view = lambda *arguments: None
    mathutils_module = types.ModuleType("mathutils")
    mathutils_module.Vector = object
    module_path = PROJECT_ROOT / "blender" / "visual_alignment.py"
    specification = importlib.util.spec_from_file_location(
        "tested_visual_alignment", module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Blender visual alignment module")
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {
        "bpy": bpy_module,
        "bpy_extras.object_utils": extras_module,
        "mathutils": mathutils_module,
    }):
        specification.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
