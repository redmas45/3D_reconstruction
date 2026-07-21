import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BlenderSceneBuilderTests(unittest.TestCase):
    def test_configure_render_uses_landscape_and_portrait_source_dimensions(self) -> None:
        scene_builder = _load_scene_builder()
        for width, height in ((1920, 1080), (720, 1280)):
            with self.subTest(width=width, height=height):
                scene = _empty_scene()
                plan = {
                    "fps": 30.0,
                    "render": {
                        "engine": "BLENDER_EEVEE_NEXT",
                        "preview_scale_percent": 75,
                        "source_width": width,
                        "source_height": height,
                    },
                }

                scene_builder.configure_render(scene, plan)

                self.assertEqual(width, scene.render.resolution_x)
                self.assertEqual(height, scene.render.resolution_y)

    def test_neutral_environment_skips_street_proxies(self) -> None:
        environment_builder = _load_environment_builder()
        environment_builder._grid_line = MagicMock()
        environment_builder._build_street_proxies = MagicMock()
        environment_builder.bpy.ops.mesh.primitive_plane_add = MagicMock()
        environment_builder.bpy.context.object = SimpleNamespace(
            name="", data=SimpleNamespace(materials=[]),
        )

        environment_builder.build_environment({"environment": {
            "ground_color": [0.0, 0.0, 0.0],
            "grid_color": [0.0, 0.5, 0.5],
            "proxy_profile": "neutral",
        }})

        environment_builder._build_street_proxies.assert_not_called()

    def test_workbench_render_uses_material_colors_and_depth_cues(self) -> None:
        scene_builder = _load_scene_builder()
        scene = _empty_scene()
        scene.display = SimpleNamespace(shading=SimpleNamespace())
        plan = {
            "fps": 30.0,
            "render": {
                "engine": "BLENDER_WORKBENCH",
                "preview_scale_percent": 75,
                "source_width": 1280,
                "source_height": 720,
            },
        }

        scene_builder.configure_render(scene, plan)

        self.assertEqual("MATERIAL", scene.display.shading.color_type)
        self.assertTrue(scene.display.shading.show_shadows)
        self.assertTrue(scene.display.shading.show_cavity)


def _load_scene_builder() -> types.ModuleType:
    stub_modules = {
        "bpy": _bpy_stub(),
        "mathutils": _module_with("mathutils", Vector=object),
        "animation": _module_with("animation", animate_entity=lambda *arguments: None),
        "environment_builder": _module_with(
            "environment_builder", build_environment=lambda plan: None, build_path_trail=lambda entity: None,
        ),
        "frame_rate": _module_with("frame_rate", blender_frame_rate=lambda fps: (round(fps), 1.0)),
        "hud": _module_with("hud", build_hud=lambda plan, camera: None),
        "human_builder": _module_with("human_builder", build_human=lambda entity: None),
        "render_device": _module_with(
            "render_device", CYCLES_RENDER_ENGINE="CYCLES", configure_cycles_render=lambda *arguments: [],
        ),
        "vehicle_builder": _module_with("vehicle_builder", build_vehicle=lambda entity: None),
    }
    module_path = PROJECT_ROOT / "blender" / "scene_builder.py"
    specification = importlib.util.spec_from_file_location("tested_scene_builder", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Blender scene builder for testing")
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, stub_modules):
        specification.loader.exec_module(module)
    return module


def _load_environment_builder() -> types.ModuleType:
    stub_modules = {
        "bpy": _bpy_stub(),
        "materials": _module_with("materials", create_material=lambda *arguments, **options: object()),
    }
    module_path = PROJECT_ROOT / "blender" / "environment_builder.py"
    specification = importlib.util.spec_from_file_location("tested_environment_builder", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Blender environment builder for testing")
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, stub_modules):
        specification.loader.exec_module(module)
    return module


def _bpy_stub() -> types.ModuleType:
    module = types.ModuleType("bpy")
    module.types = SimpleNamespace(Scene=object, Object=object, Material=object)
    module.ops = SimpleNamespace(mesh=SimpleNamespace())
    module.context = SimpleNamespace(object=None)
    return module


def _module_with(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _empty_scene() -> SimpleNamespace:
    render = SimpleNamespace(
        engine="",
        resolution_x=0,
        resolution_y=0,
        resolution_percentage=0,
        fps=0,
        fps_base=1.0,
        image_settings=SimpleNamespace(file_format=""),
        film_transparent=True,
        use_file_extension=False,
    )
    return SimpleNamespace(
        render=render,
        world=SimpleNamespace(color=None),
        view_settings=SimpleNamespace(look=""),
    )


if __name__ == "__main__":
    unittest.main()
