import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BlenderLifecycleTests(unittest.TestCase):
    def test_continuous_entities_remain_solid(self) -> None:
        keyframes = _lifecycle_keyframes("continuous")

        self.assertEqual({1: 1.0, 120: 1.0}, keyframes)

    def test_entering_entities_fade_in_and_remain_solid(self) -> None:
        keyframes = _lifecycle_keyframes("enters")

        self.assertEqual(0.0, keyframes[1])
        self.assertEqual(1.0, keyframes[120])

    def test_exiting_entities_start_solid_and_fade_out(self) -> None:
        keyframes = _lifecycle_keyframes("exits")

        self.assertEqual(1.0, keyframes[1])
        self.assertEqual(0.0, keyframes[120])

    def test_walk_cycle_bends_knees_and_levels_feet(self) -> None:
        animation = _load_animation()
        legs = [_Control(), _Control()]
        knees = [_Control(), _Control()]
        feet = [_Control(), _Control()]

        animation._animate_legs(
            {
                "legs": legs,
                "knees": knees,
                "feet": feet,
                "leg_base_heights": [0.96, 0.96],
            },
            frame_index=10,
            phase=0.0,
        )

        self.assertGreater(knees[0].rotation_euler[0], 0.0)
        self.assertEqual(0.0, knees[1].rotation_euler[0])
        self.assertLess(feet[0].rotation_euler[0], 0.0)

    def test_billboard_vehicle_wheels_rotate_around_visible_axis(self) -> None:
        animation = _load_animation()
        wheels = [_Control(), _Control()]

        animation._animate_wheels(
            {
                "wheels": wheels,
                "wheel_radius": 0.2,
                "wheel_spin_axis": 1,
                "steering_wheels": [],
            },
            frame_index=8,
            distance_travelled=1.0,
            previous_heading=0.0,
            heading=0.0,
        )

        self.assertAlmostEqual(-5.0, wheels[0].rotation_euler[1])
        self.assertAlmostEqual(-5.0, wheels[1].rotation_euler[1])

    def test_every_sparse_render_frame_receives_an_animation_pose(self) -> None:
        animation = _load_animation()

        self.assertEqual([1, 2, 3, 4], animation._sample_frames(4))

    def test_stance_compensation_reduces_straight_leg_float(self) -> None:
        animation = _load_animation()

        compensation = animation._stance_vertical_compensation(
            swing=-0.46,
            knee_bend=0.0,
            height_scale=1.0,
        )

        self.assertGreater(compensation, 0.08)


class _AlphaInput:
    def __init__(self) -> None:
        self.default_value = 1.0
        self.keyframes: dict[int, float] = {}

    def keyframe_insert(self, _property_name: str, frame: int) -> None:
        self.keyframes[frame] = self.default_value


class _Control:
    def __init__(self) -> None:
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.location = SimpleNamespace(z=0.96)
        self.keyframes: list[tuple[str, int]] = []

    def keyframe_insert(self, property_name: str, frame: int) -> None:
        self.keyframes.append((property_name, frame))


def _lifecycle_keyframes(lifecycle: str) -> dict[int, float]:
    animation = _load_animation()
    alpha_input = _AlphaInput()
    shader = SimpleNamespace(inputs={"Alpha": alpha_input})
    material = SimpleNamespace(
        use_nodes=True,
        node_tree=SimpleNamespace(nodes={"Principled BSDF": shader}),
        surface_render_method="DITHERED",
    )
    animation._animate_lifecycle(
        {"materials": [material]},
        {"lifecycle": lifecycle},
        frame_count=120,
        fps=30.0,
    )
    return alpha_input.keyframes


def _load_animation() -> types.ModuleType:
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = object
    module_path = PROJECT_ROOT / "blender" / "animation.py"
    specification = importlib.util.spec_from_file_location(
        "tested_blender_animation",
        module_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Blender animation module")
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {"mathutils": mathutils}):
        specification.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
