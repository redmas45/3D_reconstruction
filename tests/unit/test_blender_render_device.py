import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BlenderRenderDeviceTests(unittest.TestCase):
    def test_cycles_enables_only_requested_gpu_backend(self) -> None:
        devices = [
            SimpleNamespace(name="Tesla T4", type="OPTIX", use=False),
            SimpleNamespace(name="Host CPU", type="CPU", use=True),
        ]
        render_device, preferences = _load_render_device(devices)
        scene = SimpleNamespace(cycles=SimpleNamespace())

        enabled_devices = render_device.configure_cycles_render(scene, {
            "cycles_compute_device": "OPTIX",
            "cycles_samples": 16,
            "cycles_use_denoising": True,
        })

        self.assertEqual(["Tesla T4"], enabled_devices)
        self.assertTrue(devices[0].use)
        self.assertFalse(devices[1].use)
        self.assertEqual("GPU", scene.cycles.device)
        self.assertEqual(16, scene.cycles.samples)
        preferences.refresh_devices.assert_called_once()

    def test_cycles_rejects_missing_requested_gpu(self) -> None:
        render_device, _ = _load_render_device([
            SimpleNamespace(name="Host CPU", type="CPU", use=True),
        ])

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            render_device.configure_cycles_render(
                SimpleNamespace(cycles=SimpleNamespace()),
                {"cycles_compute_device": "CUDA"},
            )


def _load_render_device(devices: list[SimpleNamespace]) -> tuple[types.ModuleType, SimpleNamespace]:
    from unittest.mock import MagicMock

    preferences = SimpleNamespace(
        compute_device_type="",
        devices=devices,
        refresh_devices=MagicMock(),
    )
    bpy_stub = types.ModuleType("bpy")
    bpy_stub.types = SimpleNamespace(Scene=object)
    bpy_stub.context = SimpleNamespace(
        preferences=SimpleNamespace(
            addons={"cycles": SimpleNamespace(preferences=preferences)},
        ),
    )
    module_path = PROJECT_ROOT / "blender" / "render_device.py"
    specification = importlib.util.spec_from_file_location("tested_render_device", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load Blender render device module")
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {"bpy": bpy_stub}):
        specification.loader.exec_module(module)
    return module, preferences


if __name__ == "__main__":
    unittest.main()
