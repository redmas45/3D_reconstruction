import argparse
import sys
import time
from pathlib import Path

import bpy


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from render_device import configure_cycles_render


BENCHMARK_WIDTH = 320
BENCHMARK_HEIGHT = 180
BENCHMARK_SAMPLES = 8


def parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Verify Blender Cycles GPU rendering")
    parser.add_argument("--device", choices=("OPTIX", "CUDA"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(arguments)


def build_benchmark_scene() -> bpy.types.Scene:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    bpy.ops.object.light_add(type="AREA", location=(2.0, -2.0, 4.0))
    bpy.context.object.data.energy = 900.0
    bpy.ops.object.camera_add(location=(3.5, -3.5, 2.5))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0.0, 0.78)
    scene.camera = camera
    return scene


def configure_benchmark(scene: bpy.types.Scene, device: str, output_path: Path) -> list[str]:
    scene.render.engine = "CYCLES"
    enabled_devices = configure_cycles_render(scene, {
        "cycles_compute_device": device,
        "cycles_samples": BENCHMARK_SAMPLES,
        "cycles_use_denoising": True,
    })
    scene.render.resolution_x = BENCHMARK_WIDTH
    scene.render.resolution_y = BENCHMARK_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)
    return enabled_devices


def main() -> None:
    arguments = parse_arguments()
    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = build_benchmark_scene()
    enabled_devices = configure_benchmark(scene, arguments.device, output_path)
    started_at = time.monotonic()
    bpy.ops.render.render(write_still=True)
    elapsed_seconds = time.monotonic() - started_at
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Cycles benchmark did not produce an image")
    print(
        f"CYCLES_GPU_READY device={arguments.device} "
        f"elapsed_seconds={elapsed_seconds:.3f} names={','.join(enabled_devices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
