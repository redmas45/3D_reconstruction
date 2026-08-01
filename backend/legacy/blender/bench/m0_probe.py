"""M0 benchmark probe — runs inside Blender, measures the v3 render strategy.

Answers the questions Implementation_plan.md §11 M0 blocks on:

  1. Does EEVEE Next actually render headless on this machine (Colab needs EGL)?
  2. How much faster is actor-only rendering than v2's full-scene rendering?
  3. How much more does ROI cropping buy on top of that?
  4. What does the first frame cost versus steady state (shader/kernel compilation)?

It writes one JSON report. It never imports project business logic — the Blender
process boundary in §3 applies to benchmarks too.

Invoked as:
    blender --background --factory-startup \
            --python blender/bench/m0_probe.py -- --output benchmarks/m0.json
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import bpy


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scene_fixture import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    ROI_FULL_FRAME_AREA_THRESHOLD,
    animate_actors,
    apply_render_region,
    build_actors,
    build_camera,
    build_environment,
    build_key_light,
    projected_actor_region,
    region_area_fraction,
    reset_scene,
)


REPORT_SCHEMA_VERSION = 1

FRAMES_PER_CONFIGURATION = 8
QUICK_FRAMES_PER_CONFIGURATION = 2
CYCLES_SAMPLE_SWEEP = (2, 8, 16, 32)
EEVEE_SAMPLE_COUNT = 16
EEVEE_ENGINE_CANDIDATES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
CYCLES_ENGINE = "CYCLES"
CYCLES_DEVICE_PREFERENCE = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")

# The go/no-go from §11: actor-only ROI must beat the v2 baseline by this factor.
REQUIRED_SPEEDUP_OVER_BASELINE = 5.0

BASELINE_CONFIGURATION_NAME = "v2_baseline_cycles_fullscene"


def _log(message: str) -> None:
    print(f"@M0@ {message}", flush=True)


# --------------------------------------------------------------------------
# Capability discovery
# --------------------------------------------------------------------------

def available_engines() -> list[str]:
    """Engines this build can actually select.

    The RNA enum lists only built-in engines, so Cycles — which registers as an
    add-on — is absent from it even when it renders perfectly well. Assignment is
    the only trustworthy probe, so every candidate is verified by trying it.
    """
    engine_property = bpy.types.RenderSettings.bl_rna.properties["engine"]
    candidates = dict.fromkeys(
        [item.identifier for item in engine_property.enum_items]
        + [CYCLES_ENGINE, *EEVEE_ENGINE_CANDIDATES]
    )
    render_settings = bpy.context.scene.render
    original_engine = render_settings.engine
    verified: list[str] = []
    for candidate in candidates:
        try:
            render_settings.engine = candidate
        except (TypeError, ValueError):
            continue
        verified.append(candidate)
    render_settings.engine = original_engine
    return verified


def preferred_eevee_engine(engines: list[str]) -> str | None:
    return next((name for name in EEVEE_ENGINE_CANDIDATES if name in engines), None)


def _select_compute_device_type(preferences: object) -> str | None:
    for candidate in CYCLES_DEVICE_PREFERENCE:
        try:
            preferences.compute_device_type = candidate
        except TypeError:
            continue
        preferences.get_devices()
        if any(device.type == candidate for device in preferences.devices):
            return candidate
    return None


def configure_cycles_device() -> dict:
    """Enable the best available Cycles compute backend. Returns what was chosen."""
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return {"compute_device_type": "NONE", "devices": [], "error": "cycles addon unavailable"}
    preferences = addon.preferences
    chosen_type = _select_compute_device_type(preferences)
    if chosen_type is None:
        return {"compute_device_type": "CPU", "devices": []}
    preferences.get_devices()
    enabled = []
    for device in preferences.devices:
        device.use = device.type == chosen_type
        if device.use:
            enabled.append(device.name)
    return {"compute_device_type": chosen_type, "devices": enabled}


def capability_report() -> dict:
    engines = available_engines()
    return {
        "blender_version": bpy.app.version_string,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "available_engines": engines,
        "eevee_engine": preferred_eevee_engine(engines),
        "cycles": configure_cycles_device(),
    }


# --------------------------------------------------------------------------
# Benchmark execution
# --------------------------------------------------------------------------

def _configure_engine(scene: bpy.types.Scene, engine: str, samples: int | None) -> None:
    scene.render.engine = engine
    if engine != CYCLES_ENGINE:
        if samples is not None and hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = samples
        return
    scene.cycles.device = "GPU"
    scene.cycles.use_denoising = True
    if samples is not None:
        scene.cycles.samples = samples


def _render_one_frame(scene: bpy.types.Scene, frame: int, output_directory: Path) -> float:
    scene.frame_set(frame)
    scene.render.filepath = str(output_directory / f"bench_{frame:04d}")
    started_at = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    return time.perf_counter() - started_at


def _timing_summary(
    name: str,
    engine: str,
    samples: int | None,
    region: tuple[float, float, float, float] | None,
    durations: list[float],
) -> dict:
    steady = sorted(durations[1:]) or durations
    return {
        "name": name,
        "engine": engine,
        "samples": samples,
        "status": "ok",
        "region": list(region) if region else None,
        "region_area_fraction": round(region_area_fraction(region), 4) if region else 1.0,
        "first_frame_seconds": round(durations[0], 4),
        "steady_median_seconds": round(steady[len(steady) // 2], 4),
        "total_seconds": round(sum(durations), 4),
        "frames": len(durations),
    }


def _run_frames(
    name: str,
    engine: str,
    samples: int | None,
    region: tuple[float, float, float, float] | None,
    output_directory: Path,
    frame_count: int,
) -> dict:
    scene = bpy.context.scene
    durations: list[float] = []
    try:
        for frame in range(1, frame_count + 1):
            durations.append(_render_one_frame(scene, frame, output_directory))
            _log(f"{name}: frame {frame}/{frame_count} {durations[-1]:.3f}s")
    except RuntimeError as error:
        return {"name": name, "engine": engine, "status": "failed", "error": str(error)}
    return _timing_summary(name, engine, samples, region, durations)


def measure_configuration(
    name: str,
    engine: str,
    samples: int | None,
    use_region: bool,
    actor_only: bool,
    project_root: Path,
    output_directory: Path,
    frame_count: int,
) -> dict:
    """One full cold-process measurement of a single render configuration."""
    scene = reset_scene()
    camera = build_camera(scene)
    build_key_light(scene)
    actors = build_actors(scene, project_root)
    if actor_only:
        scene.render.film_transparent = True
    else:
        build_environment(scene)
    animate_actors(actors, frame_count)
    _configure_engine(scene, engine, samples)
    region = projected_actor_region(scene, camera, actors) if use_region else None
    if region is not None and region_area_fraction(region) > ROI_FULL_FRAME_AREA_THRESHOLD:
        region = None
    apply_render_region(scene, region)
    return _run_frames(name, engine, samples, region, output_directory, frame_count)


def benchmark_configurations(capabilities: dict, quick: bool = False) -> list[tuple]:
    """(name, engine, samples, use_region, actor_only) for every runnable config."""
    eevee = capabilities["eevee_engine"]
    configurations: list[tuple] = [
        (BASELINE_CONFIGURATION_NAME, CYCLES_ENGINE, 2, False, False),
    ]
    sample_sweep = (CYCLES_SAMPLE_SWEEP[0],) if quick else CYCLES_SAMPLE_SWEEP
    for samples in sample_sweep:
        if not quick:
            configurations.append(
                (f"v3_cycles_{samples}s_fullframe", CYCLES_ENGINE, samples, False, True)
            )
        configurations.append((f"v3_cycles_{samples}s_roi", CYCLES_ENGINE, samples, True, True))
    if eevee is not None:
        if not quick:
            configurations.append(("v3_eevee_fullframe", eevee, EEVEE_SAMPLE_COUNT, False, True))
        configurations.append(("v3_eevee_roi", eevee, EEVEE_SAMPLE_COUNT, True, True))
    return configurations


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def _fastest_ok(results: list[dict], prefix: str) -> dict | None:
    candidates = [
        item for item in results
        if item["status"] == "ok" and item["name"].startswith(prefix)
    ]
    return min(candidates, key=lambda item: item["steady_median_seconds"]) if candidates else None


def build_verdict(results: list[dict]) -> dict:
    baseline = next(
        (
            item for item in results
            if item["name"] == BASELINE_CONFIGURATION_NAME and item["status"] == "ok"
        ),
        None,
    )
    best = _fastest_ok(results, "v3_")
    if baseline is None or best is None:
        return {
            "status": "inconclusive",
            "reason": "baseline or v3 configuration failed to render",
            "required_speedup": REQUIRED_SPEEDUP_OVER_BASELINE,
        }
    speedup = baseline["steady_median_seconds"] / max(best["steady_median_seconds"], 1e-6)
    return {
        "status": "go" if speedup >= REQUIRED_SPEEDUP_OVER_BASELINE else "no_go",
        "baseline_seconds_per_frame": baseline["steady_median_seconds"],
        "best_configuration": best["name"],
        "best_seconds_per_frame": best["steady_median_seconds"],
        "measured_speedup": round(speedup, 2),
        "required_speedup": REQUIRED_SPEEDUP_OVER_BASELINE,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="M0 render-strategy benchmark")
    parser.add_argument("--output", required=True, help="Path for the JSON report")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--label", default="local", help="Environment label, e.g. colab_t4")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: fewer frames and configurations, for validating the harness",
    )
    return parser.parse_args(argv)


def run_all_configurations(
    capabilities: dict, project_root: Path, frame_count: int, quick: bool,
) -> list[dict]:
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="m0_bench_") as temporary_directory:
        output_directory = Path(temporary_directory)
        for name, engine, samples, use_region, actor_only in benchmark_configurations(
            capabilities, quick,
        ):
            _log(f"running {name}")
            results.append(measure_configuration(
                name, engine, samples, use_region, actor_only,
                project_root, output_directory, frame_count,
            ))
    return results


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(arguments.project_root).resolve()
    frame_count = QUICK_FRAMES_PER_CONFIGURATION if arguments.quick else FRAMES_PER_CONFIGURATION
    capabilities = capability_report()
    _log(f"Blender {capabilities['blender_version']} engines={capabilities['available_engines']}")
    _log(f"Cycles device: {capabilities['cycles']}")
    results = run_all_configurations(capabilities, project_root, frame_count, arguments.quick)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": arguments.label,
        "quick": arguments.quick,
        "frame_dimensions": [FRAME_WIDTH, FRAME_HEIGHT],
        "frames_per_configuration": frame_count,
        "capabilities": capabilities,
        "results": results,
        "verdict": build_verdict(results),
    }
    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"report written to {output_path}")


if __name__ == "__main__":
    main()
