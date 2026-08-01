"""Runs the M0 render-strategy benchmark and reports its verdict.

Implementation_plan.md §11 M0 blocks every later milestone on real measurements
rather than the estimates in §4. This driver launches the in-Blender probe,
streams its progress, and prints the comparison table.

    python backend/tools/run_m0_benchmark.py --label colab_t4

Exits non-zero when the measured speedup misses the §11 go/no-go threshold, so
it can gate a notebook cell or CI step.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from infrastructure.blender_runner import BlenderUnavailableError, find_blender_executable

PROBE_SCRIPT = PROJECT_ROOT / "backend" / "legacy" / "blender" / "bench" / "m0_probe.py"
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "backend" / "benchmarks"
PROGRESS_MARKER = "@M0@"
BENCHMARK_TIMEOUT_SECONDS = 3600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M0 render-strategy benchmark")
    parser.add_argument("--label", default="local", help="Environment label, e.g. colab_t4")
    parser.add_argument("--output", default=None, help="Report path (default: backend/benchmarks/m0_<label>.json)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: fewer frames and configurations, for validating the harness",
    )
    return parser.parse_args()


def resolve_report_path(label: str, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).resolve()
    return DEFAULT_REPORT_DIRECTORY / f"m0_{label}.json"


def build_command(
    blender_executable: Path, report_path: Path, label: str, quick: bool = False,
) -> list[str]:
    command = [
        str(blender_executable),
        "--background",
        "--factory-startup",
        "--python", str(PROBE_SCRIPT),
        "--",
        "--output", str(report_path),
        "--project-root", str(PROJECT_ROOT),
        "--label", label,
    ]
    if quick:
        command.append("--quick")
    return command


def stream_probe(command: list[str]) -> int:
    """Run Blender, echoing only protocol lines so Cycles chatter stays out of the way."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith(PROGRESS_MARKER):
            print(line[len(PROGRESS_MARKER):].strip(), flush=True)
    return process.wait(timeout=BENCHMARK_TIMEOUT_SECONDS)


def _format_row(result: dict) -> str:
    if result["status"] != "ok":
        return f"  {result['name']:<34} FAILED  {result.get('error', '')[:60]}"
    area = result["region_area_fraction"]
    return (
        f"  {result['name']:<34}"
        f"{result['steady_median_seconds']:>9.3f} s"
        f"{result['first_frame_seconds']:>11.3f} s"
        f"{area * 100:>9.0f}%"
    )


def print_report(report: dict) -> None:
    capabilities = report["capabilities"]
    print("\n" + "=" * 78)
    print(f"M0 BENCHMARK — {report['label']}")
    print("=" * 78)
    print(f"Blender      : {capabilities['blender_version']}")
    print(f"Platform     : {capabilities['platform']}")
    print(f"EEVEE engine : {capabilities['eevee_engine'] or 'UNAVAILABLE'}")
    print(f"Cycles device: {capabilities['cycles'].get('compute_device_type')} "
          f"{capabilities['cycles'].get('devices')}")
    print("-" * 78)
    print(f"  {'configuration':<34}{'steady':>11}{'first frame':>13}{'roi':>10}")
    print("-" * 78)
    for result in report["results"]:
        print(_format_row(result))
    print("-" * 78)
    _print_verdict(report["verdict"])


def _print_verdict(verdict: dict) -> None:
    status = verdict["status"]
    if status == "inconclusive":
        print(f"VERDICT: INCONCLUSIVE — {verdict['reason']}")
        return
    print(f"  baseline (v2 full-scene) : {verdict['baseline_seconds_per_frame']:.3f} s/frame")
    print(f"  best v3 configuration    : {verdict['best_configuration']}")
    print(f"  best v3 timing           : {verdict['best_seconds_per_frame']:.3f} s/frame")
    print(f"  measured speedup         : {verdict['measured_speedup']}x "
          f"(need >= {verdict['required_speedup']}x)")
    print("=" * 78)
    print("VERDICT: GO — proceed to M1" if status == "go"
          else "VERDICT: NO-GO — stop and re-plan per Implementation_plan.md §11")


def main() -> int:
    arguments = parse_arguments()
    try:
        blender_executable = find_blender_executable()
    except BlenderUnavailableError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not PROBE_SCRIPT.is_file():
        print(f"error: probe script missing: {PROBE_SCRIPT}", file=sys.stderr)
        return 2
    report_path = resolve_report_path(arguments.label, arguments.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Blender: {blender_executable}")
    print(f"Report : {report_path}\n")
    return_code = stream_probe(
        build_command(blender_executable, report_path, arguments.label, arguments.quick)
    )
    if return_code != 0:
        print(f"error: Blender exited with code {return_code}", file=sys.stderr)
        return return_code
    if not report_path.is_file():
        print("error: benchmark produced no report", file=sys.stderr)
        return 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print_report(report)
    return 0 if report["verdict"]["status"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
