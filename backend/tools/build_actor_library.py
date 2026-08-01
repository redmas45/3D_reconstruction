"""Builds the prebuilt actor asset library.

    python backend/tools/build_actor_library.py

Run this after changing `actor_proxies.CATALOG` or the humanoid skeleton. Until it is
run the renderer generates geometry procedurally, which produces identical output — the
library only decides when the cost is paid.

Exits non-zero if the build produced a library the loader would then reject, which is
the one failure worth catching here: a library nothing will ever use looks like success
and silently costs nothing but confusion.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from domain.actor_library import (  # noqa: E402
    LIBRARY_BLEND_NAME,
    catalog_digest,
    build_manifest,
    load_library,
    write_manifest,
)
from infrastructure.blender_runner import (  # noqa: E402
    BlenderUnavailableError,
    find_blender_executable,
)

MARKER = "@LIBRARY@"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "backend" / "assets" / "actors",
        help="Where library.blend and manifest.json are written",
    )
    return parser.parse_args()


def run_blender(output_directory: Path) -> dict:
    executable = find_blender_executable()
    completed = subprocess.run(
        [
            str(executable), "--background", "--factory-startup",
            "--python", str(PROJECT_ROOT / "backend" / "legacy" / "blender" / "build_library.py"),
            "--", "--output", str(output_directory / LIBRARY_BLEND_NAME),
        ],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise RuntimeError(
        f"Blender did not report a built library (exit {completed.returncode}).\n"
        f"{completed.stdout[-2000:]}"
    )


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_blender(arguments.output_directory)
    except BlenderUnavailableError as error:
        print(f"Cannot build the actor library: {error}", file=sys.stderr)
        return 2
    write_manifest(arguments.output_directory, build_manifest())
    library = load_library(arguments.output_directory)
    if library is None:
        print(
            "The library was written but the loader rejected it, so nothing would use "
            f"it. Expected digest {catalog_digest()}.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Built {len(report['built'])} actor assets "
        f"({report['bytes'] / 1024:.0f} KB, digest {library.digest})\n"
        f"  {library.blend_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
