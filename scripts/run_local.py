"""Starts the backend and the interface together.

Two servers are needed — uvicorn for the API and Vite for the interface — and running
them by hand in two terminals works fine. This exists so there is one command, and so
the dependency checks that would otherwise fail confusingly ten minutes into a render
happen up front.

    python scripts/run_local.py

Ctrl-C stops both.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

UI_DIRECTORY = PROJECT_ROOT / "ui"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
SHUTDOWN_GRACE_SECONDS = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-port", type=int, default=BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=FRONTEND_PORT)
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Start even if Blender, FFmpeg or the actor library are unavailable",
    )
    return parser.parse_args()


def check_environment() -> list[str]:
    """Report what is missing rather than failing at the first problem.

    An operator who is missing two things should learn both now, not one per attempt.
    """
    from domain.actor_library import library_state
    from infrastructure.blender_runner import BlenderUnavailableError, find_blender_executable
    from infrastructure.media_tools import MediaProcessingError, find_media_tool

    problems = []
    try:
        print(f"  Blender  {find_blender_executable()}")
    except BlenderUnavailableError as error:
        problems.append(f"Blender: {error}")
    for tool in ("ffmpeg", "ffprobe"):
        try:
            print(f"  {tool:<8} {find_media_tool(tool)}")
        except (MediaProcessingError, OSError, RuntimeError) as error:
            problems.append(f"{tool}: {error}")
    if not (UI_DIRECTORY / "node_modules").is_dir():
        problems.append(f"Interface dependencies are not installed. Run: cd ui && npm install")
    library = library_state(PROJECT_ROOT / "assets" / "actors")
    if library["available"]:
        print(f"  Models   {library['asset_count']} prebuilt ({library['catalog_digest']})")
    else:
        # Not fatal: geometry is generated at runtime instead, with identical output.
        print(
            f"  Models   not prebuilt ({library['reason']}) — will be generated per run.\n"
            f"           Build them with: python scripts/build_actor_library.py"
        )
    return problems


def start(command: list[str], cwd: Path, label: str) -> subprocess.Popen:
    print(f"  starting {label}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        # A new process group on POSIX so Ctrl-C reaches the children as a group and
        # neither server is left holding its port.
        start_new_session=os.name != "nt",
    )


def main() -> int:
    arguments = parse_arguments()
    print("Checking the environment:")
    problems = check_environment()
    if problems and not arguments.skip_checks:
        print("\nCannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nPass --skip-checks to start anyway.", file=sys.stderr)
        return 1

    print("\nStarting servers:")
    backend = start(
        [
            sys.executable, "-m", "uvicorn", "interfaces.api.app:app",
            "--app-dir", "src", "--host", "127.0.0.1",
            "--port", str(arguments.backend_port),
        ],
        PROJECT_ROOT, "backend",
    )
    frontend = start(
        ["npm.cmd" if os.name == "nt" else "npm", "run", "dev", "--",
         "--port", str(arguments.frontend_port)],
        UI_DIRECTORY, "interface",
    )
    print(
        f"\n  API        http://127.0.0.1:{arguments.backend_port}/api/health\n"
        f"  Interface  http://localhost:{arguments.frontend_port}/\n\n"
        "Ctrl-C to stop both."
    )
    try:
        while True:
            for process, label in ((backend, "backend"), (frontend, "interface")):
                if process.poll() is not None:
                    print(f"\nThe {label} exited with code {process.returncode}.", file=sys.stderr)
                    return process.returncode or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
        return 0
    finally:
        for process in (frontend, backend):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
