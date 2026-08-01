"""Start the single local backend that serves the browser application.

The application is intentionally one process: ``app.py`` serves the Three.js
frontend and exposes the local API on the same loopback origin. Keeping one
entrypoint prevents the old React/Vite and Blender launch paths from being
selected accidentally during a demo.

    python backend/tools/run_local.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    command = [sys.executable, str(PROJECT_ROOT / "app.py"), "--port", str(arguments.port)]
    if arguments.no_browser:
        command.append("--no-browser")
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
